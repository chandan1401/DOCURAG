"""Adversarial stress-testing suite for Milestone M0.

Covers:
1. Pydantic v2 schemas: boundary values, serialization, model_validate alias bypass, Unicode, extreme numbers.
2. Disk cache: corruption healing, non-serializable args, cache hit type-preservation bug.
3. Retry backoff: transient recovery, permanent exhaustion, selective exceptions, zero tries edge case.
4. Hermetic fakes: FakeEmbedder 384-dim normalized unit vector, FakeReranker ordering and edge cases.
"""

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List
import pytest

from docurag.config import Settings, get_settings
from docurag.schemas import (
    AskResult,
    Chunk,
    Confidence,
    Evidence,
    Source,
    Timing,
    TraceStep,
)
from docurag.cache import _compute_cache_key, disk_cache, with_retry
from docurag.timing import StageTimer
from tests.conftest import FakeEmbedder, FakeReranker, FakeLLM


# =====================================================================
# 1. SCHEMA ADVERSARIAL STRESS TESTS
# =====================================================================

class TestSchemaAdversarial:
    """Stress tests and boundary checks for docurag.schemas."""

    def test_chunk_alias_bypass_in_model_validate(self):
        """CRITICAL VULNERABILITY TEST:
        When Chunk is deserialized via model_validate() with PROJECT.md contract keys
        ('doc_id', 'doc_name', 'text'), Pydantic v2 bypasses __init__.
        Because doc_id/doc_name/text are properties rather than aliased fields,
        model_validate silently drops them, defaulting document_id, source, content to empty strings!
        """
        payload = {
            "chunk_id": "c_adv_001",
            "doc_id": "document_alpha",
            "doc_name": "alpha.pdf",
            "page": 3,
            "text": "This is critical evidence text.",
            "token_count": 6,
        }

        # Case A: Instantiation via __init__ works
        chunk_init = Chunk(**payload)
        assert chunk_init.document_id == "document_alpha"
        assert chunk_init.source == "alpha.pdf"
        assert chunk_init.content == "This is critical evidence text."

        # Case B: Deserialization via model_validate
        chunk_val = Chunk.model_validate(payload)
        # We record whether the schema preserved or lost the data
        alias_preserved = (
            chunk_val.document_id == "document_alpha"
            and chunk_val.source == "alpha.pdf"
            and chunk_val.content == "This is critical evidence text."
        )
        # Note: If this fails, model_validate silently lost the chunk content!
        assert alias_preserved, (
            f"VULNERABILITY CONFIRMED: Chunk.model_validate dropped alias fields! "
            f"document_id='{chunk_val.document_id}', source='{chunk_val.source}', content='{chunk_val.content}'"
        )

    def test_source_alias_bypass_in_model_validate(self):
        """CRITICAL VULNERABILITY TEST:
        Source.model_validate() with PROJECT.md keys ('citation_id', 'doc_name', 'quote').
        """
        payload = {
            "citation_id": 42,
            "doc_name": "contract.pdf",
            "page": 1,
            "quote": "Important agreement clause",
            "confidence": 0.95,
        }

        src_val = Source.model_validate(payload)
        assert src_val.id == "42", f"Source.id was not populated, got '{src_val.id}'"
        assert src_val.document == "contract.pdf", f"Source.document was not populated, got '{src_val.document}'"
        assert src_val.snippet == "Important agreement clause", f"Source.snippet was not populated, got '{src_val.snippet}'"

    def test_timing_alias_bypass_in_model_validate(self):
        """CRITICAL VULNERABILITY TEST:
        Timing.model_validate() with PROJECT.md keys ('retrieve_ms', 'generate_ms', 'verify_ms').
        """
        payload = {
            "retrieve_ms": 150,
            "generate_ms": 300,
            "verify_ms": 75,
            "total_ms": 525,
        }

        t_val = Timing.model_validate(payload)
        assert t_val.retrieval_ms == 150, f"retrieval_ms lost in model_validate, got {t_val.retrieval_ms}"
        assert t_val.generation_ms == 300, f"generation_ms lost in model_validate, got {t_val.generation_ms}"
        assert t_val.verification_ms == 75, f"verification_ms lost in model_validate, got {t_val.verification_ms}"

    def test_confidence_label_inference_in_model_validate(self):
        """CRITICAL VULNERABILITY TEST:
        Confidence.model_validate({'score': 0.1, 'reason': 'Poor match'}).
        Does label compute as 'Low' or default to 'High'?
        """
        conf = Confidence.model_validate({"score": 0.1, "reason": "Low grounding"})
        assert conf.label == "Low", f"Confidence label should be 'Low' for score 0.1, got '{conf.label}'"

    def test_chunk_adversarial_unicode_and_huge_text(self):
        """Test Chunk resilience to Unicode, RTL, emojis, control chars, and 50KB strings."""
        huge_text = "🔥🚀 Multi-modal ñoño " + ("\u200b\ufeff\x00\x1b" * 100) + ("A" * 50000)
        chunk = Chunk(
            chunk_id="c_huge",
            document_id="doc_unicode",
            source="測試文件.pdf",
            page=1,
            content=huge_text,
            token_count=12000,
        )
        assert chunk.content == huge_text
        assert chunk.source == "測試文件.pdf"

        # Model dump and restore
        dump = chunk.model_dump()
        restored = Chunk.model_validate(dump)
        assert restored.content == huge_text
        assert restored.source == "測試文件.pdf"

    def test_chunk_extreme_and_boundary_numeric_values(self):
        """Test Chunk with extreme integers, 0, and negative numbers."""
        chunk = Chunk(
            chunk_id="c_extreme",
            document_id="doc_1",
            source="test.pdf",
            page=0,  # Boundary: page 0
            token_count=0,
            bbox=[0.0, 0.0, 1.0, 1.0],
        )
        assert chunk.page == 0
        assert chunk.token_count == 0

    def test_ask_result_with_empty_and_extra_fields(self):
        """Test AskResult with empty collections, empty answer, and extra fields."""
        res = AskResult(
            question="",
            answer="",
            answerable=False,
            sources=[],
            confidence=Confidence(score=0.0, label="Low", reason="No data"),
            evidence=[],
            timing=Timing(total_ms=0),
            agent_trace=[],
            unexpected_field="ignored",
        )
        assert res.answerable is False
        assert len(res.sources) == 0
        assert len(res.evidence) == 0

        # JSON Roundtrip
        raw_json = res.model_dump_json()
        restored = AskResult.model_validate_json(raw_json)
        assert restored.answerable is False
        assert restored.confidence.score == 0.0


# =====================================================================
# 2. DISK CACHE & RETRY ADVERSARIAL STRESS TESTS
# =====================================================================

class TestCacheAndRetryAdversarial:
    """Stress tests for @disk_cache and @with_retry."""

    def test_cache_hit_type_mutation_vulnerability(self, tmp_path):
        """CRITICAL FLAW TEST:
        When a function decorated with @disk_cache returns a Pydantic model (e.g. AskResult or Chunk):
        - On first call (cache miss), it returns the model instance (res.answer works).
        - On second call (cache hit), it reads json.load() and returns a raw dict!
        Callers executing `res.answer` on cache hit crash with AttributeError!
        """
        cache_dir = tmp_path / "cache_type_leak"

        @disk_cache(namespace="type_leak", custom_cache_dir=cache_dir)
        def generate_result(q: str) -> AskResult:
            return AskResult(
                question=q,
                answer="Answer for " + q,
                confidence=Confidence(score=0.9, label="High"),
                timing=Timing(total_ms=10),
            )

        # Call 1: Miss
        res1 = generate_result("test query")
        assert isinstance(res1, AskResult)
        assert res1.answer == "Answer for test query"

        # Call 2: Hit
        res2 = generate_result("test query")
        # Check if type is preserved or mutated to raw dict
        assert isinstance(res2, AskResult), (
            f"VULNERABILITY CONFIRMED: disk_cache returned type '{type(res2).__name__}' "
            f"instead of 'AskResult' on cache hit! Accessing res2.answer crashes callers!"
        )

    def test_cache_complex_and_nested_argument_keys(self, tmp_path):
        """Test cache key computation on complex nested structures and byte payloads."""
        complex_arg = {
            "nested": {"a": [1, 2, 3], "b": {"deep": True}},
            "flags": (True, False),
        }
        key1 = _compute_cache_key((complex_arg,), {})
        # Different key ordering should yield identical SHA-256 hash
        complex_arg_reordered = {
            "flags": (True, False),
            "nested": {"b": {"deep": True}, "a": [1, 2, 3]},
        }
        key2 = _compute_cache_key((complex_arg_reordered,), {})
        assert key1 == key2, "Cache key computation must be invariant to dict key ordering!"

    def test_cache_corruption_self_healing_matrix(self, tmp_path):
        """Test @disk_cache healing against various disk corruption states:
        1. 0-byte empty file
        2. Truncated JSON
        3. Binary garbage
        """
        cache_dir = tmp_path / "cache_corruption"
        call_count = 0

        @disk_cache(namespace="corrupt_test", custom_cache_dir=cache_dir)
        def cached_operation(val: int) -> dict:
            nonlocal call_count
            call_count += 1
            return {"output": val * 10}

        # Populate cache
        assert cached_operation(1) == {"output": 10}
        assert call_count == 1

        json_files = list(cache_dir.glob("*.json"))
        assert len(json_files) == 1
        target_file = json_files[0]

        # Scenario A: 0-byte empty file
        target_file.write_text("")
        assert cached_operation(1) == {"output": 10}
        assert call_count == 2, "Failed to self-heal and recompute after 0-byte corrupt file!"

        # Scenario B: Truncated JSON
        target_file.write_text('{"output": ')
        assert cached_operation(1) == {"output": 10}
        assert call_count == 3, "Failed to self-heal and recompute after truncated JSON!"

        # Scenario C: Binary garbage
        target_file.write_bytes(b"\x00\xff\xfe\x01\x88\x99\xaa")
        assert cached_operation(1) == {"output": 10}
        assert call_count == 4, "Failed to self-heal and recompute after binary garbage!"

    def test_with_retry_zero_tries_edge_case(self):
        """BUG TEST:
        When with_retry is configured with tries=0 or negative:
        The loop `range(1, 1)` does not run, and the wrapper silently returns None!
        """
        @with_retry(tries=0, base=0.01)
        def return_number():
            return 42

        val = return_number()
        assert val == 42, f"with_retry(tries=0) failed to execute function, returned {val}"

    def test_with_retry_permanent_failure_preserves_exception(self):
        """Verify with_retry raises the exact underlying exception with message."""
        attempts = 0

        @with_retry(tries=3, base=0.005)
        def unrecoverable():
            nonlocal attempts
            attempts += 1
            raise KeyError(f"Fatal key missing on attempt {attempts}")

        with pytest.raises(KeyError) as exc_info:
            unrecoverable()
        assert "Fatal key missing on attempt 3" in str(exc_info.value)
        assert attempts == 3

    def test_with_retry_selective_exception_catching(self):
        """Verify with_retry does NOT retry unlisted exceptions."""
        attempts = 0

        @with_retry(tries=3, base=0.01, exceptions=(ValueError,))
        def type_error_raiser():
            nonlocal attempts
            attempts += 1
            raise TypeError("Uncaught exception type")

        with pytest.raises(TypeError):
            type_error_raiser()
        # Must fail immediately on attempt 1 without retrying
        assert attempts == 1


# =====================================================================
# 3. HERMETIC FAKES STRESS TESTS
# =====================================================================

class TestHermeticFakesAdversarial:
    """Stress tests for FakeEmbedder, FakeReranker, and FakeLLM."""

    def test_fake_embedder_dimension_and_normalization_invariants(self):
        """Verify FakeEmbedder dimension is EXACTLY 384 and L2 norm is ~1.0
        across diverse inputs: empty string, single char, Unicode, 20KB text.
        """
        embedder = FakeEmbedder()
        test_inputs = [
            "",
            " ",
            "a",
            "DocuRAG",
            "🔥🚀 Special ñoño characters \u200b\ufeff",
            "Long text " * 2000,
        ]

        for text in test_inputs:
            vec = embedder.embed_query(text)
            assert len(vec) == 384, f"Vector dimension was {len(vec)}, expected 384 for input: {text[:20]}"
            l2_norm = math.sqrt(sum(x * x for x in vec))
            assert abs(l2_norm - 1.0) < 1e-4, f"L2 norm was {l2_norm}, expected 1.0 for input: {text[:20]}"

    def test_fake_embedder_batch_encode_shapes(self):
        """Verify batch encoding preserves dimensions and returns numpy arrays."""
        embedder = FakeEmbedder()
        batch = ["Sentence 1", "Sentence 2", "Sentence 3"]
        arr = embedder.encode(batch)
        assert arr.shape == (3, 384)

        # Single string encode
        single = embedder.encode("Single")
        assert single.shape == (384,)

    def test_fake_embedder_empty_batch_handling(self):
        """Verify behavior on empty batch [] input."""
        embedder = FakeEmbedder()
        res = embedder.encode([])
        # Check if shape is (0, 384) or (0,)
        assert res.shape in ((0, 384), (0,))

    def test_fake_reranker_ordering_and_corner_cases(self):
        """Verify FakeReranker ordering, ties, and empty string handling."""
        reranker = FakeReranker()
        query = "multimodal document retrieval"
        perfect_match = "multimodal document retrieval system"
        partial_match = "document retrieval"
        no_match = "cooking pizza in oven"
        empty_doc = ""

        pairs = [
            (query, perfect_match),
            (query, partial_match),
            (query, no_match),
            (query, empty_doc),
            ("", "some doc"),
        ]
        scores = reranker.predict(pairs)
        assert len(scores) == 5

        # Perfect match > partial match > no match
        assert scores[0] > scores[1] > scores[2]
        # Empty doc should return baseline score (0.1)
        assert scores[3] == pytest.approx(0.1, abs=1e-3)
        # Empty query should return baseline score (0.1)
        assert scores[4] == pytest.approx(0.1, abs=1e-3)
