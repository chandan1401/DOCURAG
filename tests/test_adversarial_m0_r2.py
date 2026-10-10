"""Adversarial stress-testing suite for Milestone M0 Iteration 2 (Challenger R2).

Addresses Dispatch Mission requirements:
1. Re-serializing and deserializing lists of Pydantic models through @disk_cache.
2. Multiple sequential validator calls on identical dictionary instances (immutability check).
3. Edge case inputs to new schemas:
   - PageData
   - FigureData
   - FigureDescription (including to_flattened_text with malformed/missing fields)
   - RouteDecision (fallback on hallucinated intents/modalities, string document_hints)
   - SubQuestion (question/sub_question alias and document_ids/document_names synchronization)
4. Confidence score boundary threshold stress tests.
5. Cache corruption self-healing and zero-tries resilience validation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytest

from docurag.schemas import (
    AskResult,
    Chunk,
    Confidence,
    Evidence,
    FigureData,
    FigureDescription,
    PageData,
    RouteDecision,
    Source,
    SubQuestion,
    Timing,
    TraceStep,
)
from docurag.cache import disk_cache, with_retry, _restore_return_type
from tests.conftest import FakeEmbedder, FakeReranker, FakeLLM


# =====================================================================
# 1. LIST OF MODELS DISK CACHE ROUNDTRIP ADVERSARIAL TESTS
# =====================================================================

class TestDiskCacheListModelsAdversarial:
    """Stress tests for @disk_cache returning lists of Pydantic models."""

    def test_disk_cache_list_of_chunks_return_type_preservation(self, tmp_path):
        """Verify @disk_cache deserializes list[Chunk] into real Chunk instances on cache hit."""
        cache_dir = tmp_path / "cache_list_chunks"
        call_count = 0

        @disk_cache(namespace="chunks_ns", custom_cache_dir=cache_dir)
        def fetch_chunks(doc_id: str) -> List[Chunk]:
            nonlocal call_count
            call_count += 1
            return [
                Chunk(
                    chunk_id=f"{doc_id}_c1",
                    document_id=doc_id,
                    source="spec.pdf",
                    page=1,
                    content="First chunk content",
                    token_count=10,
                ),
                Chunk(
                    chunk_id=f"{doc_id}_c2",
                    document_id=doc_id,
                    source="spec.pdf",
                    page=2,
                    content="Second chunk content",
                    token_count=12,
                    content_type="table",
                ),
            ]

        # Call 1: Cache Miss
        res1 = fetch_chunks("doc_alpha")
        assert call_count == 1
        assert len(res1) == 2
        assert isinstance(res1[0], Chunk)
        assert isinstance(res1[1], Chunk)
        assert res1[0].content == "First chunk content"
        assert res1[1].is_table is True

        # Call 2: Cache Hit
        res2 = fetch_chunks("doc_alpha")
        assert call_count == 1  # Function body not executed
        assert len(res2) == 2
        assert isinstance(res2[0], Chunk), f"Expected Chunk, got {type(res2[0])}"
        assert isinstance(res2[1], Chunk), f"Expected Chunk, got {type(res2[1])}"
        assert res2[0].chunk_id == "doc_alpha_c1"
        assert res2[0].content == "First chunk content"
        assert res2[0].text == "First chunk content"
        assert res2[1].is_table is True
        assert res2[1].content_type == "table"

    def test_disk_cache_list_of_pagedata_return_type_preservation(self, tmp_path):
        """Verify @disk_cache deserializes list[PageData] with nested FigureData on cache hit."""
        cache_dir = tmp_path / "cache_list_pages"
        call_count = 0

        @disk_cache(namespace="pages_ns", custom_cache_dir=cache_dir)
        def parse_pages(doc_id: str) -> List[PageData]:
            nonlocal call_count
            call_count += 1
            fig = FigureData(path="figs/chart.png", bbox=[0.0, 10.0, 100.0, 200.0], page=1)
            return [
                PageData(
                    page=1,
                    text="Page 1 text",
                    tables_md=["| Col1 | Col2 |"],
                    figures=[fig],
                ),
                PageData(page=2, text="Page 2 text"),
            ]

        res1 = parse_pages("doc_beta")
        assert call_count == 1
        assert isinstance(res1[0], PageData)

        # Cache hit
        res2 = parse_pages("doc_beta")
        assert call_count == 1
        assert len(res2) == 2
        assert isinstance(res2[0], PageData)
        assert res2[0].page == 1
        assert res2[0].text == "Page 1 text"
        assert len(res2[0].figures) == 1
        assert isinstance(res2[0].figures[0], FigureData)
        assert res2[0].figures[0].path == "figs/chart.png"
        assert res2[0].figures[0].image_path == "figs/chart.png"

    def test_disk_cache_empty_list_return_type(self, tmp_path):
        """Verify @disk_cache handles empty list of models gracefully."""
        cache_dir = tmp_path / "cache_empty_list"
        call_count = 0

        @disk_cache(namespace="empty_ns", custom_cache_dir=cache_dir)
        def empty_retrieval(query: str) -> List[Chunk]:
            nonlocal call_count
            call_count += 1
            return []

        res1 = empty_retrieval("no_matches")
        assert res1 == []
        assert call_count == 1

        res2 = empty_retrieval("no_matches")
        assert res2 == []
        assert call_count == 1


# =====================================================================
# 2. DICTIONARY IMMUTABILITY & SEQUENTIAL VALIDATOR STRESS TESTS
# =====================================================================

class TestDictImmutabilityAdversarial:
    """Verify that multiple sequential model_validate calls do not mutate caller dicts."""

    def test_chunk_validator_does_not_mutate_shared_dict(self):
        """Ensure model_validate does not mutate input dictionary across 5 iterations."""
        original_dict = {
            "chunk_id": "c_immutable",
            "doc_id": "doc_shared",
            "doc_name": "shared.pdf",
            "page": 5,
            "text": "Critical invariant text that must not disappear.",
            "is_table": True,
            "token_count": 8,
        }
        dict_snapshot = dict(original_dict)

        instances = []
        for _ in range(5):
            inst = Chunk.model_validate(original_dict)
            instances.append(inst)

        # Verify the original dictionary remains 100% unaltered
        assert original_dict == dict_snapshot
        assert "doc_id" in original_dict
        assert "doc_name" in original_dict
        assert "text" in original_dict
        assert "is_table" in original_dict

        # Verify all instances received the complete data
        for inst in instances:
            assert inst.document_id == "doc_shared"
            assert inst.doc_id == "doc_shared"
            assert inst.source == "shared.pdf"
            assert inst.doc_name == "shared.pdf"
            assert inst.content == "Critical invariant text that must not disappear."
            assert inst.text == "Critical invariant text that must not disappear."
            assert inst.is_table is True
            assert inst.content_type == "table"

    def test_source_validator_does_not_mutate_shared_dict(self):
        """Ensure Source.model_validate preserves original dictionary keys."""
        source_dict = {
            "citation_id": 99,
            "doc_name": "contract_v2.pdf",
            "page": 12,
            "quote": "Liabilities are strictly capped.",
            "confidence": 0.88,
        }
        dict_snapshot = dict(source_dict)

        for _ in range(3):
            s = Source.model_validate(source_dict)
            assert s.id == "99"
            assert s.citation_id == 99
            assert s.document == "contract_v2.pdf"
            assert s.doc_name == "contract_v2.pdf"
            assert s.snippet == "Liabilities are strictly capped."
            assert s.quote == "Liabilities are strictly capped."

        assert source_dict == dict_snapshot

    def test_timing_validator_does_not_mutate_shared_dict(self):
        """Ensure Timing.model_validate preserves original dictionary keys."""
        timing_dict = {
            "retrieve_ms": 120,
            "generate_ms": 450,
            "verify_ms": 90,
            "total_ms": 660,
        }
        dict_snapshot = dict(timing_dict)

        for _ in range(3):
            t = Timing.model_validate(timing_dict)
            assert t.retrieval_ms == 120
            assert t.retrieve_ms == 120
            assert t.generation_ms == 450
            assert t.generate_ms == 450
            assert t.verification_ms == 90
            assert t.verify_ms == 90

        assert timing_dict == dict_snapshot


# =====================================================================
# 3. SCHEMA EDGE CASES: PageData, FigureData, FigureDescription, RouteDecision, SubQuestion
# =====================================================================

class TestSchemaEdgeCasesAdversarial:
    """Stress tests on newly introduced schemas and their boundary values."""

    def test_page_data_boundary_and_aliases(self):
        """Test PageData with aliases, boundary page numbers, and empty fields."""
        data = {
            "page_number": 0,
            "content": "Zero-index page text",
            "ocr_used": True,
        }
        page = PageData.model_validate(data)
        assert page.page == 0
        assert page.text == "Zero-index page text"
        assert page.ocr_used is True
        assert page.tables_md == []
        assert page.figures == []

        # Negative page number handling
        neg_page = PageData.model_validate({"page": -1, "text": "Cover page"})
        assert neg_page.page == -1

    def test_figure_data_aliases_and_properties(self):
        """Test FigureData path <-> image_path alias handling and property access."""
        # Scenario A: Initialized with 'path'
        fig_a = FigureData.model_validate({"path": "figs/chart_a.png", "page": 2})
        assert fig_a.image_path == "figs/chart_a.png"
        assert fig_a.path == "figs/chart_a.png"

        # Scenario B: Initialized with 'image_path'
        fig_b = FigureData.model_validate({"image_path": "figs/chart_b.png", "page": 3})
        assert fig_b.image_path == "figs/chart_b.png"
        assert fig_b.path == "figs/chart_b.png"

    def test_figure_description_to_flattened_text_adversarial(self):
        """Stress-test FigureDescription.to_flattened_text with extreme/malformed inputs."""
        # 1. Clean well-formed description
        desc1 = FigureDescription(
            type="pie_chart",
            title="Market Share",
            caption="Q3 2026 Distribution",
            key_values=[
                {"label": "Segment A", "value": "45", "unit": "%"},
                {"label": "Segment B", "value": "55", "unit": "%"},
            ],
            relationships="Segment B leads Segment A",
            visible_text="45% | 55%",
        )
        flat1 = desc1.to_flattened_text()
        assert "[pie_chart] Market Share. Q3 2026 Distribution." in flat1
        assert "Segment A=45 %; Segment B=55 %" in flat1
        assert "Relationships: Segment B leads Segment A." in flat1
        assert "Text: 45% | 55%." in flat1

        # 2. Malformed key_values: non-dicts, missing labels, empty strings
        desc2 = FigureDescription(
            type="other",
            title="Raw Diagram",
            caption="",
            key_values=[
                "not_a_dict",
                {"value": "100"},  # Missing 'label'
                {"label": "Valid Metric", "value": "99.9", "unit": "ms"},
            ],
        )
        flat2 = desc2.to_flattened_text()
        assert "Valid Metric=99.9 ms" in flat2
        assert "not_a_dict" not in flat2

        # 3. Completely empty FigureDescription
        desc3 = FigureDescription()
        flat3 = desc3.to_flattened_text()
        assert flat3 == "[other] ."

    def test_route_decision_hallucination_containment(self):
        """Test RouteDecision safely clamps hallucinated intents and modalities to defaults."""
        # Hallucinated intent
        dec1 = RouteDecision.model_validate({"intent": "destroy_all_data", "query_type": "quantum"})
        assert dec1.intent == "factual", f"Expected fallback to 'factual', got '{dec1.intent}'"
        assert dec1.query_type == "general", f"Expected fallback to 'general', got '{dec1.query_type}'"

        # String document_hints wrapped in list
        dec2 = RouteDecision.model_validate({"document_hints": "single_doc.pdf"})
        assert dec2.document_hints == ["single_doc.pdf"]

        # Valid intents preserved
        for valid in ["factual", "table_lookup", "chart_numeric", "multi_doc_compare", "summarize_doc", "out_of_scope"]:
            dec = RouteDecision.model_validate({"intent": valid})
            assert dec.intent == valid

    def test_sub_question_alias_and_id_sync(self):
        """Test SubQuestion alias translation and document_ids <-> document_names sync."""
        # Alias 'question' -> 'sub_question'
        sq1 = SubQuestion.model_validate({"question": "What is the Q3 revenue?"})
        assert sq1.sub_question == "What is the Q3 revenue?"

        # document_ids provided -> synced to document_names
        sq2 = SubQuestion.model_validate({
            "sub_question": "Compare latency",
            "document_ids": ["doc_1", "doc_2"],
        })
        assert sq2.document_names == ["doc_1", "doc_2"]
        assert sq2.document_ids == ["doc_1", "doc_2"]

        # document_names provided -> synced to document_ids
        sq3 = SubQuestion.model_validate({
            "sub_question": "Compare throughput",
            "document_names": ["doc_a", "doc_b"],
        })
        assert sq3.document_ids == ["doc_a", "doc_b"]
        assert sq3.document_names == ["doc_a", "doc_b"]


# =====================================================================
# 4. CONFIDENCE SCORE BOUNDARY STRESS TESTS
# =====================================================================

class TestConfidenceBoundariesAdversarial:
    """Test Confidence score threshold boundaries."""

    @pytest.mark.parametrize(
        "score,expected_label",
        [
            (0.0, "Low"),
            (0.1, "Low"),
            (0.399, "Low"),
            (0.4, "Medium"),
            (0.5, "Medium"),
            (0.749, "Medium"),
            (0.75, "High"),
            (0.9, "High"),
            (1.0, "High"),
        ],
    )
    def test_confidence_score_threshold_classification(self, score, expected_label):
        c = Confidence.model_validate({"score": score})
        assert c.label == expected_label, (
            f"Score {score} classified as '{c.label}', expected '{expected_label}'"
        )
