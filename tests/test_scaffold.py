"""Comprehensive unit tests for Milestone M0 scaffolding, config, schemas, cache, timing, and pipeline."""

import json
import os
import time
from pathlib import Path
import pytest

from docurag.config import Settings, get_settings
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
from docurag.cache import disk_cache, with_retry
from docurag.timing import StageTimer
from docurag.pipeline import ask, _ask_linear, ingest_file, ingest_directory
from tests.conftest import FakeEmbedder, FakeReranker, FakeLLM


class TestConfig:
    """Tests for configuration dataclass, environment variables, and directories."""

    def test_default_config_loading(self):
        settings = Settings()
        assert settings.DOCURAG_STORAGE == "./storage"
        assert settings.EMBED_MODEL == "BAAI/bge-small-en-v1.5"
        assert settings.RERANK_MODEL == "BAAI/bge-reranker-base"
        assert settings.OCR_MIN_CHARS == 50
        assert settings.RENDER_DPI == 200
        assert settings.CHUNK_TOKENS == 400
        assert settings.CHUNK_OVERLAP == 50
        assert settings.DENSE_K == 30
        assert settings.BM25_K == 30
        assert settings.RERANK_IN == 20
        assert settings.FINAL_K == 6
        assert settings.RRF_K == 60
        assert settings.SUPPORT_THRESHOLD == 0.7
        assert settings.USE_AGENTS is False
        assert settings.MCP_MODE == "direct"

    def test_environment_variable_override(self, monkeypatch):
        monkeypatch.setenv("OCR_MIN_CHARS", "100")
        monkeypatch.setenv("USE_AGENTS", "1")
        monkeypatch.setenv("FINAL_K", "10")
        monkeypatch.setenv("DOCURAG_FAKES", "true")

        settings = Settings()
        assert settings.OCR_MIN_CHARS == 100
        assert settings.USE_AGENTS is True
        assert settings.FINAL_K == 10
        assert settings.DOCURAG_FAKES is True

    def test_storage_directories_resolution_and_creation(self, tmp_path):
        custom_storage = tmp_path / "custom_storage"
        settings = Settings(DOCURAG_STORAGE=str(custom_storage))
        assert settings.storage_dir == custom_storage
        assert settings.uploads_dir == custom_storage / "uploads"
        assert settings.pages_dir == custom_storage / "pages"
        assert settings.figs_dir == custom_storage / "figs"
        assert settings.index_dir == custom_storage / "index"
        assert settings.cache_dir == custom_storage / "cache"

        settings.ensure_dirs()
        assert settings.uploads_dir.exists()
        assert settings.pages_dir.exists()
        assert settings.figs_dir.exists()
        assert settings.index_dir.exists()
        assert settings.cache_dir.exists()


class TestSchemas:
    """Tests for Pydantic v2 schemas and compatibility property accessors."""

    def test_chunk_canonical_and_compatibility_properties(self):
        chunk = Chunk(
            chunk_id="c_100",
            document_id="doc_abc",
            source="manual.pdf",
            page=2,
            content_type="table",
            content="Table of parameters",
            token_count=15,
        )
        assert chunk.chunk_id == "c_100"
        assert chunk.document_id == "doc_abc"
        assert chunk.doc_id == "doc_abc"
        assert chunk.source == "manual.pdf"
        assert chunk.doc_name == "manual.pdf"
        assert chunk.text == "Table of parameters"
        assert chunk.is_table is True
        assert chunk.is_figure is False

    def test_chunk_alias_initialization(self):
        chunk = Chunk(
            chunk_id="c_200",
            doc_id="doc_xyz",
            doc_name="spec.pdf",
            page=4,
            text="Specification text",
            is_figure=True,
        )
        assert chunk.document_id == "doc_xyz"
        assert chunk.source == "spec.pdf"
        assert chunk.content == "Specification text"
        assert chunk.content_type == "image"
        assert chunk.is_figure is True

    def test_source_and_evidence_serialization(self, sample_chunks):
        c = sample_chunks[0]
        ev = Evidence(
            eid="E1",
            chunk=c,
            score=0.92,
            norm_score=0.9,
            dense_score=0.85,
            sparse_score=11.2,
            rerank_score=0.92,
            combined_score=0.89,
        )
        src = Source(
            id="1",
            document=c.source,
            page=c.page,
            type=c.content_type,
            snippet=c.content,
            score=0.92,
            confidence=0.9,
        )
        assert src.citation_id == 1
        assert src.doc_name == "architecture.pdf"
        assert src.quote == c.content

        ev_dump = ev.model_dump()
        assert ev_dump["eid"] == "E1"
        assert ev_dump["chunk"]["chunk_id"] == "c_001"

    def test_ask_result_json_roundtrip(self, sample_ask_result):
        json_str = sample_ask_result.model_dump_json()
        assert "DocuRAG" in json_str
        assert "E1" in json_str

        data = json.loads(json_str)
        restored = AskResult.model_validate(data)
        assert restored.question == sample_ask_result.question
        assert restored.answer == sample_ask_result.answer
        assert len(restored.sources) == len(sample_ask_result.sources)
        assert restored.confidence.score == sample_ask_result.confidence.score
        assert len(restored.agent_trace) == 1
        assert restored.agent_trace[0].agent == "router"


class TestResilience:
    """Tests for disk caching and retry decorators."""

    def test_disk_cache_hit_and_miss(self, tmp_path):
        cache_dir = tmp_path / "cache"
        invocation_count = 0

        @disk_cache(namespace="test_ns", custom_cache_dir=cache_dir)
        def expensive_calc(a: int, b: str):
            nonlocal invocation_count
            invocation_count += 1
            return {"result": a * 2, "label": b}

        # First call: cache miss
        res1 = expensive_calc(5, "items")
        assert res1 == {"result": 10, "label": "items"}
        assert invocation_count == 1

        # Second call: cache hit (function body not executed)
        res2 = expensive_calc(5, "items")
        assert res2 == {"result": 10, "label": "items"}
        assert invocation_count == 1

        # Different argument: cache miss
        res3 = expensive_calc(6, "items")
        assert res3 == {"result": 12, "label": "items"}
        assert invocation_count == 2

    def test_disk_cache_bytes_argument(self, tmp_path):
        cache_dir = tmp_path / "cache_bytes"
        calls = 0

        @disk_cache(namespace="bytes_ns", custom_cache_dir=cache_dir)
        def hash_bytes(payload: bytes):
            nonlocal calls
            calls += 1
            return {"size": len(payload)}

        data = b"\x00\x01\x02\x03\x04"
        r1 = hash_bytes(data)
        assert r1["size"] == 5
        assert calls == 1

        r2 = hash_bytes(data)
        assert r2["size"] == 5
        assert calls == 1

    def test_disk_cache_corrupt_file_recovery(self, tmp_path):
        cache_dir = tmp_path / "cache_corrupt"
        calls = 0

        @disk_cache(namespace="corrupt_ns", custom_cache_dir=cache_dir)
        def robust_call(x: int):
            nonlocal calls
            calls += 1
            return {"val": x}

        # First run populates cache
        robust_call(42)
        assert calls == 1

        # Corrupt all JSON files in the cache folder
        for file in cache_dir.glob("*.json"):
            file.write_text("INVALID_NON_JSON_CORRUPTED_DATA")

        # Second run should detect corruption, evict corrupted file, and recompute
        result = robust_call(42)
        assert result == {"val": 42}
        assert calls == 2

    def test_with_retry_success_after_failure(self):
        attempts = 0

        @with_retry(tries=3, base=0.01)
        def flaky_function():
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise ValueError("Transient network glitch")
            return "SUCCESS"

        result = flaky_function()
        assert result == "SUCCESS"
        assert attempts == 2

    def test_with_retry_exhaustion_raises(self):
        attempts = 0

        @with_retry(tries=3, base=0.01)
        def always_failing():
            nonlocal attempts
            attempts += 1
            raise RuntimeError("Permanent API outage")

        with pytest.raises(RuntimeError) as exc_info:
            always_failing()
        assert "Permanent API outage" in str(exc_info.value)
        assert attempts == 3


class TestTiming:
    """Tests for StageTimer context manager and measurement."""

    def test_stage_timer_records_positive_ms(self):
        timer = StageTimer()
        with timer.stage("retrieval"):
            time.sleep(0.015)  # 15 ms sleep
        with timer.stage("rerank"):
            time.sleep(0.010)  # 10 ms sleep

        assert timer.ms("retrieval") >= 10
        assert timer.ms("rerank") >= 5
        assert timer.total_ms() >= (timer.ms("retrieval") + timer.ms("rerank"))

        timing_model = timer.to_timing()
        assert timing_model.retrieval_ms == timer.ms("retrieval")
        assert timing_model.rerank_ms == timer.ms("rerank")
        assert timing_model.total_ms == timer.total_ms()

    def test_stage_timer_cumulative_accumulation(self):
        timer = StageTimer()
        with timer.stage("retrieval"):
            time.sleep(0.01)
        first_ms = timer.ms("retrieval")

        # Simulate retry loop executing retrieval a second time
        with timer.stage("retrieval"):
            time.sleep(0.01)
        total_retrieval_ms = timer.ms("retrieval")

        assert total_retrieval_ms > first_ms


class TestHermeticFakes:
    """Tests for deterministic FakeEmbedder, FakeReranker, and FakeLLM."""

    def test_fake_embedder_dimension_and_normalization(self, fake_embedder):
        vec = fake_embedder.embed_query("DocuRAG multimodal RAG")
        assert len(vec) == 384
        # Verify L2 norm is ~ 1.0
        norm = sum(x * x for x in vec) ** 0.5
        assert pytest.approx(norm, 0.001) == 1.0

        # Determinism check: same string yields identical vector
        vec2 = fake_embedder.embed_query("DocuRAG multimodal RAG")
        assert vec == vec2

    def test_fake_reranker_scoring(self, fake_reranker):
        query = "multimodal RAG system"
        doc_relevant = "A multimodal RAG system for documents."
        doc_irrelevant = "Cooking Italian pizza in a stone oven."

        scores = fake_reranker.predict([(query, doc_relevant), (query, doc_irrelevant)])
        assert scores[0] > scores[1]

    def test_fake_llm_responses(self, fake_llm):
        answer = fake_llm.generate_text("generate answer for query")
        assert "[E1]" in answer

        router_json = fake_llm.generate_text("classify the user's question: compare doc A vs doc B")
        data = json.loads(router_json)
        assert data["intent"] == "multi_doc_compare"


class TestPipelineFaçade:
    """Tests for pipeline entry points in hermetic fake mode."""

    def test_ask_linear_and_contract(self, monkeypatch):
        monkeypatch.setenv("DOCURAG_FAKES", "1")
        from docurag.config import get_settings
        get_settings(reload=True)
        result = _ask_linear("What is DocuRAG?")
        assert isinstance(result, AskResult)
        assert result.answerable is True
        assert len(result.sources) > 0
        assert "[E1]" in result.answer
        assert result.timing.total_ms >= 0

    def test_ask_agent_trace_population(self, monkeypatch):
        monkeypatch.setenv("DOCURAG_FAKES", "1")
        from docurag.config import get_settings
        get_settings(reload=True)
        result = ask("What is DocuRAG?", use_agents=True)
        assert isinstance(result, AskResult)
        assert len(result.agent_trace) > 0
        assert any(step.agent == "router" for step in result.agent_trace)
        assert any(step.agent == "verifier" for step in result.agent_trace)

    def test_ingest_stubs(self, monkeypatch):
        monkeypatch.setenv("DOCURAG_FAKES", "1")
        # Ensure the setting is picked up
        from docurag.config import get_settings
        settings = get_settings(reload=True)
        
        file_res = ingest_file("sample.pdf")
        assert file_res["status"] == "ready"
        assert file_res["filename"] == "sample.pdf"

        dir_res = ingest_directory("./data")
        assert dir_res["status"] == "ready"


class TestM0Iteration2Extensions:
    """Test suite verifying Iteration 2 fixes: synthetic fixtures, empty batch fakes, schemas, and stubs."""

    def test_fake_embedder_empty_batch_exact_2d_shape(self, fake_embedder):
        """Verify FakeEmbedder.encode([]) produces exactly 2D shape (0, 384)."""
        empty_matrix = fake_embedder.encode([])
        assert empty_matrix.shape == (0, 384)
        assert len(empty_matrix.shape) == 2
        assert empty_matrix.shape[1] == 384

    def test_synthetic_pdf_and_png_generators(self, minimal_pdf_bytes, minimal_png_bytes):
        """Verify pure Python generators produce valid PDF and PNG signatures."""
        assert minimal_pdf_bytes.startswith(b"%PDF-1.4")
        assert minimal_pdf_bytes.endswith(b"%%EOF\n")
        assert minimal_png_bytes.startswith(b"\x89PNG\r\n\x1a\n")

    def test_synthetic_file_fixtures(self, synthetic_corpus):
        """Verify all 5 synthetic file fixtures exist and have non-zero size."""
        for name, path in synthetic_corpus.items():
            assert path.exists(), f"Synthetic fixture '{name}' was not created!"
            assert path.stat().st_size > 0, f"Synthetic fixture '{name}' is empty!"

    def test_page_data_and_figure_data_schemas(self):
        """Verify PageData and FigureData validation and compatibility properties."""
        fig = FigureData(path="storage/figs/fig_1.png", bbox=[10.0, 20.0, 100.0, 150.0], page=1)
        assert fig.image_path == "storage/figs/fig_1.png"

        page = PageData(page=1, text="Page content", figures=[fig], tables_md=["| A | B |"])
        assert page.page == 1
        assert len(page.figures) == 1
        assert page.figures[0].path == "storage/figs/fig_1.png"

    def test_figure_description_flattening(self):
        """Verify FigureDescription flattens to searchable string format."""
        desc = FigureDescription(
            type="bar_chart",
            title="Hit Rate",
            caption="Shows hit rate.",
            key_values=[{"label": "DocuRAG", "value": "0.95", "unit": ""}],
        )
        flattened = desc.to_flattened_text()
        assert "[bar_chart] Hit Rate. Shows hit rate." in flattened
        assert "DocuRAG=0.95" in flattened

    def test_route_decision_invalid_fallback(self):
        """Verify RouteDecision defaults invalid intents to 'factual'."""
        decision = RouteDecision.model_validate({"intent": "invalid_hallucinated_intent"})
        assert decision.intent == "factual"
        assert decision.query_type == "general"

    def test_sub_question_alias_normalization(self):
        """Verify SubQuestion normalizes 'question' alias to 'sub_question'."""
        sq = SubQuestion.model_validate({"question": "Decomposed question?", "document_ids": ["doc.pdf"]})
        assert sq.sub_question == "Decomposed question?"
        assert sq.document_names == ["doc.pdf"]

    def test_public_interface_stubs_importable(self):
        """Verify all public interface stubs across all modules are importable and callable."""
        from docurag.ingest import (
            load_pdf,
            load_image,
            extract_pages,
            iter_pages,
            ocr_page,
            extract_figures,
            describe_figure,
            transcribe_page,
            classify_page,
        )
        from docurag.chunk import chunk_document, chunk_page
        from docurag.index import Embedder, ChunkStore, FAISSIndex, BM25Index
        from docurag.retrieve import hybrid_search, rerank, retrieve, retrieve_for_documents
        from docurag.generate import (
            generate_answer,
            synthesize_comparison,
            verify_answer,
            compute_confidence,
        )
        from docurag.agents.mcp_servers import (
            list_documents,
            get_page_text,
            get_page_image,
            hybrid_search as mcp_search,
            search_tables,
            search_images,
            calculate,
            extract_chart_values,
        )

        calc_res = calculate("2 + 3 * 4")
        assert calc_res["result"] == 14.0
        assert calc_res["error"] is None

        attack_res = calculate("__import__('os').system('ls')")
        assert attack_res["result"] is None
        assert attack_res["error"] is not None
