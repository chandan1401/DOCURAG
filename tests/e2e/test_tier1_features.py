"""Tier 1 E2E Feature Coverage Test Suite.

Validates core public entrypoints:
- `docurag.pipeline.ask()`
- `docurag.pipeline._ask_linear()`
- `docurag.pipeline.ingest_file()`
- `docurag.pipeline.ingest_directory()`

Authoritative Output Derivation:
- CONTRACTS.md §1: Core Data Schemas (AskResult, Source, Confidence, Evidence, Timing, TraceStep)
- CONTRACTS.md §2: REST API Query Response Shape
- CONTRACTS.md §4: Pipeline Engine Public API
- DocuRAG_AI_Build_Spec.md §5: Data Contracts and Verification Invariants
"""

import json
import os
import re
from pathlib import Path
import pytest

from docurag.config import Settings, get_settings
from docurag.pipeline import _ask_linear, ask, ingest_directory, ingest_file
from docurag.schemas import (
    AskResult,
    Confidence,
    Evidence,
    Source,
    Timing,
    TraceStep,
)


@pytest.fixture(autouse=True)
def ensure_hermetic_fakes(monkeypatch):
    """Enforce hermetic test execution without external APIs or GPU."""
    monkeypatch.setenv("DOCURAG_FAKES", "1")


class TestTier1AskLinear:
    """Tier 1 tests for the linear RAG pipeline public entrypoint."""

    def test_ask_linear_returns_conforming_ask_result(self):
        """Derivation: CONTRACTS.md §1 & §4 -> _ask_linear returns valid AskResult."""
        question = "What is DocuRAG and what are its core capabilities?"
        result = _ask_linear(question, top_k=5)

        assert isinstance(result, AskResult), f"Expected AskResult instance, got {type(result)}"
        assert result.question == question
        assert len(result.answer) > 0, "Answer must not be empty"
        assert result.answerable is True
        assert isinstance(result.sources, list)
        assert len(result.sources) > 0, "Linear ask must return at least one source"
        assert isinstance(result.confidence, Confidence)
        assert isinstance(result.timing, Timing)
        assert isinstance(result.evidence, list)
        assert len(result.evidence) > 0, "Linear ask must return evidence"

    def test_ask_linear_evidence_and_sources_alignment(self):
        """Derivation: CONTRACTS.md §1 Source and Evidence models must correspond."""
        result = _ask_linear("Explain hybrid retrieval architecture.")

        assert len(result.sources) == len(result.evidence)
        for idx, (source, evidence) in enumerate(zip(result.sources, result.evidence)):
            assert isinstance(source, Source)
            assert isinstance(evidence, Evidence)
            # Source id corresponds to 1-based index or citation ID
            assert source.id == str(idx + 1) or source.citation_id == idx + 1
            # Source document matches evidence chunk source
            assert source.document == evidence.chunk.source or source.doc_name == evidence.chunk.doc_name
            # Source page matches evidence chunk page
            assert source.page == evidence.chunk.page
            # Score must be non-negative
            assert source.score >= 0.0
            assert evidence.score >= 0.0

    def test_ask_linear_citation_invariants(self):
        """Derivation: DocuRAG_AI_Build_Spec.md §7.2 Rule 2 -> [E#] citations map to evidence."""
        result = _ask_linear("How are tables extracted in DocuRAG?")

        # Extract citation tokens [E1], [E2], etc. from generated answer
        citations_found = re.findall(r"\[E(\d+)\]", result.answer)
        assert len(citations_found) > 0, (
            f"Grounded answer must contain [E#] citation tags. Answer: {result.answer}"
        )

        valid_eids = {ev.eid for ev in result.evidence}
        for cite_num in citations_found:
            expected_eid = f"E{cite_num}"
            assert expected_eid in valid_eids, (
                f"Answer cited unknown evidence tag [{expected_eid}]. Valid evidence: {valid_eids}"
            )

    def test_ask_linear_confidence_contract(self):
        """Derivation: CONTRACTS.md §1 Confidence label thresholds: >=0.75 High, 0.4..0.75 Med, <0.4 Low."""
        result = _ask_linear("Summarize multimodal indexing.")

        score = result.confidence.score
        label = result.confidence.label

        assert 0.0 <= score <= 1.0, f"Confidence score {score} outside [0.0, 1.0]"
        if score >= 0.75:
            assert label == "High", f"Score {score} expected High label, got {label}"
        elif score >= 0.4:
            assert label == "Medium", f"Score {score} expected Medium label, got {label}"
        else:
            assert label == "Low", f"Score {score} expected Low label, got {label}"

        assert 0.0 <= result.confidence.support_ratio <= 1.0
        assert 0.0 <= result.confidence.citation_coverage <= 1.0

    def test_ask_linear_timing_breakdown(self):
        """Derivation: CONTRACTS.md §1 Timing model non-negative stage latencies."""
        result = _ask_linear("Verify timing breakdown fields.")

        timing = result.timing
        assert timing.total_ms >= 0
        assert timing.retrieval_ms >= 0
        assert timing.rerank_ms >= 0
        assert timing.generation_ms >= 0
        assert timing.verification_ms >= 0
        # Total latency must accommodate stages
        assert timing.total_ms >= max(timing.retrieval_ms, timing.generation_ms)

    def test_ask_linear_top_k_parameter(self):
        """Derivation: CONTRACTS.md §4 Pipeline Engine API top_k parameter."""
        res_default = _ask_linear("Top-k test query", top_k=5)
        res_custom = _ask_linear("Top-k test query", top_k=2)

        assert isinstance(res_default, AskResult)
        assert isinstance(res_custom, AskResult)


class TestTier1AskAgentic:
    """Tier 1 tests for the multi-agent and routed RAG query entrypoint."""

    def test_ask_with_agents_flag_enabled(self):
        """Derivation: CONTRACTS.md §1 TraceStep v2 and AskResult.agent_trace when USE_AGENTS=True."""
        question = "Compare revenue growth across quarters."
        result = ask(question, use_agents=True)

        assert isinstance(result, AskResult)
        assert result.question == question
        assert len(result.agent_trace) > 0, (
            "Agent mode (use_agents=True) must populate agent_trace with TraceStep records"
        )

    def test_ask_agent_trace_schema_conformance(self):
        """Derivation: CONTRACTS.md §1 TraceStep schema validation."""
        result = ask("Find data points in the table", use_agents=True)

        for step in result.agent_trace:
            assert isinstance(step, TraceStep)
            assert step.agent in ["router", "retrieval", "analyst", "planner", "synthesizer", "verifier"], (
                f"Unexpected agent name: {step.agent}"
            )
            assert len(step.action) > 0, "TraceStep action must not be empty"
            assert step.ms >= 0, "TraceStep execution duration must be non-negative"

        agent_names = [s.agent for s in result.agent_trace]
        assert "router" in agent_names, "Agent trace must include router classification step"

    def test_ask_with_agents_flag_disabled(self):
        """Derivation: CONTRACTS.md §4 ask(..., use_agents=False) routes to linear pipeline."""
        result = ask("Standard linear query", use_agents=False)

        assert isinstance(result, AskResult)
        assert len(result.agent_trace) == 0, (
            "Linear mode (use_agents=False) must not populate multi-agent trace steps"
        )
        assert len(result.sources) > 0
        assert len(result.evidence) > 0

    def test_ask_default_agent_behavior(self, monkeypatch):
        """Derivation: Settings.USE_AGENTS controls default dispatch when use_agents is None."""
        monkeypatch.setenv("USE_AGENTS", "0")
        res_linear = ask("Default setting query")
        assert len(res_linear.agent_trace) == 0

        monkeypatch.setenv("USE_AGENTS", "1")
        res_agent = ask("Default setting query")
        assert len(res_agent.agent_trace) > 0


class TestTier1Ingestion:
    """Tier 1 tests for single-file and directory ingestion entrypoints."""

    def test_ingest_single_pdf_file(self, tmp_path, minimal_pdf_bytes):
        """Derivation: CONTRACTS.md §4 ingest_file contract with synthetic PDF."""
        pdf_path = tmp_path / "sample_report.pdf"
        pdf_path.write_bytes(minimal_pdf_bytes)

        res = ingest_file(str(pdf_path))

        assert isinstance(res, dict)
        assert res.get("status") == "ready"
        assert res.get("filename") == "sample_report.pdf"
        assert res.get("document_id", "").startswith("doc_sample_report_")
        assert res.get("collection_id") == "default"
        assert res.get("pages", 0) >= 1
        assert res.get("chunks_indexed", 0) >= 1

    def test_ingest_single_image_file(self, tmp_path, minimal_png_bytes):
        """Derivation: CONTRACTS.md §4 ingest_file contract with synthetic PNG image."""
        png_path = tmp_path / "architecture_diagram.png"
        png_path.write_bytes(minimal_png_bytes)

        res = ingest_file(str(png_path))

        assert isinstance(res, dict)
        assert res.get("status") == "ready"
        assert res.get("filename") == "architecture_diagram.png"
        assert res.get("document_id", "").startswith("doc_architecture_diagram_")

    def test_ingest_file_custom_collection_id(self, tmp_path, minimal_pdf_bytes):
        """Derivation: CONTRACTS.md §4 collection_id parameter isolation."""
        pdf_path = tmp_path / "tax_form.pdf"
        pdf_path.write_bytes(minimal_pdf_bytes)

        res = ingest_file(str(pdf_path), collection_id="financial_records_2026")

        assert res.get("collection_id") == "financial_records_2026"
        assert res.get("filename") == "tax_form.pdf"

    def test_ingest_directory(self, tmp_path, minimal_pdf_bytes, minimal_png_bytes):
        """Derivation: CONTRACTS.md §4 ingest_directory batch document ingestion."""
        docs_dir = tmp_path / "documents"
        docs_dir.mkdir()
        (docs_dir / "doc1.pdf").write_bytes(minimal_pdf_bytes)
        (docs_dir / "doc2.png").write_bytes(minimal_png_bytes)

        res = ingest_directory(str(docs_dir))

        assert isinstance(res, dict)
        assert res.get("status") == "ready"
        assert res.get("directory") == str(docs_dir)
        assert res.get("files_ingested", 0) >= 1
        assert res.get("total_chunks", 0) >= 1

    def test_ingest_directory_custom_collection(self, tmp_path, minimal_pdf_bytes):
        """Derivation: CONTRACTS.md §4 ingest_directory with explicit collection_id."""
        docs_dir = tmp_path / "legal_docs"
        docs_dir.mkdir()
        (docs_dir / "contract.pdf").write_bytes(minimal_pdf_bytes)

        res = ingest_directory(str(docs_dir), collection_id="legal_corpus")

        assert res.get("collection_id") == "legal_corpus"
        assert res.get("status") == "ready"


class TestTier1SerializationAndContracts:
    """Tier 1 serialization, JSON compliance, and REST compatibility tests."""

    def test_ask_result_json_serialization_roundtrip(self):
        """Derivation: CONTRACTS.md §1 Pydantic model serialization and deserialization."""
        original = ask("Can AskResult serialize cleanly to JSON?", use_agents=True)

        json_str = original.model_dump_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0

        restored_data = json.loads(json_str)
        restored = AskResult.model_validate(restored_data)

        assert restored.question == original.question
        assert restored.answer == original.answer
        assert restored.answerable == original.answerable
        assert len(restored.sources) == len(original.sources)
        assert len(restored.evidence) == len(original.evidence)
        assert len(restored.agent_trace) == len(original.agent_trace)
        assert restored.confidence.score == original.confidence.score
        assert restored.confidence.label == original.confidence.label
        assert restored.timing.total_ms == original.timing.total_ms

    def test_ask_result_matches_rest_contract_schema(self):
        """Derivation: CONTRACTS.md §2 POST /api/query/ response contract specification."""
        result = ask("REST API schema validation query", use_agents=True)
        dump = result.model_dump()

        expected_root_keys = {
            "question",
            "answer",
            "answerable",
            "sources",
            "confidence",
            "evidence",
            "timing",
            "verified",
            "support_ratio",
            "agent_trace",
        }
        assert expected_root_keys.issubset(set(dump.keys())), (
            f"Missing required keys in REST AskResult response: {expected_root_keys - set(dump.keys())}"
        )

        # Validate nested sources format
        for src in dump["sources"]:
            assert {"id", "document", "page", "type", "snippet", "score"}.issubset(set(src.keys()))

        # Validate nested confidence format
        assert {"score", "label", "support_ratio", "citation_coverage", "reason"}.issubset(
            set(dump["confidence"].keys())
        )

        # Validate nested timing format
        assert {"retrieval_ms", "rerank_ms", "generation_ms", "verification_ms", "total_ms"}.issubset(
            set(dump["timing"].keys())
        )
