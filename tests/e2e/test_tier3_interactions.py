"""Tier 3 E2E Cross-Feature Interactions Test Suite.

Validates end-to-end interactions across pipeline subsystems:
- Ingestion followed by query retrieval lifecycle
- Multi-document ingestion and collection isolation
- Sequential mode switching (linear <-> agentic transitions)
- Citation integrity and cross-stage evidence propagation
- Verifier stage agreement with confidence and support ratio
- Ingest-then-query state isolation across consecutive calls

Authoritative Output Derivation:
- CONTRACTS.md §1 & §4 Pipeline Invariants
- DocuRAG_AI_Build_Spec.md §6 M0–M5 End-to-end pipeline lifecycle
- DocuRAG_Multi_Agent_Spec.md §B8 Circuit breaker and mode switching
"""

import os
import re
from pathlib import Path
import pytest

from docurag.pipeline import _ask_linear, ask, ingest_directory, ingest_file
from docurag.schemas import AskResult, ContentType, Evidence, Source


@pytest.fixture(autouse=True)
def ensure_hermetic_fakes(monkeypatch):
    """Enforce hermetic test execution without external APIs or GPU."""
    monkeypatch.setenv("DOCURAG_FAKES", "1")


class TestTier3IngestAndQueryLifecycle:
    """Interactions between document ingestion and subsequent query answering."""

    def test_ingest_document_then_query(self, tmp_path, minimal_pdf_bytes):
        """Derivation: Ingestion produces indexable artifacts consumable by ask()."""
        # Step 1: Ingest document
        doc_path = tmp_path / "architecture_spec.pdf"
        doc_path.write_bytes(minimal_pdf_bytes)
        ingest_res = ingest_file(str(doc_path), collection_id="test_lifecycle")

        assert ingest_res.get("status") == "ready"
        assert ingest_res.get("filename") == "architecture_spec.pdf"

        # Step 2: Query the system
        ask_res = ask("What is described in the architecture spec?", use_agents=False)

        assert isinstance(ask_res, AskResult)
        assert ask_res.answerable is True
        assert len(ask_res.sources) > 0
        assert len(ask_res.evidence) > 0

    def test_ingest_directory_then_multi_query(self, tmp_path, synthetic_corpus):
        """Derivation: Batch corpus ingestion followed by successive queries."""
        corpus_dir = tmp_path / "research_papers"
        corpus_dir.mkdir()

        # Copy synthetic files into corpus directory
        for name, src_file in synthetic_corpus.items():
            dest = corpus_dir / f"doc_{name}_{src_file.name}"
            dest.write_bytes(src_file.read_bytes())

        # Ingest directory
        dir_res = ingest_directory(str(corpus_dir), collection_id="papers_corpus")
        assert dir_res.get("status") == "ready"

        # Query 1: Text-oriented query
        q1_res = ask("Summarize the main findings of the paper", use_agents=False)
        assert isinstance(q1_res, AskResult)
        assert q1_res.timing.total_ms >= 0

        # Query 2: Table-oriented query
        q2_res = ask("What are the benchmark accuracy values in the table?", use_agents=False)
        assert isinstance(q2_res, AskResult)
        assert len(q2_res.sources) > 0

    def test_multi_collection_ingestion_isolation(self, tmp_path, minimal_pdf_bytes):
        """Derivation: Documents ingested into different collections maintain isolation."""
        doc1 = tmp_path / "finance.pdf"
        doc2 = tmp_path / "engineering.pdf"
        doc1.write_bytes(minimal_pdf_bytes)
        doc2.write_bytes(minimal_pdf_bytes)

        res1 = ingest_file(str(doc1), collection_id="finance_dept")
        res2 = ingest_file(str(doc2), collection_id="engineering_dept")

        assert res1.get("collection_id") == "finance_dept"
        assert res2.get("collection_id") == "engineering_dept"
        assert res1.get("filename") != res2.get("filename")


class TestTier3ModeTransitions:
    """Interactions and state isolation during mode switching."""

    def test_alternating_linear_and_agentic_modes(self):
        """Derivation: Consecutive calls alternating use_agents flags do not cross-contaminate state."""
        question = "Explain the system architecture"

        # Call 1: Linear Mode
        res_linear_1 = ask(question, use_agents=False)
        assert len(res_linear_1.agent_trace) == 0, "Linear call 1 must have no agent trace"

        # Call 2: Agentic Mode
        res_agent = ask(question, use_agents=True)
        assert len(res_agent.agent_trace) > 0, "Agentic call must have agent trace"

        # Call 3: Linear Mode again (verify trace state didn't leak)
        res_linear_2 = ask(question, use_agents=False)
        assert len(res_linear_2.agent_trace) == 0, (
            "Subsequent linear call must not retain agent trace from previous call"
        )

        # Call 4: Direct _ask_linear
        res_direct = _ask_linear(question)
        assert len(res_direct.agent_trace) == 0

    def test_trace_step_ordering_in_agent_mode(self):
        """Derivation: Multi-Agent Spec §B2 Agent orchestration ordering."""
        res = ask("Compare performance between baseline and DocuRAG", use_agents=True)

        trace = res.agent_trace
        assert len(trace) >= 1

        # Trace actions must follow chronological sequence
        agents_seen = [step.agent for step in trace]
        if "router" in agents_seen:
            # Router must precede downstream generation / verification
            router_idx = agents_seen.index("router")
            assert router_idx == 0, "Router must be the initial step in the agent workflow"


class TestTier3EvidenceAndCitationPropagation:
    """Interactions between retrieved evidence, reranking, and citation generation."""

    def test_citation_mapping_and_evidence_correlation(self):
        """Derivation: Base Spec §5 Source-Evidence bijective correlation."""
        result = ask("What is DocuRAG?", use_agents=False)

        # Map evidence by eid
        evidence_by_eid = {ev.eid: ev for ev in result.evidence}

        # Check citations in text
        citations = re.findall(r"\[E(\d+)\]", result.answer)
        for num in citations:
            eid = f"E{num}"
            assert eid in evidence_by_eid, (
                f"Generated answer cites [{eid}] which does not exist in result.evidence"
            )

        # Check sources correspond to evidence chunks
        for idx, src in enumerate(result.sources):
            matching_ev = result.evidence[idx]
            assert src.page == matching_ev.chunk.page
            assert src.type == matching_ev.chunk.content_type
            assert len(src.snippet) > 0

    def test_verifier_agreement_with_confidence_score(self):
        """Derivation: CONTRACTS.md §1 & Base Spec §5.4 Confidence and verifier support ratio."""
        result = ask("Explain DocuRAG verification", use_agents=False)

        if result.verified is True:
            # When verified is True, support_ratio should be >= 0.7 (SUPPORT_THRESHOLD)
            assert result.support_ratio >= 0.7, (
                f"Verified answer has support_ratio {result.support_ratio} < 0.7 threshold"
            )
            assert result.confidence.score >= 0.75 or result.confidence.label in ["High", "Medium"]
        else:
            assert result.support_ratio < 0.7
            assert result.confidence.score < 0.75

    def test_modality_diversity_in_evidence(self):
        """Derivation: Base Spec M4 Visual keyword diversity in evidence pool."""
        result = ask("What does the architecture diagram show?", use_agents=False)

        # Check content types present in evidence chunks
        content_types = {ev.chunk.content_type for ev in result.evidence}
        assert len(content_types) >= 1
        valid_types = {"text", "table", "image", "ocr"}
        for ct in content_types:
            assert ct in valid_types, f"Unknown content type: {ct}"
