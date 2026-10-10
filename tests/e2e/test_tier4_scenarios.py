"""Tier 4 E2E Real-World Application Scenarios Test Suite.

Simulates end-to-end user workflows:
- Financial Table Lookup & Numerical Verification
- Academic / Technical Research Paper Q&A
- Multi-Document Comparative Synthesis
- Multimodal Chart & Figure Visual Q&A
- Refusal & Unanswerable Query Handling with Capped Confidence

Authoritative Output Derivation:
- CONTRACTS.md §1 Core Data Schemas & Unanswerable Confidence Invariant (score <= 0.2)
- DocuRAG_AI_Build_Spec.md §1 "What We Are Building" & §10 Evaluation Categories
- DocuRAG_Multi_Agent_Spec.md §B2 Query Workflows
"""

import os
import re
from pathlib import Path
import pytest

from docurag.pipeline import _ask_linear, ask, ingest_directory, ingest_file
from docurag.schemas import AskResult, Confidence, Source


@pytest.fixture(autouse=True)
def ensure_hermetic_fakes(monkeypatch):
    """Enforce hermetic test execution without external APIs or GPU."""
    monkeypatch.setenv("DOCURAG_FAKES", "1")


class TestTier4FinancialTableScenario:
    """Scenario: User asks questions over financial filings and tabular disclosures."""

    def test_financial_table_metric_lookup(self, tmp_path, synthetic_table_pdf):
        """Derivation: Base Spec §1 & §10 Table lookup with numerical precision."""
        # Step 1: Ingest financial table document
        ingest_res = ingest_file(str(synthetic_table_pdf), collection_id="financial_q3")
        assert ingest_res.get("status") == "ready"

        # Step 2: Query tabular metrics
        question = "What was the Hit@5 accuracy and latency reported in the benchmark table?"
        result = ask(question, use_agents=False)

        assert isinstance(result, AskResult)
        assert result.answerable is True
        assert len(result.sources) > 0

        # Verify that table content was incorporated
        table_sources = [s for s in result.sources if s.type in ["table", "text"]]
        assert len(table_sources) > 0, "Expected at least one table or text source for financial query"

        # Verify answer has citation tags
        assert re.search(r"\[E\d+\]", result.answer), (
            f"Table lookup answer must include [E#] citation tags: {result.answer}"
        )


class TestTier4ResearchPaperScenario:
    """Scenario: Researcher asks technical questions against an academic paper."""

    def test_academic_paper_architecture_qa(self, tmp_path, synthetic_text_pdf):
        """Derivation: Base Spec §1 Paraphrased query over technical document."""
        # Step 1: Ingest research paper
        ingest_res = ingest_file(str(synthetic_text_pdf), collection_id="arxiv_papers")
        assert ingest_res.get("status") == "ready"

        # Step 2: Paraphrased conceptual question
        question = "Can you outline how the hybrid indexing mechanism operates in this architecture?"
        result = ask(question, use_agents=False)

        assert isinstance(result, AskResult)
        assert result.answerable is True
        assert result.confidence.score >= 0.75 or result.confidence.label in ["High", "Medium"]
        assert result.verified is True
        assert result.support_ratio >= 0.7

        # Verify citation attribution
        for src in result.sources:
            assert src.page >= 1
            assert len(src.snippet) > 0


class TestTier4MultiDocComparisonScenario:
    """Scenario: Analyst compares metrics and architectures across multiple documents."""

    def test_multi_document_comparison_workflow(self, tmp_path, synthetic_text_pdf, synthetic_table_pdf):
        """Derivation: Multi-Agent Spec §B2 multi_doc_compare workflow."""
        # Step 1: Ingest two distinct documents into the same collection
        ingest_file(str(synthetic_text_pdf), collection_id="comparison_corpus")
        ingest_file(str(synthetic_table_pdf), collection_id="comparison_corpus")

        # Step 2: Comparative query in agentic mode
        comparison_question = (
            "Compare the linear pipeline vs multi-agent system performance across latency and accuracy."
        )
        result = ask(comparison_question, use_agents=True)

        assert isinstance(result, AskResult)
        assert len(result.agent_trace) > 0, "Multi-document comparison must generate agent trace"

        # Trace should include router classification
        step_actions = [s.action for s in result.agent_trace]
        assert any("classif" in a.lower() for a in step_actions), (
            f"Expected classification step in agent trace: {step_actions}"
        )


class TestTier4ChartAndFigureScenario:
    """Scenario: Visual element reasoning and chart context retrieval."""

    def test_figure_extraction_and_query(self, tmp_path, synthetic_chart_pdf):
        """Derivation: Base Spec M1 figure extraction and M4 visual injection."""
        ingest_res = ingest_file(str(synthetic_chart_pdf), collection_id="charts_corpus")
        assert ingest_res.get("status") == "ready"

        question = "What trends are illustrated in Figure 1 regarding retrieval latency?"
        result = ask(question, use_agents=False)

        assert isinstance(result, AskResult)
        assert len(result.sources) >= 1
        assert result.timing.total_ms >= 0


class TestTier4RefusalAndUnanswerableScenario:
    """Scenario: Query cannot be answered by evidence pool (hallucination prevention)."""

    def test_unanswerable_query_confidence_cap_invariant(self):
        """Derivation: CONTRACTS.md §1 When unanswerable, confidence score capped at 0.20."""
        # Querying completely out-of-scope domain with zero grounding
        out_of_scope_q = "What is the secret recipe for Martian space gelato on Jupiter in 2150?"
        result = ask(out_of_scope_q, use_agents=False)

        assert isinstance(result, AskResult)
        # If the engine identifies unanswerable=False, confidence must be capped at <= 0.20
        if not result.answerable:
            assert result.confidence.score <= 0.20, (
                f"CONTRACT VIOLATION: Unanswerable query confidence score {result.confidence.score} > 0.20"
            )
            assert result.confidence.label == "Low", (
                f"CONTRACT VIOLATION: Unanswerable query must have 'Low' confidence, got {result.confidence.label}"
            )
        else:
            # If answerable, confidence must remain within valid range [0, 1]
            assert 0.0 <= result.confidence.score <= 1.0
