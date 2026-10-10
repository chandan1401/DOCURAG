"""Public pipeline facade for DocuRAG ingestion and query workflows."""

import logging
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from docurag.config import get_settings
from docurag.schemas import (
    AskResult,
    Chunk,
    Confidence,
    Evidence,
    Source,
    Timing,
    TraceStep,
    Filters,
)
from docurag.timing import StageTimer
from docurag.index.store import ChunkStore
from docurag.index.embedder import Embedder
from docurag.retrieve.hybrid import retrieve
from docurag.generate.answer import generate_answer

logger = logging.getLogger(__name__)

# Only one ingestion at a time, so concurrent uploads can't overwrite each other's index
_INGEST_LOCK = threading.Lock()

# Singleton engine components
_ENGINE_CACHE = {}


def get_engine():
    """Lazy singleton for the DocuRAG engine components."""
    if "engine" not in _ENGINE_CACHE:
        settings = get_settings()
        embedder = Embedder(model_name=settings.EMBED_MODEL)
        store = ChunkStore(embedder=embedder, persistence_path=settings.DOCURAG_STORAGE + "/index")
        store.load()

        reranker = None
        try:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder(os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
        except Exception as e:
            logger.warning("Reranker unavailable (%s); using fusion order.", e)

        _ENGINE_CACHE["engine"] = {
            "embedder": embedder,
            "store": store,
            "reranker": reranker,
        }
    return _ENGINE_CACHE["engine"]


def _get_hermetic_fake_result(
    question: str,
    use_agents: bool,
    document_ids: Optional[List[str]] = None,
    content_types: Optional[List[str]] = None
) -> AskResult:
    """Generate deterministic AskResult using hermetic fake components."""
    with StageTimer() as timer:
        with timer.stage("retrieval"):
            sample_chunk_1 = Chunk(
                chunk_id="c_fake_001",
                document_id="doc_intro",
                source="intro.pdf",
                page=1,
                content_type="text",
                section="Overview",
                content=(
                    "DocuRAG is a multimodal RAG system designed for document collections, "
                    "supporting table extraction, image reasoning, and verified answers."
                ),
                token_count=24,
            )
            sample_chunk_2 = Chunk(
                chunk_id="c_fake_002",
                document_id="doc_intro",
                source="intro.pdf",
                page=2,
                content_type="table",
                section="Architecture",
                content="| Component | Description |\n| Engine | Hybrid RRF + Rerank |",
                token_count=18,
            )
            ev1 = Evidence(
                eid="E1",
                chunk=sample_chunk_1,
                score=0.96,
                norm_score=0.95,
                dense_score=0.92,
                sparse_score=14.5,
                rerank_score=0.96,
                combined_score=0.94,
            )
            ev2 = Evidence(
                eid="E2",
                chunk=sample_chunk_2,
                score=0.88,
                norm_score=0.85,
                dense_score=0.81,
                sparse_score=10.2,
                rerank_score=0.88,
                combined_score=0.86,
            )
            evidence = [ev1, ev2]

        with timer.stage("rerank"):
            pass

        with timer.stage("generation"):
            answer = (
                f"DocuRAG is a multimodal RAG system that ingests PDFs and images [E1], "
                f"retrieving relevant text and tables via hybrid search [E2]."
            )

        with timer.stage("verification"):
            verified = True
            support_ratio = 1.0

    sources = [
        Source(
            id=str(idx + 1),
            document=ev.chunk.source,
            page=ev.chunk.page,
            type=ev.chunk.content_type,
            snippet=ev.chunk.content[:120],
            score=ev.score,
            confidence=0.95,
        )
        for idx, ev in enumerate(evidence)
    ]

    confidence = Confidence(
        score=0.94,
        label="High",
        support_ratio=support_ratio,
        citation_coverage=1.0,
        reason="Fully supported by retrieved evidence",
    )

    agent_trace: List[TraceStep] = []
    if use_agents:
        agent_trace = [
            TraceStep(
                agent="router",
                action="classify",
                tool=None,
                detail="Classified question as factual query",
                ms=15,
            ),
            TraceStep(
                agent="retrieval",
                action="hybrid_search",
                tool="retrieval_server.hybrid_search",
                detail="Retrieved 2 top chunks using BM25 and BGE dense embeddings",
                ms=timer.ms("retrieval"),
            ),
            TraceStep(
                agent="verifier",
                action="verify_claims",
                tool=None,
                detail="All claims verified against cited sources",
                ms=timer.ms("verification"),
            ),
        ]

    return AskResult(
        question=question,
        answer=answer,
        answerable=True,
        sources=sources,
        confidence=confidence,
        evidence=evidence,
        timing=timer.to_timing(),
        verified=verified,
        support_ratio=support_ratio,
        agent_trace=agent_trace,
    )


def _compute_confidence(answerable, cited_ids, valid_ids):
    """Heuristic confidence based on whether the answer is grounded in cited evidence."""
    if not valid_ids or not answerable:
        return Confidence(
            score=0.1, label="Low", support_ratio=0.0, citation_coverage=0.0,
            reason="Retrieved evidence did not answer the question",
        )
    if not cited_ids:
        return Confidence(
            score=0.3, label="Low", support_ratio=0.0, citation_coverage=0.0,
            reason="Answer cites no retrieved evidence",
        )
    coverage = len(cited_ids) / len(valid_ids)
    score = round(min(0.95, 0.6 + 0.35 * min(1.0, len(cited_ids) / 3)), 2)
    label = "High" if score >= 0.8 else "Medium"
    return Confidence(
        score=score, label=label, support_ratio=1.0,
        citation_coverage=round(coverage, 2),
        reason="Answer is grounded in cited evidence",
    )


def _ask_linear(
    question: str,
    top_k: int = 5,
    document_ids: Optional[List[str]] = None,
    content_types: Optional[List[str]] = None,
    instructions: Optional[str] = None,
) -> AskResult:
    """Execute standard linear RAG pipeline without multi-agent routing."""
    settings = get_settings()

    # MODE CHECK: Only return fakes if explicitly requested and enabled
    if settings.DOCURAG_FAKES or os.getenv("DOCURAG_FAKES") == "1":
        return _get_hermetic_fake_result(
            question,
            use_agents=False,
            document_ids=document_ids,
            content_types=content_types
        )

    # REAL PIPELINE IMPLEMENTATION
    with StageTimer() as timer:
        engine = get_engine()
        store = engine["store"]
        embedder = engine["embedder"]

        # 1. Retrieval & Reranking
        with timer.stage("retrieval"):
            filters = Filters(document_ids=document_ids, content_types=content_types)
            evidence = retrieve(
                query=question,
                store=store,
                embedder=embedder,
                reranker=engine.get("reranker"),
                filters=filters,
                final_k=top_k
            )

        # 2. Grounded Generation
        with timer.stage("generation"):
            gen_result = generate_answer(question, evidence, instructions=instructions)

        # 3. Verification: the answer must cite evidence that actually exists
        with timer.stage("verification"):
            valid_ids = {ev.eid for ev in evidence}
            cited_ids = set(re.findall(r"\[(E\d+)\]", gen_result.answer or "")) & valid_ids
            supported = bool(gen_result.answerable and cited_ids)
            support_ratio = 1.0 if supported else 0.0

        # 4. Confidence scoring
        confidence = _compute_confidence(gen_result.answerable, cited_ids, valid_ids)

    sources = [
        Source(
            id=str(idx + 1),
            document=ev.chunk.source,
            page=ev.chunk.page,
            type=ev.chunk.content_type,
            snippet=ev.chunk.content[:300],
            score=ev.score,
            confidence=round(float(ev.score), 2),
        )
        for idx, ev in enumerate(evidence)
    ]

    return AskResult(
        question=question,
        answer=gen_result.answer,
        answerable=gen_result.answerable,
        sources=sources,
        confidence=confidence,
        evidence=evidence,
        timing=timer.to_timing(),
        verified=supported,
        support_ratio=support_ratio,
        agent_trace=[],
    )


def ask(
    question: str,
    top_k: int = 5,
    use_agents: Optional[bool] = None,
    document_ids: Optional[List[str]] = None,
    content_types: Optional[List[str]] = None,
    instructions: Optional[str] = None,
) -> AskResult:
    """Primary query entry point. Dispatches to linear or agentic pipeline."""
    settings = get_settings(reload=True)
    if use_agents is None:
        use_agents = settings.USE_AGENTS

    if settings.DOCURAG_FAKES or os.getenv("DOCURAG_FAKES") == "1":
        return _get_hermetic_fake_result(
            question,
            use_agents=use_agents,
            document_ids=document_ids,
            content_types=content_types
        )

    # Agent runtime (M8) is not implemented yet; both paths use the linear pipeline.
    return _ask_linear(
        question,
        top_k=top_k,
        document_ids=document_ids,
        content_types=content_types,
        instructions=instructions,
    )


def ingest_file(file_path: str, collection_id: str = "default", document_id: str = None) -> Dict[str, Any]:
    """Ingest a single document into the system."""
    path = Path(file_path)
    settings = get_settings(reload=True)
    if settings.DOCURAG_FAKES and not path.exists():
        return {
            "document_id": document_id or f"doc_{path.stem}",
            "filename": path.name,
            "collection_id": collection_id,
            "status": "ready",
            "pages": 1,
            "chunks_indexed": 2,
        }
    from docurag.ingest.coordinator import ingest_document

    # Only one ingestion at a time, so uploads can't overwrite each other's index
    with _INGEST_LOCK:
        report = ingest_document(path, collection_id=collection_id, document_id=document_id)
        engine = _ENGINE_CACHE.get("engine")
        if engine is not None:
            refreshed_store = ChunkStore(
                embedder=engine["embedder"],
                persistence_path=settings.DOCURAG_STORAGE + "/index",
            )
            refreshed_store.load()
            engine["store"] = refreshed_store

    report.pop("pages_data", None)
    return report


def remove_document(document_id: str) -> None:
    """Remove a document's chunks from the index and persist the change."""
    settings = get_settings()
    with _INGEST_LOCK:
        engine = get_engine()
        store = ChunkStore(
            embedder=engine["embedder"],
            persistence_path=settings.DOCURAG_STORAGE + "/index",
        )
        store.load()
        store.remove_document(str(document_id))
        store.save()
        engine["store"] = store


def ingest_directory(dir_path: str, collection_id: str = "default") -> Dict[str, Any]:
    """Ingest all supported documents from a directory."""
    path = Path(dir_path)
    settings = get_settings(reload=True)
    if settings.DOCURAG_FAKES and not path.exists():
        return {
            "collection_id": collection_id,
            "directory": str(path),
            "files_ingested": 1,
            "total_chunks": 2,
            "status": "ready",
        }
    from docurag.ingest.coordinator import ingest_directory as ingest_dir

    report = ingest_dir(path, collection_id=collection_id)
    if settings.DOCURAG_FAKES and report.get("status") == "empty":
        report["status"] = "ready"
    return report


def main():
    """CLI entrypoint for executing ad-hoc queries."""
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "What is DocuRAG?"

    result = ask(question)

    print("=" * 60)
    print(f"Question: {result.question}")
    print(f"Answer:   {result.answer}")
    print(f"Confidence: {result.confidence.score:.2f} ({result.confidence.label})")
    print(f"Timing:     {result.timing.total_ms}ms (Retrieval: {result.timing.retrieval_ms}ms, Generation: {result.timing.generation_ms}ms)")
    print(f"Sources:    {len(result.sources)}")
    for src in result.sources:
        print(f"  [{src.id}] {src.document} (p. {src.page}): {src.snippet[:80]}...")
    if result.agent_trace:
        print(f"Agent Trace: {len(result.agent_trace)} steps")
        for step in result.agent_trace:
            print(f"  [{step.agent}] {step.action} ({step.ms}ms): {step.detail}")
    print("=" * 60)


if __name__ == "__main__":
    main()