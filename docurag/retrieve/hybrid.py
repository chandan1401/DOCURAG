"""Hybrid dense+sparse retrieval (RRF) with cross-encoder reranking and multi-document coverage."""

import os
import re
import numpy as np
from typing import Any, List, Optional, Tuple, Dict
from collections import defaultdict

from docurag.config import get_settings
from docurag.schemas import Chunk, Evidence, Filters
from docurag.retrieve.rerank import rerank

# "what does all the files contain", "what files do I have", "summarize all documents"
_BROAD_RE = re.compile(
    r"\b(all|every|each)\b.{0,20}\b(files?|documents?|docs?|pdfs?|uploads?)\b"
    r"|\bwhat\b.{0,25}\b(files?|documents?|docs?)\b.{0,25}\b(have|uploaded|contain|in)\b"
    r"|\b(list|summari[sz]e|overview of)\b.{0,20}\b(files?|documents?|docs?)\b",
    re.IGNORECASE,
)


def _is_fakes() -> bool:
    if os.getenv("DOCURAG_FAKES") in ("1", "true", "True"):
        return True
    try:
        return get_settings().DOCURAG_FAKES
    except Exception:
        return False


def _is_broad_query(query: str) -> bool:
    return bool(_BROAD_RE.search(query or ""))


def hybrid_search(
    query: str,
    store: Any = None,
    embedder: Any = None,
    *,
    filters: Optional[Filters] = None,
    dense_k: int = 30,
    bm25_k: int = 30,
    rrf_k: int = 60,
) -> List[Tuple[Chunk, float]]:
    """Dense + BM25 candidates fused with Reciprocal Rank Fusion.

    Returns (Chunk, rrf_score) tuples sorted by fused score descending.
    """
    if _is_fakes() or store is None or embedder is None:
        c1 = Chunk(
            chunk_id="c_fake_001", document_id="doc_intro", source="intro.pdf", page=1,
            content_type="text", content="DocuRAG multimodal RAG hybrid architecture.", token_count=6,
        )
        c2 = Chunk(
            chunk_id="c_fake_002", document_id="doc_intro", source="intro.pdf", page=2,
            content_type="table", content="| System | Accuracy |\n| DocuRAG | 95% |", token_count=8,
        )
        return [(c1, 0.95), (c2, 0.88)]

    n_chunks = len(store.chunks)
    if n_chunks == 0:
        return []

    allowed_docs = {str(d) for d in filters.document_ids} if filters and filters.document_ids else None
    allowed_types = set(filters.content_types) if filters and filters.content_types else None

    # With filters active, search the whole index so selected documents can't be crowded out
    if allowed_docs or allowed_types:
        dense_k = bm25_k = min(n_chunks, 500)
    dense_k = min(dense_k, n_chunks)
    bm25_k = min(bm25_k, n_chunks)

    # 1. Dense retrieval
    dense_idx: List[int] = []
    if store.vector_index is not None:
        query_vec = embedder.embed_query(query)
        _, found = store.vector_index.search(query_vec, k=dense_k)
        dense_idx = [int(i) for i in np.asarray(found).ravel() if int(i) >= 0]

    # 2. Sparse retrieval (BM25); ignore zero-score "matches"
    sparse_idx: List[int] = []
    if store.bm25_index is not None:
        s_scores, s_found = store.bm25_index.search(query, k=bm25_k)
        for s, i in zip(np.asarray(s_scores).ravel(), np.asarray(s_found).ravel()):
            if int(i) >= 0 and float(s) > 0:
                sparse_idx.append(int(i))

    def keep(idx: int) -> bool:
        chunk = store.get(idx)
        if chunk is None:
            return False
        if allowed_docs and str(chunk.document_id) not in allowed_docs:
            return False
        if allowed_types and chunk.content_type not in allowed_types:
            return False
        return True

    dense_idx = [i for i in dense_idx if keep(i)]
    sparse_idx = [i for i in sparse_idx if keep(i)]

    # 3. Reciprocal Rank Fusion: score(c) = sum 1 / (rrf_k + rank_i(c))
    rrf_scores: Dict[int, float] = defaultdict(float)
    for rank, idx in enumerate(dense_idx, start=1):
        rrf_scores[idx] += 1.0 / (rrf_k + rank)
    for rank, idx in enumerate(sparse_idx, start=1):
        rrf_scores[idx] += 1.0 / (rrf_k + rank)

    ordered = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)
    return [(store.get(i), rrf_scores[i]) for i in ordered]


def _coverage_chunks(store: Any, filters: Optional[Filters], max_total: int = 12) -> List[Chunk]:
    """For broad questions: the opening chunks of every selected document."""
    if filters and filters.document_ids:
        doc_ids = [str(d) for d in filters.document_ids]
    else:
        doc_ids = list(store.doc_map.keys())
    doc_ids = [d for d in doc_ids if d in store.doc_map]
    if not doc_ids:
        return []

    per_doc = 2 if len(doc_ids) * 2 <= max_total else 1
    picked: List[Chunk] = []
    for d in doc_ids:
        chunks = [store.get(i) for i in store.doc_map[d]]
        chunks = [c for c in chunks if c is not None]
        if filters and filters.content_types:
            chunks = [c for c in chunks if c.content_type in filters.content_types]
        # earliest pages first, text before tables/images
        chunks.sort(key=lambda c: (c.page or 0, 0 if c.content_type == "text" else 1))
        picked.extend(chunks[:per_doc])
    return picked[:max_total]


def retrieve(
    query: str,
    store: Any = None,
    embedder: Any = None,
    reranker: Any = None,
    *,
    final_k: int = 6,
    filters: Optional[Filters] = None,
    query_type: str = "general",
    expand_neighbor_pages: bool = False,
) -> List[Evidence]:
    """Full retrieval: hybrid search, cross-encoder rerank, modality diversity.

    Broad "what do all my files contain" questions use coverage mode instead.
    """
    # Broad / meta questions: sample every selected document
    if store is not None and not _is_fakes() and _is_broad_query(query):
        chunks = _coverage_chunks(store, filters)
        if chunks:
            return [
                Evidence(
                    eid=f"E{i}", chunk=c, score=0.5, norm_score=0.5,
                    dense_score=0.0, sparse_score=0.0, rerank_score=0.5, combined_score=0.5,
                )
                for i, c in enumerate(chunks, start=1)
            ]

    # 1. Hybrid search; take a wider pool for the reranker
    candidates = hybrid_search(query, store=store, embedder=embedder, filters=filters)
    candidates = candidates[: max(final_k * 4, 20)]

    # 2. Rerank
    reranked = rerank(query=query, candidates=candidates, reranker=reranker, top_n=final_k)

    # 3. Modality diversity rule
    diversity_keywords = {"chart", "graph", "figure", "plot", "trend", "diagram", "image", "picture", "table"}
    query_lower = query.lower()
    if reranked and any(kw in query_lower for kw in diversity_keywords):
        if not any(c.content_type in ("image", "table") for c, _ in reranked):
            best_chunk, best_score = None, -1.0
            for chunk, score in candidates:
                if chunk.content_type in ("image", "table") and score > best_score:
                    best_chunk, best_score = chunk, score
            if best_chunk:
                reranked[-1] = (best_chunk, best_score)

    # 4. Map to Evidence
    evidence_list: List[Evidence] = []
    for idx, (chunk, score) in enumerate(reranked, start=1):
        evidence_list.append(
            Evidence(
                eid=f"E{idx}",
                chunk=chunk,
                score=score,
                norm_score=score,
                dense_score=0.0,   # not tracked after fusion
                sparse_score=0.0,
                rerank_score=score,
                combined_score=score,
            )
        )
    return evidence_list


def retrieve_for_documents(
    query: str,
    document_ids: List[str],
    store: Any = None,
    embedder: Any = None,
    reranker: Any = None,
    *,
    final_k: int = 4,
) -> List[Evidence]:
    """Scoped retrieval constrained to specific document IDs for multi-doc comparison."""
    filters = Filters(document_ids=document_ids)
    return retrieve(
        query=query, store=store, embedder=embedder, reranker=reranker,
        final_k=final_k, filters=filters,
    )