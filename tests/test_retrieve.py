import pytest
import numpy as np
import os
from unittest.mock import MagicMock
from docurag.schemas import Chunk, Filters
from docurag.retrieve.hybrid import retrieve, hybrid_search
from docurag.retrieve.rerank import rerank
import docurag.retrieve.hybrid

# Disable fakes for retrieval tests
docurag.retrieve.hybrid._is_fakes = lambda: False

class MockStore:
    def __init__(self):
        self.chunks = [
            Chunk(chunk_id="chunk_0", document_id="doc_0", content="The capital of France is Paris.", content_type="text"),
            Chunk(chunk_id="chunk_1", document_id="doc_1", content="The Eiffel Tower is in Paris.", content_type="text"),
            Chunk(chunk_id="chunk_2", document_id="doc_2", content="A table showing GDP of countries.", content_type="table"),
            Chunk(chunk_id="chunk_3", document_id="doc_3", content="A graph of population trends.", content_type="image"),
            Chunk(chunk_id="chunk_4", document_id="doc_4", content="The sun is a star.", content_type="text")
        ]
        self.chunk_to_doc = {i: f"doc_{i}" for i in range(len(self.chunks))}
        self.vector_index = MagicMock()
        self.bm25_index = MagicMock()

    def get(self, idx):
        if hasattr(idx, "item"):
            idx = idx.item()
        return self.chunks[int(idx)]

def test_hybrid_search_rrf():
    store = MockStore()
    embedder = MagicMock()

    # Mock dense results: [0, 1, 2]
    store.vector_index.search.return_value = (
        np.array([0.9, 0.8, 0.7]),
        np.array([0, 1, 2])
    )
    # Mock sparse results: [1, 2, 3]
    store.bm25_index.search.return_value = (
        np.array([10.0, 5.0, 1.0]),
        np.array([1, 2, 3])
    )

    results = hybrid_search("Paris", store=store, embedder=embedder)

    # Index 1 is top in both, should be first
    assert results[0][0].chunk_id == "chunk_1"
    # Index 2 is in both, should be second
    assert results[1][0].chunk_id == "chunk_2"
    # Index 0 is only in dense, Index 3 is only in sparse
    # They should both be present
    ids = [r[0].chunk_id for r in results]
    assert "chunk_0" in ids
    assert "chunk_3" in ids

def test_rerank_normalization():
    # Use a mock reranker that returns raw logits
    reranker = MagicMock()
    reranker.predict.return_value = np.array([2.0, -2.0, 0.0])

    chunks = [
        Chunk(chunk_id="c1", content="content 1"),
        Chunk(chunk_id="c2", content="content 2"),
        Chunk(chunk_id="c3", content="content 3"),
    ]

    results = rerank("query", chunks, reranker=reranker)

    # Logit 2.0 -> sigmoid(2) ~ 0.88
    # Logit -2.0 -> sigmoid(-2) ~ 0.12
    # Logit 0.0 -> sigmoid(0) = 0.5

    assert results[0][1] > 0.8
    assert results[1][1] == 0.5
    assert results[2][1] < 0.2

def test_modality_diversity_rule():
    store = MockStore()
    embedder = MagicMock()
    reranker = MagicMock()

    query = "Show me a table of GDP"

    def mock_hybrid_search_results(query, store, embedder, **kwargs):
        res = []
        for i in range(3):
            res.append((Chunk(chunk_id=f"text_{i}", content="text content", content_type="text"), 0.9 - i*0.1))
        res.append((Chunk(chunk_id="table_1", content="table content", content_type="table"), 0.5))
        return res

    original_hybrid = docurag.retrieve.hybrid.hybrid_search
    docurag.retrieve.hybrid.hybrid_search = mock_hybrid_search_results

    def mock_rerank(query, candidates, reranker, top_n):
        return [(c, s) for c, s in candidates[:top_n]]

    original_rerank = docurag.retrieve.hybrid.rerank
    docurag.retrieve.hybrid.rerank = mock_rerank

    try:
        evidence = retrieve(query, store=store, embedder=embedder, reranker=reranker, final_k=3)

        assert any(e.chunk.content_type == "table" for e in evidence)
        assert evidence[-1].chunk.chunk_id == "table_1"
    finally:
        # Restore original functions
        docurag.retrieve.hybrid.hybrid_search = original_hybrid
        docurag.retrieve.hybrid.rerank = original_rerank

def test_exact_id_filter():
    store = MockStore()
    embedder = MagicMock()

    store.vector_index.search.return_value = (np.array([0.9]), np.array([0]))
    store.bm25_index.search.return_value = (np.array([10.0]), np.array([0]))

    filters = Filters(document_ids=["doc_100"]) # Not present in our MockStore

    results = hybrid_search("query", store=store, embedder=embedder, filters=filters)

    # Should be empty because doc_0 != doc_100
    assert len(results) == 0
