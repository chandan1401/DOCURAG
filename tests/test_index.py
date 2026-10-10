import pytest
import numpy as np
from docurag.index.embedder import Embedder
from docurag.index.store import ChunkStore

def test_index_flow():
    from docurag.schemas import Chunk
    embedder = Embedder()
    store = ChunkStore(embedder)

    # Test adding documents
    doc1_chunks = [
        Chunk(chunk_id="c1", document_id="doc1", content="The quick brown fox jumps over the lazy dog."),
        Chunk(chunk_id="c2", document_id="doc1", content="This is a test document.")
    ]
    doc2_chunks = [
        Chunk(chunk_id="c3", document_id="doc2", content="M-204 is a specific part number for the engine."),
        Chunk(chunk_id="c4", document_id="doc2", content="Another chunk about M-204.")
    ]

    store.add_document("doc1", doc1_chunks)
    store.add_document("doc2", doc2_chunks)

    # Test stats
    stats = store.stats()
    assert stats["doc1"]["total"] == 2
    assert stats["doc2"]["total"] == 2

    # Test removal
    store.remove_document("doc1")
    stats = store.stats()
    assert "doc1" not in stats
    assert stats["doc2"]["total"] == 2
    assert len(store.all_chunks()) == 2

    # Test BM25 for exact ID tokens
    # Query for M-204 should find doc2 chunks
    bm25_scores, indices = store.bm25_index.search("M-204")
    assert len(indices) > 0
    assert "M-204" in store.get(indices[0]).content

    # Test FAISS search
    query_vec = embedder.embed_query("Where is the engine part?")
    dist, idx = store.vector_index.search(query_vec, k=1)
    assert idx[0] < len(store.all_chunks())
    assert "M-204" in store.get(idx[0]).content

def test_persistence():
    from docurag.schemas import Chunk
    embedder = Embedder()
    store = ChunkStore(embedder, persistence_path="test_index_data")

    chunks = [
        Chunk(chunk_id="c1", document_id="doc1", content="Chunk 1"),
        Chunk(chunk_id="c2", document_id="doc1", content="Chunk 2")
    ]
    store.add_document("doc1", chunks)
    store.save()

    new_store = ChunkStore(embedder, persistence_path="test_index_data")
    new_store.load()

    assert len(new_store.all_chunks()) == 2
    assert new_store.stats()["doc1"]["total"] == 2

    # Cleanup
    import shutil
    import os
    shutil.rmtree("test_index_data")

if __name__ == "__main__":
    pytest.main([__file__])
