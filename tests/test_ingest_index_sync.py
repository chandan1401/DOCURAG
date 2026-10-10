from pathlib import Path

import pytest

import docurag.ingest.coordinator as coordinator
import docurag.pipeline as pipeline
from docurag.schemas import Chunk, PageData


class MemoryPersistentStore:
    saved_chunks = []

    def __init__(self, embedder, persistence_path):
        self.chunks = []

    def load(self):
        self.chunks = list(self.saved_chunks)

    def remove_document(self, document_id):
        self.chunks = [chunk for chunk in self.chunks if chunk.document_id != document_id]

    def add_document(self, document_id, chunks):
        self.chunks.extend(chunks)

    def save(self):
        type(self).saved_chunks = list(self.chunks)


def test_ingest_appends_documents_and_replaces_reingested_document(tmp_path, monkeypatch):
    storage = tmp_path / "storage"
    monkeypatch.setenv("DOCURAG_STORAGE", str(storage))
    monkeypatch.setenv("DOCURAG_FAKES", "1")
    monkeypatch.setattr(coordinator, "ChunkStore", MemoryPersistentStore)
    monkeypatch.setattr(coordinator, "Embedder", lambda **kwargs: object())
    monkeypatch.setattr(
        coordinator,
        "iter_document_pages",
        lambda path, document_id: [PageData(text=f"content for {document_id}")],
    )
    MemoryPersistentStore.saved_chunks = []

    first_file = tmp_path / "first.pdf"
    second_file = tmp_path / "second.pdf"
    first_file.touch()
    second_file.touch()

    coordinator.ingest_document(first_file, document_id="doc-1")
    coordinator.ingest_document(second_file, document_id="doc-2")
    coordinator.ingest_document(first_file, document_id="doc-1")

    assert {chunk.document_id for chunk in MemoryPersistentStore.saved_chunks} == {"doc-1", "doc-2"}
    assert len([chunk for chunk in MemoryPersistentStore.saved_chunks if chunk.document_id == "doc-1"]) == 1


def test_ingest_refreshes_an_already_loaded_query_engine(tmp_path, monkeypatch):
    storage = tmp_path / "storage"
    monkeypatch.setenv("DOCURAG_STORAGE", str(storage))
    monkeypatch.setenv("DOCURAG_FAKES", "1")
    monkeypatch.setattr(pipeline, "_ENGINE_CACHE", {"engine": {"embedder": object(), "store": object()}})
    monkeypatch.setattr(coordinator, "ingest_document", lambda *args, **kwargs: {"status": "ready"})

    refreshed_stores = []

    class RefreshedStore:
        def __init__(self, embedder, persistence_path):
            self.embedder = embedder
            self.persistence_path = persistence_path
            self.loaded = False
            refreshed_stores.append(self)

        def load(self):
            self.loaded = True

    monkeypatch.setattr(pipeline, "ChunkStore", RefreshedStore)
    doc_path = tmp_path / "uploaded.pdf"
    doc_path.touch()
    existing_embedder = pipeline._ENGINE_CACHE["engine"]["embedder"]

    pipeline.ingest_file(str(doc_path), document_id="doc-3")

    engine = pipeline._ENGINE_CACHE["engine"]
    assert len(refreshed_stores) == 1
    assert refreshed_stores[0].loaded
    assert refreshed_stores[0].embedder is existing_embedder
    assert engine["store"] is refreshed_stores[0]
