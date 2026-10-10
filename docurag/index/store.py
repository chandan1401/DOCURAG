import json
import os
import threading
import numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from docurag.index.vector_store import FAISSIndex
from docurag.index.bm25_store import BM25Index
from docurag.schemas import Chunk

class ChunkStore:
    def __init__(self, embedder, persistence_path: str = "storage/index"):
        self.embedder = embedder
        self.persistence_path = persistence_path
        self.lock = threading.Lock()

        # Detect mode change and prevent cache poisoning
        from docurag.config import get_settings
        settings = get_settings()
        self._last_known_mode = settings.DOCURAG_FAKES

        # If we are starting up and the index exists but we are in a different mode than
        # what might have been used to create it, we should ideally know.
        # However, since we don't persist the mode in metadata.json yet,
        # we'll handle the automatic wipe in a dedicated method or here.
        self._check_and_wipe_stale_index()

        # In-memory storage: List of full Chunk objects
        self.chunks: List[Chunk] = []
        self.doc_map = defaultdict(list) # doc_id -> list of chunk indices
        self.chunk_to_doc = {} # chunk_index -> doc_id

        # Indices
        self.vector_index = None # Initialized on first add
        self.bm25_index = BM25Index()

    def _check_and_wipe_stale_index(self):
        """Wipe index and cache if DOCURAG_FAKES changed from 1 to 0."""
        import shutil
        import logging
        from docurag.config import get_settings
        logger = logging.getLogger(__name__)
        settings = get_settings()

        # We store the mode in a small marker file to detect changes across restarts
        mode_marker = Path(self.persistence_path) / ".mode_marker"
        current_mode = "1" if settings.DOCURAG_FAKES else "0"

        if mode_marker.exists():
            try:
                old_mode = mode_marker.read_text().strip()
                if old_mode == "1" and current_mode == "0":
                    logger.warning(
                        "Detected switch from FAKE to LIVE mode. Wiping stale index and cache to prevent poisoning."
                    )
                    # Wipe index
                    if Path(self.persistence_path).exists():
                        shutil.rmtree(self.persistence_path)
                    # Wipe cache
                    if settings.cache_dir.exists():
                        shutil.rmtree(settings.cache_dir)
            except Exception as e:
                logger.error("Failed to check mode marker: %s", e)

        # Update marker
        os.makedirs(self.persistence_path, exist_ok=True)
        mode_marker.write_text(current_mode)

    def add_document(self, doc_id: str, chunks: List[Chunk]):
        """
        Adds a document's chunks to the store. Uses a lock for concurrency.
        """
        if not chunks:
            return
            
        with self.lock:
            start_idx = len(self.chunks)

            # Store full Chunk objects
            self.chunks.extend(chunks)

            # Update maps
            for i, chunk in enumerate(chunks):
                global_idx = start_idx + i
                self.doc_map[doc_id].append(global_idx)
                self.chunk_to_doc[global_idx] = doc_id

            # Update Vector Index (embed the content)
            texts = [c.content for c in chunks]
            embeddings = self.embedder.embed_chunks(texts)
            if self.vector_index is None:
                self.vector_index = FAISSIndex(embeddings.shape[1])
            self.vector_index.add(embeddings)

            # Update BM25 Index (rebuilds using the content)
            all_texts = [c.content for c in self.chunks]
            self.bm25_index.add(all_texts)

    def remove_document(self, doc_id: str):
        """
        Removes all chunks associated with a document.
        """
        with self.lock:
            if doc_id not in self.doc_map:
                return

            # Get indices to remove
            indices_to_remove = set(self.doc_map[doc_id])

            # Filter chunks and maps
            new_chunks = []
            new_chunk_to_doc = {}
            new_doc_map = defaultdict(list)

            current_idx = 0
            for i, chunk in enumerate(self.chunks):
                if i not in indices_to_remove:
                    new_chunks.append(chunk)
                    new_chunk_to_doc[current_idx] = self.chunk_to_doc[i]
                    new_doc_map[self.chunk_to_doc[i]].append(current_idx)
                    current_idx += 1

            self.chunks = new_chunks
            self.chunk_to_doc = new_chunk_to_doc
            self.doc_map = new_doc_map

            # Rebuild indices
            if self.vector_index:
                all_texts = [c.content for c in self.chunks]
                all_embeddings = self.embedder.embed_chunks(all_texts)
                self.vector_index.reset()
                self.vector_index.add(all_embeddings)

            all_texts = [c.content for c in self.chunks]
            self.bm25_index.add(all_texts)

    def get(self, idx: int) -> Optional[Chunk]:
        """
        Retrieves a full Chunk object by index.
        """
        if 0 <= idx < len(self.chunks):
            return self.chunks[idx]
        return None

    def all_chunks(self) -> List[Chunk]:
        """
        Returns all stored Chunk objects.
        """
        return self.chunks

    def stats(self) -> Dict[str, Any]:
        """
        Returns counts per doc and content type.
        """
        res = {}
        for doc_id, indices in self.doc_map.items():
            counts = defaultdict(int)
            for idx in indices:
                chunk = self.chunks[idx]
                counts[chunk.content_type] += 1
            res[doc_id] = {
                "total": len(indices),
                "breakdown": dict(counts)
            }
        return res

    def save(self):
        """
        Persists chunks as JSONL, FAISS index, and BM25 index.
        """
        os.makedirs(self.persistence_path, exist_ok=True)

        # Save metadata
        metadata = {
            "doc_map": {k: v for k, v in self.doc_map.items()},
            "chunk_to_doc": self.chunk_to_doc
        }
        with open(os.path.join(self.persistence_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        # Save chunks as JSONL
        with open(os.path.join(self.persistence_path, "chunks.jsonl"), "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk.model_dump()) + "\n")

        # Save FAISS index
        if self.vector_index:
            import faiss
            faiss.write_index(self.vector_index.index, os.path.join(self.persistence_path, "vector.index"))

    def load(self):
        """
        Loads persistence data.
        """
        if not os.path.exists(self.persistence_path):
            return

        # Load chunks
        chunks_path = os.path.join(self.persistence_path, "chunks.jsonl")
        if os.path.exists(chunks_path):
            with open(chunks_path, "r", encoding="utf-8") as f:
                self.chunks = [Chunk.model_validate(json.loads(line)) for line in f]

        # Load metadata
        meta_path = os.path.join(self.persistence_path, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)
                self.doc_map = defaultdict(list, metadata["doc_map"])
                self.chunk_to_doc = {int(k): v for k, v in metadata["chunk_to_doc"].items()}

        # Load FAISS index
        vec_path = os.path.join(self.persistence_path, "vector.index")
        if os.path.exists(vec_path):
            import faiss
            index = faiss.read_index(vec_path)
            if self.vector_index:
                self.vector_index.index = index
            else:
                dim = index.d
                self.vector_index = FAISSIndex(dim)
                self.vector_index.index = index

        # Rebuild BM25 index
        all_texts = [c.content for c in self.chunks]
        self.bm25_index.add(all_texts)
