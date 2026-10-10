import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)
        self.query_prefix = "Represent this sentence for searching relevant passages: "

    def embed_chunks(self, chunks: list[str]) -> np.ndarray:
        """
        Embeds a list of text chunks into float32 L2-normalized vectors.
        """
        embeddings = self.model.encode(
            chunks,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embeds a query string into a float32 L2-normalized vector.
        Queries must be prefixed for the BGE model.
        """
        full_query = self.query_prefix + query
        embedding = self.model.encode(
            full_query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embedding.astype(np.float32).reshape(1, -1)
