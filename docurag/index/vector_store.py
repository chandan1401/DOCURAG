import faiss
import numpy as np

class FAISSIndex:
    def __init__(self, dimension: int):
        # IndexFlatIP is used for Inner Product.
        # Since vectors are L2-normalized, Inner Product is equivalent to Cosine Similarity.
        self.index = faiss.IndexFlatIP(dimension)

    def add(self, vectors: np.ndarray):
        """
        Adds vectors to the index. Vectors must be float32 and L2-normalized.
        """
        self.index.add(vectors)

    def search(self, query_vector: np.ndarray, k: int = 10):
        """
        Searches for the top k most similar vectors.
        """
        distances, indices = self.index.search(query_vector, k)
        return distances[0], indices[0]

    def reset(self):
        self.index = faiss.IndexFlatIP(self.index.d)
