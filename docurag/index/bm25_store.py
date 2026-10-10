import re
import numpy as np
from rank_bm25 import BM25Okapi

class BM25Index:
    def __init__(self):
        self.bm25 = None
        self.corpus = []

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenization: lowercase on \w+, preserving hyphenated IDs (e.g., 'M-204').
        """
        # \w includes letters, numbers, and underscores.
        # To preserve hyphenated IDs, we include hyphens in the pattern.
        return re.findall(r'\w+(?:-\w+)*', text.lower())

    def add(self, corpus: list[str]):
        """
        Initializes/Updates the BM25 index with a provided corpus.
        """
        self.corpus = corpus
        if not corpus:
            self.bm25 = None
            return
        tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, k: int = 10):
        """
        Searches for the top k most similar documents.
        """
        if not self.bm25:
            return [], []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Sort by score and get top k
        top_k_indices = np.argsort(scores)[::-1][:k]
        return scores[top_k_indices], top_k_indices

    def reset(self):
        self.bm25 = None
        self.corpus = []
