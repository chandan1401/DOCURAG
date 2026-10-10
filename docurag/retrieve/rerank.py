"""Cross-encoder semantic reranking."""

import numpy as np
from typing import Any, List, Tuple, Union
from sentence_transformers import CrossEncoder
from docurag.schemas import Chunk

def sigmoid(x: float) -> float:
    """Compute sigmoid for score normalization (0..1)."""
    return 1 / (1 + np.exp(-x))

def rerank(
    query: str,
    candidates: Union[List[Tuple[Chunk, float]], List[Chunk]],
    reranker: Any = None,
    top_n: int = 6,
) -> List[Tuple[Chunk, float]]:
    """Rerank candidates using Cross-Encoder model.

    Args:
        query: User query string.
        candidates: Candidate chunks or (Chunk, score) tuples.
        reranker: CrossEncoder reranker instance.
        top_n: Number of top results to return.

    Returns:
        List of reranked (Chunk, norm_score) tuples.
    """
    if not candidates:
        return []

    # Extract chunks from (Chunk, score) tuples if necessary
    chunks = [item[0] if isinstance(item, tuple) else item for item in candidates]

    if reranker is None:
        # Fallback: preserve original order but cap to top_n
        # In real scenario, this should probably log a warning
        results = []
        for i in range(min(len(candidates), top_n)):
            chunk = chunks[i]
            score = 1.0 - (i * 0.01)
            results.append((chunk, score))
        return results

    # Pair query with each chunk content
    pairs = [[query, chunk.content] for chunk in chunks]

    # Get raw logits from CrossEncoder
    scores = reranker.predict(pairs)

    # Normalise scores using sigmoid
    norm_scores = [float(sigmoid(s)) for s in scores]

    # Combine and sort
    ranked = sorted(
        zip(chunks, norm_scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_n]
