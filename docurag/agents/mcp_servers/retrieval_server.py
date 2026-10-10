"""Retrieval MCP tool server."""

from typing import Any, Dict, List, Optional


def hybrid_search(
    query: str,
    k: int = 6,
    document_ids: Optional[List[str]] = None,
    content_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Execute hybrid search returning JSON ChunkHits.

    Returns:
        List of ChunkHit dicts: chunk_id, document, page, content_type, snippet, score.
    """
    return [
        {
            "chunk_id": "c_fake_001",
            "document": "sample_document.pdf",
            "page": 1,
            "content_type": "text",
            "snippet": f"Retrieved snippet matching query '{query}'",
            "score": 0.95,
        }
    ]


def search_tables(
    query: str,
    k: int = 4,
    document_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve table chunks matching query."""
    return hybrid_search(query=query, k=k, document_ids=document_ids, content_types=["table"])


def search_images(
    query: str,
    k: int = 4,
    document_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve figure/chart chunks matching query."""
    return hybrid_search(query=query, k=k, document_ids=document_ids, content_types=["image"])


if __name__ == "__main__":
    import sys
    print("retrieval_server running in direct mode", file=sys.stderr)
