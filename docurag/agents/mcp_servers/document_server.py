"""Document inspection MCP tool server."""

from typing import Any, Dict, List


def list_documents() -> List[Dict[str, Any]]:
    """List all ingested documents and their status.

    Returns:
        List of dicts with document_id, filename, pages, status.
    """
    return [
        {
            "document_id": "doc_001",
            "filename": "sample_document.pdf",
            "pages": 1,
            "status": "ready",
        }
    ]


def get_page_text(document_id: str, page: int) -> Dict[str, Any]:
    """Retrieve full extracted text and content types for a page."""
    return {
        "text": f"Sample extracted text for document {document_id} page {page}.",
        "content_types": ["text"],
    }


def get_page_image(document_id: str, page: int) -> Dict[str, Any]:
    """Get relative disk path to the rendered page PNG image."""
    return {"path": f"storage/pages/{document_id}_p{page}.png"}


if __name__ == "__main__":
    import sys
    print("document_server running in direct mode", file=sys.stderr)
