"""FastMCP tool servers (document, retrieval, analysis)."""

from docurag.agents.mcp_servers.document_server import (
    get_page_image,
    get_page_text,
    list_documents,
)
from docurag.agents.mcp_servers.retrieval_server import (
    hybrid_search,
    search_images,
    search_tables,
)
from docurag.agents.mcp_servers.analysis_server import calculate, extract_chart_values

__all__ = [
    "list_documents",
    "get_page_text",
    "get_page_image",
    "hybrid_search",
    "search_tables",
    "search_images",
    "calculate",
    "extract_chart_values",
]
