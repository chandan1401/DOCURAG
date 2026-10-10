"""Page triage heuristic classifier for ingestion workflow."""

from typing import Any, Literal


def classify_page(page: Any) -> Literal["text", "scanned", "chart_like", "mixed"]:
    """Deterministically classify page modality based on text density and visual area ratio.

    Args:
        page: Page object (PyMuPDF page or PageData).

    Returns:
        One of 'text', 'scanned', 'chart_like', or 'mixed'.
    """
    text = getattr(page, "text", "") or ""
    figures = getattr(page, "figures", []) or []
    tables = getattr(page, "tables_md", []) or []
    ocr_used = bool(getattr(page, "ocr_used", False))

    if ocr_used and len(text.strip()) < 200 and not figures and not tables:
        return "scanned"
    if figures and len(text.strip()) < 120 and not tables:
        return "chart_like"
    if figures or tables or ocr_used:
        return "mixed"
    return "text"
