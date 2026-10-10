"""PyMuPDF-based PDF extraction helpers."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

from docurag.config import get_settings
from docurag.ingest.ocr import ocr_page
from docurag.ingest.tables import extract_tables_from_page, mask_table_text
from docurag.ingest.vision import extract_figures, transcribe_page
from docurag.schemas import PageData

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Normalize extracted page text without destroying paragraph breaks."""
    text = re.sub(r"[ \t]+", " ", text or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def render_page_image(page: Any, document_id: str, page_number: int) -> str:
    """Render a PDF page to the configured page preview directory."""
    settings = get_settings()
    settings.ensure_dirs()
    out_path = settings.pages_dir / f"{document_id}_p{page_number}.png"
    zoom = settings.RENDER_DPI / 72.0
    matrix = page.parent.__class__.Matrix(zoom, zoom) if False else None
    try:
        import fitz

        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    except Exception:
        pix = page.get_pixmap(alpha=False)
    pix.save(str(out_path))
    return str(out_path)


def extract_text_blocks(page: Any, table_bboxes: Sequence[Sequence[float]] = ()) -> str:
    """Extract sorted, table-deduplicated text blocks from a PyMuPDF page."""
    try:
        blocks = page.get_text("blocks", sort=True)
    except TypeError:
        blocks = page.get_text("blocks")
    filtered = mask_table_text(list(blocks), [list(b) for b in table_bboxes])
    text_parts: List[str] = []
    for block in filtered:
        if len(block) >= 5 and str(block[4]).strip():
            text_parts.append(str(block[4]).strip())
    return clean_text("\n\n".join(text_parts) or page.get_text("text"))


def page_char_density(page: Any, text: str) -> float:
    """Approximate extracted character density per square point."""
    area = max(float(page.rect.width) * float(page.rect.height), 1.0)
    return len((text or "").strip()) / area


def extract_pdf_page(
    doc_path: Path,
    document_id: str,
    page: Any,
    page_index: int,
    pdfplumber_page: Any = None,
) -> PageData:
    """Extract a single PDF page into the frozen PageData contract."""
    settings = get_settings()
    page_number = page_index + 1
    page_image_path = render_page_image(page, document_id, page_number)
    tables_md, table_bboxes = extract_tables_from_page(page, pdfplumber_page)
    text = extract_text_blocks(page, table_bboxes)
    ocr_used = False

    if len(text.strip()) < settings.OCR_MIN_CHARS:
        ocr_text, confidence = ocr_page(page_image_path)
        if confidence < 0.5:
            vision_text = transcribe_page(page_image_path)
            if vision_text:
                ocr_text = vision_text
        if ocr_text.strip():
            text = clean_text(ocr_text)
            ocr_used = True

    figures = extract_figures(doc_path, page_number=page_number, document_id=document_id)

    return PageData(
        page=page_number,
        text=text,
        ocr_used=ocr_used,
        tables_md=tables_md,
        figures=figures,
        page_image_path=page_image_path,
    )


def iter_pdf_pages(path: Path, document_id: str) -> Iterable[PageData]:
    """Yield PDF pages with per-page fail-soft containment."""
    import fitz

    pdfplumber_pdf = None
    try:
        import pdfplumber

        pdfplumber_pdf = pdfplumber.open(str(path))
    except Exception as exc:
        logger.debug("pdfplumber unavailable for %s: %s", path, exc)

    doc = fitz.open(str(path))
    try:
        for index in range(doc.page_count):
            try:
                plumber_page = None
                if pdfplumber_pdf is not None and index < len(pdfplumber_pdf.pages):
                    plumber_page = pdfplumber_pdf.pages[index]
                yield extract_pdf_page(path, document_id, doc[index], index, plumber_page)
            except Exception as exc:
                logger.warning("Failed to extract page %s from %s: %s", index + 1, path, exc)
                yield PageData(page=index + 1, text="", tables_md=[], figures=[], page_image_path="")
    finally:
        doc.close()
        if pdfplumber_pdf is not None:
            pdfplumber_pdf.close()
