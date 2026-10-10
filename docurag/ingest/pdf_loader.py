"""PDF and image document ingestion loaders."""

import os
from pathlib import Path
from typing import Iterator, List, Union

from docurag.config import get_settings
from docurag.schemas import FigureData, PageData
from docurag.ingest.ocr import ocr_page
from docurag.ingest.pdf import iter_pdf_pages


def _is_fakes() -> bool:
    if os.getenv("DOCURAG_FAKES") in ("1", "true", "True"):
        return True
    try:
        return get_settings().DOCURAG_FAKES
    except Exception:
        return False


def load_pdf(path: Union[Path, str], document_id: str) -> List[PageData]:
    """Ingest a PDF file and extract per-page text, tables, figures, and render image.

    Args:
        path: Path to the target PDF file on disk.
        document_id: Unique identifier string for the document.

    Returns:
        List of PageData objects representing extracted content per page.

    Raises:
        FileNotFoundError: If the specified PDF file does not exist.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        if _is_fakes():
            return [
                PageData(
                    page=1,
                    text=f"Simulated content extracted from PDF: {path_obj.name}",
                    ocr_used=False,
                    tables_md=[],
                    figures=[],
                    page_image_path=f"storage/pages/{document_id}_p1.png",
                )
            ]
        raise FileNotFoundError(f"PDF document not found: {path}")

    return list(iter_pdf_pages(path_obj, document_id=document_id))


def load_image(path: Union[Path, str], document_id: str) -> List[PageData]:
    """Ingest a standalone image file as a single-page document.

    Args:
        path: Path to the image file (PNG, JPG, WEBP).
        document_id: Unique identifier string for the document.

    Returns:
        List containing a single PageData object representing the image.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        if not _is_fakes():
            raise FileNotFoundError(f"Image document not found: {path}")
        text = "Simulated OCR transcription of standalone image."
    else:
        text, _ = ocr_page(path_obj)
        if not text and _is_fakes():
            text = "Simulated OCR transcription of standalone image."

    fig = FigureData(
        figure_id=f"{document_id}_fig1",
        image_path=str(path_obj),
        page=1,
        bbox=[0.0, 0.0, 1.0, 1.0],
        caption="Standalone image figure",
    )
    return [
        PageData(
            page=1,
            text=text,
            ocr_used=True,
            tables_md=[],
            figures=[fig],
            page_image_path=str(path_obj),
        )
    ]


def extract_pages(path: Union[Path, str], document_id: str) -> List[PageData]:
    """Extract pages from a document (convenience alias for load_pdf)."""
    return load_pdf(path=path, document_id=document_id)


def iter_pages(path: Union[Path, str], document_id: str = "") -> Iterator[PageData]:
    """Incrementally yield PageData objects page-by-page for progress reporting.

    Args:
        path: Path to document.
        document_id: Document ID.

    Yields:
        PageData for each processed page.
    """
    pages = load_pdf(path=path, document_id=document_id or Path(path).stem)
    for page in pages:
        yield page
