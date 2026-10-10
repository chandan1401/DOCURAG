"""Document ingestion module (PDF, OCR, vision, chunking)."""

from docurag.ingest.pdf_loader import extract_pages, iter_pages, load_image, load_pdf
from docurag.ingest.ocr import ocr_page
from docurag.ingest.vision import describe_figure, extract_figures, transcribe_page
from docurag.ingest.triage import classify_page
from docurag.ingest.text import load_docx, load_txt
from docurag.ingest.coordinator import (
    generate_document_id,
    ingest_directory,
    ingest_document,
    iter_document_pages,
)

__all__ = [
    "load_pdf",
    "load_image",
    "extract_pages",
    "iter_pages",
    "ocr_page",
    "extract_figures",
    "describe_figure",
    "transcribe_page",
    "classify_page",
    "load_docx",
    "load_txt",
    "generate_document_id",
    "iter_document_pages",
    "ingest_document",
    "ingest_directory",
]
