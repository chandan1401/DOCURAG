"""Milestone M1 ingestion tests."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from docurag.ingest import (
    classify_page,
    describe_figure,
    extract_figures,
    generate_document_id,
    ingest_directory,
    ingest_document,
    iter_document_pages,
    load_docx,
    load_image,
    load_pdf,
    load_txt,
    ocr_page,
)
from docurag.ingest.tables import is_inside_table, mask_table_text, table_to_markdown
from docurag.schemas import FigureDescription, PageData


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCURAG_FAKES", "1")
    monkeypatch.setenv("DOCURAG_STORAGE", str(tmp_path / "storage"))
    yield


def test_table_to_markdown_formats_and_escapes_pipes():
    md = table_to_markdown([["Metric", "Value"], ["A|B", "42"]])
    assert md is not None
    assert "| Metric | Value |" in md
    assert "A\\|B" in md


def test_table_to_markdown_rejects_invalid_tables():
    assert table_to_markdown([]) is None
    assert table_to_markdown([["Only one row"]]) is None
    assert table_to_markdown([["A"], ["B"]]) is None


def test_is_inside_table_detects_overlap():
    assert is_inside_table((10, 10, 50, 50), [[0, 0, 100, 100]])
    assert not is_inside_table((120, 120, 180, 180), [[0, 0, 100, 100]])


def test_mask_table_text_removes_overlapping_blocks():
    blocks = [(0, 0, 100, 40, "table text", 0, 0), (0, 100, 100, 140, "body", 1, 0)]
    filtered = mask_table_text(blocks, [[0, 0, 100, 50]])
    assert len(filtered) == 1
    assert filtered[0][4] == "body"


def test_load_pdf_extracts_text_and_renders_page(synthetic_text_pdf):
    pages = load_pdf(synthetic_text_pdf, "doc_text")
    assert len(pages) == 1
    assert "DocuRAG Architecture Overview" in pages[0].text
    assert pages[0].ocr_used is False
    assert Path(pages[0].page_image_path).exists()


def test_load_pdf_triggers_ocr_for_scanned_page(synthetic_scanned_pdf):
    pages = load_pdf(synthetic_scanned_pdf, "doc_scanned")
    assert len(pages) == 1
    assert pages[0].ocr_used is True
    assert "Simulated OCR" in pages[0].text


def test_extract_figures_saves_embedded_chart(synthetic_chart_pdf):
    figures = extract_figures(synthetic_chart_pdf, document_id="doc_chart")
    assert len(figures) >= 1
    assert Path(figures[0].image_path).exists()
    assert figures[0].bbox is not None


def test_load_pdf_includes_figures(synthetic_chart_pdf):
    pages = load_pdf(synthetic_chart_pdf, "doc_chart")
    assert len(pages) == 1
    assert pages[0].figures
    assert pages[0].figures[0].page == 1


def test_load_image_returns_single_page_with_figure(synthetic_png_file):
    pages = load_image(synthetic_png_file, "doc_image")
    assert len(pages) == 1
    assert pages[0].ocr_used is True
    assert len(pages[0].figures) == 1
    assert pages[0].figures[0].image_path.endswith(".png")


def test_ocr_page_fake_is_deterministic(synthetic_png_file):
    text, confidence = ocr_page(synthetic_png_file)
    assert "Simulated OCR" in text
    assert confidence == pytest.approx(0.95)


def test_describe_figure_fake_flattens(synthetic_png_file):
    desc = describe_figure(synthetic_png_file)
    assert isinstance(desc, FigureDescription)
    assert "Simulated Metric Comparison" in desc.to_flattened_text()


def test_load_txt_virtual_pagination(tmp_path):
    txt = tmp_path / "notes.txt"
    txt.write_text(" ".join(f"word{i}" for i in range(1100)), encoding="utf-8")
    pages = load_txt(txt, "doc_txt")
    assert len(pages) == 3
    assert all(Path(page.page_image_path).exists() for page in pages)


def test_load_docx_zip_fallback(tmp_path):
    docx = tmp_path / "mini.docx"
    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>DocuRAG DOCX paragraph</w:t></w:r></w:p></w:body></w:document>"
    )
    with zipfile.ZipFile(docx, "w") as zf:
        zf.writestr("word/document.xml", xml)
    pages = load_docx(docx, "doc_docx")
    assert len(pages) == 1
    assert "DocuRAG DOCX paragraph" in pages[0].text


def test_generate_document_id_is_content_stable(tmp_path):
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "a-copy.txt"
    file_a.write_text("same", encoding="utf-8")
    file_b.write_text("same", encoding="utf-8")
    assert generate_document_id(file_a).split("_")[-1] == generate_document_id(file_b).split("_")[-1]


def test_iter_document_pages_routes_pdf(synthetic_text_pdf):
    pages = list(iter_document_pages(synthetic_text_pdf))
    assert len(pages) == 1
    assert isinstance(pages[0], PageData)


def test_ingest_document_report_omits_crash(synthetic_text_pdf):
    report = ingest_document(synthetic_text_pdf, collection_id="m1")
    assert report["status"] == "ready"
    assert report["pages"] == 1
    assert report["chunks_indexed"] >= 1
    assert report["collection_id"] == "m1"


def test_ingest_directory_collects_supported_files(synthetic_corpus):
    report = ingest_directory(synthetic_corpus["text"].parent, collection_id="batch")
    assert report["status"] == "ready"
    assert report["files_ingested"] >= 5
    assert report["total_chunks"] >= 5


def test_ingest_directory_continues_after_bad_file(tmp_path, synthetic_text_pdf):
    work_dir = tmp_path / "corpus"
    work_dir.mkdir()
    bad = work_dir / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4 broken")
    good = work_dir / "good.pdf"
    good.write_bytes(Path(synthetic_text_pdf).read_bytes())
    report = ingest_directory(work_dir, collection_id="resilient")
    assert report["files_ingested"] == 1
    assert len(report["errors"]) == 1


def test_classify_page_modalities():
    assert classify_page(PageData(text="hello " * 100)) == "text"
    assert classify_page(PageData(text="short", ocr_used=True)) == "scanned"
    assert classify_page(PageData(text="short", figures=[{"figure_id": "f1", "image_path": "x"}])) == "chart_like"
    assert classify_page(PageData(text="body", tables_md=["| A | B |\n|---|---|"])) == "mixed"


def test_load_pdf_missing_file_uses_fake_stub_in_fake_mode(tmp_path):
    pages = load_pdf(tmp_path / "missing.pdf", "doc_missing")
    assert len(pages) == 1
    assert "Simulated content" in pages[0].text
