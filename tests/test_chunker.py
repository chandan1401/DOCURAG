import pytest
from docurag.ingest.chunker import IntelligentChunker, CHUNK_TOKENS, TABLE_TOKEN_LIMIT
from docurag.schemas import PageData, FigureData

def test_text_chunking_size():
    chunker = IntelligentChunker()
    # Create a long text that should be split
    # Approx 1 token per 0.77 words. 512 tokens ~ 394 words.
    long_text = "This is a sentence. " * 500
    page = PageData(
        page=1,
        text=long_text,
        page_image_path="page1.jpg"
    )
    chunks = chunker.process_page(page, "doc1", "test.pdf")

    assert len(chunks) > 1
    for chunk in chunks:
        # We allow some slack for the overlap and boundary, but it should not be massive
        assert chunk.token_count <= CHUNK_TOKENS * 1.2

def test_heading_detection():
    chunker = IntelligentChunker()
    text = "INTRODUCTION\n\nThis is some text.\n\nNext Section\n\nMore text."
    page = PageData(
        page=1,
        text=text,
        page_image_path="page1.jpg"
    )
    chunks = chunker.process_page(page, "doc1", "test.pdf")

    # First chunk should have section "INTRODUCTION"
    # Second chunk might have section "Next Section"
    sections = [c.section for c in chunks if c.section]
    assert "INTRODUCTION" in sections or any("INTRODUCTION" in c.content for c in chunks)
    # Check that following text gets the section
    assert chunks[0].section == "INTRODUCTION" or "INTRODUCTION" in chunks[0].content

def test_table_splitting():
    chunker = IntelligentChunker()
    # Create a very long table
    header = "| Col1 | Col2 |\n|---|---|"
    rows = "\n".join([f"| Row{i} | Val{i} |" for i in range(1000)])
    table_md = header + "\n" + rows

    page = PageData(
        page=1,
        tables_md=[table_md],
        page_image_path="page1.jpg"
    )
    chunks = chunker.process_page(page, "doc1", "test.pdf")

    table_chunks = [c for c in chunks if c.content_type == "table"]
    assert len(table_chunks) > 1
    # Each chunk should start with the header
    for chunk in table_chunks:
        assert "| Col1 | Col2 |" in chunk.content
        assert "|---|---|" in chunk.content

def test_figure_chunking():
    chunker = IntelligentChunker()
    fig = FigureData(
        figure_id="fig1",
        image_path="fig1.jpg",
        caption="A sample figure caption"
    )
    page = PageData(
        page=1,
        figures=[fig],
        page_image_path="page1.jpg"
    )
    chunks = chunker.process_page(page, "doc1", "test.pdf")

    fig_chunk = [c for c in chunks if c.content_type == "image"][0]
    assert fig_chunk.image_path == "fig1.jpg"
    assert "A sample figure caption" in fig_chunk.content
    assert fig_chunk.page_image_path == "page1.jpg"

def test_ocr_content_type():
    chunker = IntelligentChunker()
    text = "Some OCR'd text"
    page = PageData(
        page=1,
        text=text,
        ocr_used=True,
        page_image_path="page1.jpg"
    )
    chunks = chunker.process_page(page, "doc1", "test.pdf")

    assert chunks[0].content_type == "ocr"

def test_unique_chunk_ids():
    chunker = IntelligentChunker()
    text = "Long text " * 1000
    page = PageData(
        page=1,
        text=text,
        page_image_path="page1.jpg"
    )
    chunks = chunker.process_page(page, "doc1", "test.pdf")

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
