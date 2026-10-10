"""Page-level semantic chunking with table and figure preservation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from docurag.config import get_settings
from docurag.schemas import Chunk, FigureData, PageData

TABLE_SPLIT_TOKENS = 1500
CAPTION_RE = re.compile(r"^\s*(figure|fig\.|chart|table)\b", re.IGNORECASE)
TERMINAL_PUNCTUATION = (".", "?", "!", ":", ";")


def approx_tokens(text: str) -> int:
    """Approximate tokenizer length using the project-wide words * 1.3 rule."""
    return int(len((text or "").split()) * 1.3)


def is_heading(line: str) -> bool:
    """Heuristically detect a section heading in extracted page text."""
    clean = line.strip().strip("#").strip()
    if not clean:
        return False
    words = clean.split()
    if len(words) > 12 or approx_tokens(clean) > 24:
        return False
    if clean.endswith(TERMINAL_PUNCTUATION):
        return False
    alpha_chars = [ch for ch in clean if ch.isalpha()]
    if not alpha_chars:
        return False
    return clean.isupper() or clean.istitle() or line.lstrip().startswith("#")


def extract_caption_lines(text: str) -> List[str]:
    """Return figure/chart/table caption-looking lines from page text."""
    captions: List[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if CAPTION_RE.match(stripped):
            captions.append(stripped)
    return captions


@dataclass
class Paragraph:
    text: str
    section: Optional[str]


def _paragraphs_with_sections(text: str) -> List[Paragraph]:
    current_section: Optional[str] = None
    paragraphs: List[Paragraph] = []
    buffer: List[str] = []

    def flush() -> None:
        nonlocal buffer
        content = " ".join(part.strip() for part in buffer if part.strip()).strip()
        if content:
            paragraphs.append(Paragraph(text=content, section=current_section))
        buffer = []

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if is_heading(line):
            flush()
            current_section = line.strip().strip("#").strip()
            continue
        buffer.append(line)

    flush()
    return paragraphs


def _split_long_words(words: Sequence[str], max_tokens: int, overlap_tokens: int) -> List[str]:
    if not words:
        return []
    max_words = max(1, int(max_tokens / 1.3))
    overlap_words = max(0, min(int(overlap_tokens / 1.3), max_words - 1))
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_words)
        chunks.append(" ".join(words[start:end]).strip())
        if end >= len(words):
            break
        start = end - overlap_words
    return chunks


def _text_chunks(page: PageData, max_tokens: int, overlap_tokens: int) -> List[Tuple[str, Optional[str]]]:
    paragraphs = _paragraphs_with_sections(page.text)
    chunks: List[Tuple[str, Optional[str]]] = []
    current: List[str] = []
    current_section: Optional[str] = None
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        content = "\n\n".join(current).strip()
        if content:
            chunks.append((content, current_section))
        current = []
        current_tokens = 0

    for paragraph in paragraphs:
        p_tokens = approx_tokens(paragraph.text)
        if p_tokens > max_tokens:
            flush()
            for split in _split_long_words(paragraph.text.split(), max_tokens, overlap_tokens):
                chunks.append((split, paragraph.section))
            current_section = paragraph.section
            continue

        if current and (paragraph.section != current_section or current_tokens + p_tokens > max_tokens):
            flush()

        current_section = paragraph.section
        current.append(paragraph.text)
        current_tokens += p_tokens

    flush()
    return chunks


def _table_lines(table_md: str) -> Tuple[List[str], List[str]]:
    lines = [line.strip() for line in (table_md or "").splitlines() if line.strip()]
    if len(lines) < 3:
        return lines[:2], lines[2:]
    return lines[:2], lines[2:]


def _split_table(table_md: str, max_tokens: int = TABLE_SPLIT_TOKENS) -> List[str]:
    if approx_tokens(table_md) <= max_tokens:
        return [table_md.strip()] if table_md.strip() else []

    header, rows = _table_lines(table_md)
    if not rows:
        return [table_md.strip()]

    chunks: List[str] = []
    current_rows: List[str] = []
    current_tokens = approx_tokens("\n".join(header))
    for row in rows:
        row_tokens = approx_tokens(row)
        if current_rows and current_tokens + row_tokens > max_tokens:
            chunks.append("\n".join(header + current_rows))
            current_rows = []
            current_tokens = approx_tokens("\n".join(header))
        current_rows.append(row)
        current_tokens += row_tokens
    if current_rows:
        chunks.append("\n".join(header + current_rows))
    return chunks


def _figure_text(fig: FigureData, page_text: str) -> str:
    parts: List[str] = []
    if fig.description:
        parts.append(fig.description)
    if fig.caption:
        parts.append(fig.caption)
    parts.extend(extract_caption_lines(page_text))
    text = " ".join(part.strip() for part in parts if part and part.strip())
    return text or "Figure image"


def _make_chunk(
    *,
    chunk_id: str,
    document_id: str,
    source: str,
    page: PageData,
    content_type: str,
    content: str,
    section: Optional[str] = None,
    image_path: Optional[str] = None,
    bbox: Optional[List[float]] = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        source=source,
        page=page.page,
        content_type=content_type,
        section=section,
        content=content.strip(),
        image_path=image_path,
        page_image_path=page.page_image_path,
        ocr_used=page.ocr_used if content_type == "ocr" else False,
        token_count=approx_tokens(content),
        bbox=bbox,
    )


def chunk_page(page: PageData, document_id: str, source: str) -> List[Chunk]:
    """Chunk a single PageData instance into discrete Chunks."""
    settings = get_settings(reload=True)
    chunks: List[Chunk] = []
    counter = 1

    text_type = "ocr" if page.ocr_used else "text"
    for content, section in _text_chunks(page, settings.CHUNK_TOKENS, settings.CHUNK_OVERLAP):
        chunks.append(
            _make_chunk(
                chunk_id=f"{document_id}_p{page.page}_c{counter}",
                document_id=document_id,
                source=source,
                page=page,
                content_type=text_type,
                content=content,
                section=section,
            )
        )
        counter += 1

    paragraphs = _paragraphs_with_sections(page.text)
    nearest_section = paragraphs[-1].section if paragraphs else None
    for table_idx, table_md in enumerate(page.tables_md, start=1):
        for split_idx, table_content in enumerate(_split_table(table_md), start=1):
            chunks.append(
                _make_chunk(
                    chunk_id=f"{document_id}_p{page.page}_t{table_idx}_{split_idx}",
                    document_id=document_id,
                    source=source,
                    page=page,
                    content_type="table",
                    content=table_content,
                    section=nearest_section,
                )
            )

    for fig_idx, fig in enumerate(page.figures, start=1):
        chunks.append(
            _make_chunk(
                chunk_id=f"{document_id}_p{page.page}_f{fig_idx}",
                document_id=document_id,
                source=source,
                page=page,
                content_type="image",
                content=_figure_text(fig, page.text),
                image_path=fig.image_path,
                bbox=fig.bbox,
            )
        )

    return chunks


def chunk_document(pages: List[PageData], document_id: str, source: str) -> List[Chunk]:
    """Chunk all pages of a document into atomic chunks ready for indexing."""
    all_chunks: List[Chunk] = []
    for page in pages:
        all_chunks.extend(chunk_page(page, document_id=document_id, source=source))
    return all_chunks
