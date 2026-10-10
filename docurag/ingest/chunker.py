import re
import uuid
from typing import List, Optional, Tuple
from docurag.schemas import Chunk, PageData, FigureDescription

# Configuration Constants
CHUNK_TOKENS = 512
CHUNK_OVERLAP = 50
TABLE_TOKEN_LIMIT = 1500

class IntelligentChunker:
    """
    Implements M2 Intelligent Chunking.
    Handles paragraph-aware text splitting, heading detection,
    table splitting with header preservation, and figure integration.
    """
    def __init__(self, store=None):
        # store is used to get/increment globally unique chunk IDs
        self.store = store
        self._local_counter = 0

    def _get_next_chunk_id(self) -> str:
        """
        Generates a globally unique chunk ID.
        Prefers store counter if available, otherwise falls back to UUID.
        """
        if self.store and hasattr(self.store, "get_next_id"):
            return self.store.get_next_id()

        self._local_counter += 1
        return f"chunk_{uuid.uuid4().hex[:8]}_{self._local_counter}"

    def _estimate_tokens(self, text: str) -> int:
        """Approximate tokens = len(text.split()) * 1.3 as per spec."""
        return int(len(text.split()) * 1.3)

    def _is_heading(self, line: str) -> bool:
        """
        Heuristic for heading detection:
        - Short line
        - Title Case or ALL CAPS
        - No terminal period
        """
        line = line.strip()
        if not line:
            return False
        if len(line) > 100:
            return False
        if line.endswith('.'):
            return False

        # Title Case or ALL CAPS
        is_all_caps = line.isupper()
        is_title_case = line.istitle()

        return is_all_caps or is_title_case

    def _chunk_text(self, text: str, page: int, page_image_path: str,
                    document_id: str, source: str, ocr_used: bool) -> List[Chunk]:
        chunks = []
        current_section = None

        # Split by double newline to preserve paragraphs
        paragraphs = text.split('\n\n')
        if len(paragraphs) == 1 and len(text) > CHUNK_TOKENS * 2:
            # If there are no double newlines but the text is long,
            # fall back to splitting by single newline or fixed size
            paragraphs = text.split('\n')
            if len(paragraphs) == 1:
                # Fallback for single massive line
                words = text.split()
                paragraphs = [" ".join(words[i:i + 100]) for i in range(0, len(words), 100)]

        buffer = []
        buffer_tokens = 0

        for p in paragraphs:
            p = p.strip()
            if not p:
                continue

            # Heading Detection
            if self._is_heading(p):
                # Flush current buffer before starting new section
                if buffer:
                    chunks.append(self._create_text_chunk(
                        buffer, page, page_image_path, document_id, source, ocr_used, current_section
                    ))
                    buffer = []
                    buffer_tokens = 0

                current_section = p
                # Headings themselves are usually small, we can either
                # put them in a chunk or let them be the 'section' for the next.
                # Requirement says "set the section field for following chunks".
                # We'll include the heading in the first chunk of the section too.
                buffer.append(p)
                buffer_tokens += self._estimate_tokens(p)
            else:
                p_tokens = self._estimate_tokens(p)

                if buffer_tokens + p_tokens > CHUNK_TOKENS and buffer:
                    # Flush buffer
                    chunks.append(self._create_text_chunk(
                        buffer, page, page_image_path, document_id, source, ocr_used, current_section
                    ))

                    # Overlap: keep last few paragraphs if possible
                    # Simplification: just keep the last paragraph as overlap
                    overlap_p = buffer[-1]
                    buffer = [overlap_p]
                    buffer_tokens = self._estimate_tokens(overlap_p)

                buffer.append(p)
                buffer_tokens += p_tokens

        if buffer:
            chunks.append(self._create_text_chunk(
                buffer, page, page_image_path, document_id, source, ocr_used, current_section
            ))

        return chunks

    def _create_text_chunk(self, paragraphs: List[str], page: int, page_image_path: str,
                           document_id: str, source: str, ocr_used: bool, section: Optional[str]) -> Chunk:
        content = "\n\n".join(paragraphs)
        return Chunk(
            chunk_id=self._get_next_chunk_id(),
            document_id=document_id,
            source=source,
            page=page,
            content_type="text" if not ocr_used else "ocr",
            section=section,
            content=content,
            page_image_path=page_image_path,
            ocr_used=ocr_used,
            token_count=self._estimate_tokens(content)
        )

    def _chunk_table(self, table_md: str, page: int, page_image_path: str,
                     document_id: str, source: str, section: Optional[str]) -> List[Chunk]:
        lines = table_md.split('\n')
        if not lines:
            return []

        header_row = lines[0]
        separator_row = lines[1] if len(lines) > 1 and '---' in lines[1] else ""
        data_rows = lines[2:] if separator_row else lines[1:]

        full_tokens = self._estimate_tokens(table_md)
        if full_tokens <= TABLE_TOKEN_LIMIT:
            return [self._create_table_chunk(table_md, page, page_image_path, document_id, source, section)]

        # Split by row groups
        chunks = []
        current_rows = []
        current_tokens = self._estimate_tokens(header_row + "\n" + separator_row)

        for row in data_rows:
            row_tokens = self._estimate_tokens(row)
            if current_tokens + row_tokens > TABLE_TOKEN_LIMIT and current_rows:
                # Create chunk with header
                table_content = header_row + "\n" + separator_row + "\n" + "\n".join(current_rows)
                chunks.append(self._create_table_chunk(table_content, page, page_image_path, document_id, source, section))

                # Reset
                current_rows = []
                current_tokens = self._estimate_tokens(header_row + "\n" + separator_row)

            current_rows.append(row)
            current_tokens += row_tokens

        if current_rows:
            table_content = header_row + "\n" + separator_row + "\n" + "\n".join(current_rows)
            chunks.append(self._create_table_chunk(table_content, page, page_image_path, document_id, source, section))

        return chunks

    def _create_table_chunk(self, content: str, page: int, page_image_path: str,
                            document_id: str, source: str, section: Optional[str]) -> Chunk:
        return Chunk(
            chunk_id=self._get_next_chunk_id(),
            document_id=document_id,
            source=source,
            page=page,
            content_type="table",
            section=section,
            content=content,
            page_image_path=page_image_path,
            token_count=self._estimate_tokens(content)
        )

    def _chunk_figure(self, figure_data: "FigureData", page: int, page_image_path: str,
                      document_id: str, source: str, section: Optional[str],
                      vision_desc: Optional[FigureDescription] = None) -> Chunk:
        # Vision text flattened + caption
        content = ""
        if vision_desc:
            content = vision_desc.to_flattened_text()

        if figure_data.caption:
            content += "\nCaption: " + figure_data.caption

        return Chunk(
            chunk_id=self._get_next_chunk_id(),
            document_id=document_id,
            source=source,
            page=page,
            content_type="image",
            section=section,
            content=content,
            image_path=figure_data.image_path,
            page_image_path=page_image_path,
            token_count=self._estimate_tokens(content)
        )

    def process_page(self, page_data: PageData, document_id: str, source: str) -> List[Chunk]:
        """
        Main entry point for chunking a page.
        """
        all_chunks = []

        # 1. Handle Text and OCR
        text_chunks = self._chunk_text(
            page_data.text,
            page_data.page,
            page_data.page_image_path,
            document_id,
            source,
            page_data.ocr_used
        )

        # We need to identify the sections from text_chunks to apply to tables/figures
        # For simplicity, we'll use the last detected section if available.
        current_section = None
        for tc in text_chunks:
            if tc.section:
                current_section = tc.section

        all_chunks.extend(text_chunks)

        # 2. Handle Tables
        for table_md in page_data.tables_md:
            table_chunks = self._chunk_table(
                table_md,
                page_data.page,
                page_data.page_image_path,
                document_id,
                source,
                current_section
            )
            all_chunks.extend(table_chunks)

        # 3. Handle Figures
        for fig in page_data.figures:
            # In a real scenario, we'd look up FigureDescription from a vision store
            # For now, we assume it's passed or handled.
            fig_chunk = self._chunk_figure(
                fig,
                page_data.page,
                page_data.page_image_path,
                document_id,
                source,
                current_section
            )
            all_chunks.append(fig_chunk)

        return all_chunks
