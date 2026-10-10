"""Table extraction, Markdown formatting, and table-text masking utilities."""

import logging
import re
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


def table_to_markdown(raw_table: List[List[Optional[str]]]) -> Optional[str]:
    """Convert a 2D grid of cell values into a standardized GitHub-flavored Markdown table.

    Args:
        raw_table: 2D list of row strings extracted from a table.

    Returns:
        Clean formatted Markdown table string, or None if table is invalid or empty.
    """
    if not raw_table or len(raw_table) < 2:
        return None

    # Calculate max column count across rows
    max_cols = max(len(row) for row in raw_table if row is not None)
    if max_cols < 2:
        return None

    clean_rows: List[List[str]] = []
    has_meaningful_content = False

    for row in raw_table:
        if row is None:
            continue
        padded_row = list(row) + [""] * (max_cols - len(row))
        clean_row: List[str] = []
        for cell in padded_row:
            cell_str = "" if cell is None else str(cell).strip()
            # Replace internal newlines with space or <br>
            cell_str = re.sub(r"[\r\n]+", " ", cell_str)
            # Escape pipes
            cell_str = cell_str.replace("|", r"\|")
            if cell_str:
                has_meaningful_content = True
            clean_row.append(cell_str)
        clean_rows.append(clean_row)

    if not has_meaningful_content or len(clean_rows) < 2:
        return None

    # Process header
    header = clean_rows[0]
    if not any(header):
        header = [f"Col_{i+1}" for i in range(max_cols)]

    separator = ["---"] * max_cols

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in clean_rows[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def is_inside_table(
    block_bbox: Tuple[float, float, float, float],
    table_bboxes: List[List[float]],
    overlap_threshold: float = 0.4,
) -> bool:
    """Check if a text block bounding box overlaps significantly with any table bounding box.

    Args:
        block_bbox: Tuple of (x0, y0, x1, y1) for text block.
        table_bboxes: List of [x0, y0, x1, y1] for detected tables.
        overlap_threshold: Minimum intersection area ratio over block area to consider masked.

    Returns:
        True if the block is located inside a table, False otherwise.
    """
    bx0, by0, bx1, by1 = block_bbox
    b_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    if b_area <= 0.0:
        return False

    for t_bbox in table_bboxes:
        tx0, ty0, tx1, ty1 = t_bbox
        ix0 = max(bx0, tx0)
        iy0 = max(by0, ty0)
        ix1 = min(bx1, tx1)
        iy1 = min(by1, ty1)

        if ix1 > ix0 and iy1 > iy0:
            inter_area = (ix1 - ix0) * (iy1 - iy0)
            if (inter_area / b_area) >= overlap_threshold:
                return True

    return False


def mask_table_text(
    blocks: List[Any],
    table_bboxes: List[List[float]],
    overlap_threshold: float = 0.4,
) -> List[Any]:
    """Filter out text blocks whose bounding boxes intersect with detected tables.

    Args:
        blocks: List of PyMuPDF text block tuples (x0, y0, x1, y1, text, block_no, block_type).
        table_bboxes: List of table bounding boxes [x0, y0, x1, y1].
        overlap_threshold: Area overlap threshold.

    Returns:
        List of blocks that do not intersect with any table bounding box.
    """
    if not table_bboxes:
        return blocks

    filtered_blocks = []
    for b in blocks:
        if len(b) >= 4:
            b_bbox = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            if is_inside_table(b_bbox, table_bboxes, overlap_threshold):
                continue
        filtered_blocks.append(b)

    return filtered_blocks


def extract_tables_from_page(
    page: Any,
    pdfplumber_page: Any = None,
) -> Tuple[List[str], List[List[float]]]:
    """Extract tables from a PyMuPDF page, with optional pdfplumber fallback.

    Args:
        page: PyMuPDF Page object.
        pdfplumber_page: Optional pdfplumber Page object for fallback extraction.

    Returns:
        Tuple of (list_of_markdown_tables, list_of_table_bounding_boxes).
    """
    tables_md: List[str] = []
    table_bboxes: List[List[float]] = []

    # 1. Primary engine: PyMuPDF page.find_tables()
    try:
        if hasattr(page, "find_tables"):
            tabs = page.find_tables()
            for tab in tabs:
                raw = tab.extract()
                md = table_to_markdown(raw)
                if md:
                    tables_md.append(md)
                    bbox = [float(c) for c in tab.bbox]
                    table_bboxes.append(bbox)
    except Exception as exc:
        logger.warning("PyMuPDF find_tables encountered error: %s", exc)

    # 2. Secondary engine: pdfplumber fallback if 0 tables found but page looks tabular
    if not tables_md and pdfplumber_page is not None:
        try:
            plumber_tables = pdfplumber_page.extract_tables()
            if plumber_tables:
                for raw_table in plumber_tables:
                    md = table_to_markdown(raw_table)
                    if md:
                        tables_md.append(md)
                        # Estimate table bbox from plumber page or fallback
                        table_bboxes.append([0.0, 0.0, float(page.rect.width), float(page.rect.height)])
        except Exception as exc:
            logger.warning("pdfplumber table extraction encountered error: %s", exc)

    return tables_md, table_bboxes
