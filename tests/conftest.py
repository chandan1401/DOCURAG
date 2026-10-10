"""Hermetic fakes and reusable test fixtures for DocuRAG."""

import hashlib
import json
import math
import os
import re
import struct
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pytest

from docurag.config import Settings, get_settings
from docurag.schemas import (
    AskResult,
    Chunk,
    Confidence,
    ContentType,
    Evidence,
    Source,
    Timing,
    TraceStep,
)


# =====================================================================
# PURE PYTHON SYNTHETIC BYTE GENERATORS (Zero External Dependencies)
# =====================================================================

def generate_minimal_pdf_bytes(text: str = "DocuRAG Synthetic PDF Document") -> bytes:
    """Generate minimal valid PDF-1.4 bytes in pure Python without external dependencies.

    Constructs a structurally complete single-page PDF with /Catalog, /Pages,
    /Page, /Font, /Contents stream, and valid xref table.
    """
    escaped_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream_content = f"BT /F1 12 Tf 50 700 Td ({escaped_text}) Tj ET".encode("latin-1", errors="replace")
    stream_len = len(stream_content)

    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"

    obj1 = b"1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n"
    obj2 = b"2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n"
    obj3 = (
        b"3 0 obj\n"
        b"<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources <</Font <</F1 5 0 R>>>>>>\n"
        b"endobj\n"
    )
    obj4 = (
        f"4 0 obj\n<</Length {stream_len}>>\nstream\n".encode("ascii")
        + stream_content
        + b"\nendstream\nendobj\n"
    )
    obj5 = b"5 0 obj\n<</Type /Font /Subtype /Type1 /BaseFont /Helvetica>>\nendobj\n"

    offsets = [0]
    offsets.append(len(header))
    offsets.append(offsets[1] + len(obj1))
    offsets.append(offsets[2] + len(obj2))
    offsets.append(offsets[3] + len(obj3))
    offsets.append(offsets[4] + len(obj4))

    xref_offset = offsets[5] + len(obj5)

    xref = b"xref\n0 6\n0000000000 65535 f \n"
    for off in offsets[1:]:
        xref += f"{off:010d} 00000 n \n".encode("ascii")

    trailer = (
        f"trailer\n<</Size 6 /Root 1 0 R>>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )

    return header + obj1 + obj2 + obj3 + obj4 + obj5 + xref + trailer


def generate_minimal_png_bytes(
    width: int = 10,
    height: int = 10,
    rgb: Tuple[int, int, int] = (41, 128, 185),
) -> bytes:
    """Generate minimal valid RFC 2083 PNG bytes using only built-in struct and zlib.

    Creates signature, IHDR chunk (8-bit RGB), IDAT compressed scanlines, and IEND chunk.
    """
    sig = b"\x89PNG\r\n\x1a\n"

    # IHDR: width (4B), height (4B), bit_depth (1B=8), color_type (1B=2 for RGB),
    # compression (1B=0), filter (1B=0), interlace (1B=0)
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr_crc = struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF)
    ihdr_chunk = struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data + ihdr_crc

    r, g, b = rgb
    pixel = bytes([r, g, b])
    raw_scanlines = bytearray()
    for _ in range(height):
        raw_scanlines.append(0)  # Filter byte None
        raw_scanlines.extend(pixel * width)

    compressed_data = zlib.compress(bytes(raw_scanlines))
    idat_crc = struct.pack(">I", zlib.crc32(b"IDAT" + compressed_data) & 0xFFFFFFFF)
    idat_chunk = struct.pack(">I", len(compressed_data)) + b"IDAT" + compressed_data + idat_crc

    iend_crc = struct.pack(">I", zlib.crc32(b"IEND") & 0xFFFFFFFF)
    iend_chunk = struct.pack(">I", 0) + b"IEND" + iend_crc

    return sig + ihdr_chunk + idat_chunk + iend_chunk


# =====================================================================
# HERMETIC FAKES (DOCURAG_FAKES=1)
# =====================================================================

class FakeEmbedder:
    """Deterministic, CPU-only embedding generator using SHA-256 projections.

    Matches sentence-transformers API (dim=384, normalized L2).
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _hash_to_vector(self, text: str) -> List[float]:
        vec = []
        for i in range(self.dim):
            h = hashlib.sha256(f"{text}:{i}".encode("utf-8")).digest()
            val = int.from_bytes(h[:4], "big", signed=True) / (2**31)
            vec.append(val)
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def encode(
        self,
        sentences: Union[str, List[str]],
        normalize_embeddings: bool = True,
        **kwargs: Any,
    ) -> Any:
        import numpy as np

        if isinstance(sentences, str):
            v = self._hash_to_vector(sentences)
            return np.array(v, dtype=np.float32)

        # Invariant: Empty batch returns 2D array of shape (0, dim)
        if len(sentences) == 0:
            return np.empty((0, self.dim), dtype=np.float32)

        vectors = [self._hash_to_vector(s) for s in sentences]
        return np.array(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> List[float]:
        return self._hash_to_vector(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return [self._hash_to_vector(t) for t in texts]


class FakeReranker:
    """Deterministic cross-encoder reranker using token overlap Jaccard similarity."""

    def __init__(self, model_name: str = "fake-bge-reranker"):
        self.model_name = model_name

    def predict(self, sentences: List[Tuple[str, str]], **kwargs: Any) -> Any:
        import numpy as np

        scores = []
        for query, doc in sentences:
            q_words = set(re.findall(r"\w+", query.lower()))
            d_words = set(re.findall(r"\w+", doc.lower()))
            if not q_words or not d_words:
                scores.append(0.1)
                continue
            intersection = q_words.intersection(d_words)
            union = q_words.union(d_words)
            jaccard = len(intersection) / len(union) if union else 0.0
            score = 0.1 + (0.9 * jaccard)
            scores.append(score)
        return np.array(scores, dtype=np.float32)


class FakeLLM:
    """Deterministic script-based LLM fake generating structured responses."""

    def __init__(self, model_name: str = "fake-gemini"):
        self.model_name = model_name

    def generate_text(self, prompt: str) -> str:
        prompt_lower = prompt.lower()

        # Figure description indexing prompt (Base Spec §7.1)
        if "indexing a figure" in prompt_lower or "describe the image" in prompt_lower:
            return json.dumps({
                "type": "bar_chart",
                "title": "Model Performance Comparison",
                "caption": "Bar chart comparing DocuRAG against baseline RAG systems across retrieval accuracy.",
                "key_values": [
                    {"label": "DocuRAG", "value": "0.95", "unit": "Hit@5"},
                    {"label": "Baseline", "value": "0.78", "unit": "Hit@5"},
                ],
                "relationships": "",
                "visible_text": "DocuRAG: 0.95, Baseline: 0.78",
            })

        # Planner sub-question prompt (Multi-Agent Spec §B7.3)
        if "sub-questions" in prompt_lower or "break the question" in prompt_lower:
            return json.dumps({
                "sub_questions": [
                    {"sub_question": "What is the accuracy of Model A?", "document_names": ["doc_a.pdf"]},
                    {"sub_question": "What is the accuracy of Model B?", "document_names": ["doc_b.pdf"]},
                ]
            })

        # Router intent classification
        if "classify the user's question" in prompt_lower or "router" in prompt_lower:
            if any(k in prompt_lower for k in ["compare", "versus", "vs", "difference"]):
                return json.dumps({
                    "intent": "multi_doc_compare",
                    "query_type": "general",
                    "document_hints": [],
                    "needs_calculation": False,
                })
            if any(k in prompt_lower for k in ["table", "row", "column"]):
                return json.dumps({
                    "intent": "table_lookup",
                    "query_type": "table",
                    "document_hints": [],
                    "needs_calculation": False,
                })
            if any(k in prompt_lower for k in ["chart", "graph", "trend", "plot"]):
                return json.dumps({
                    "intent": "chart_numeric",
                    "query_type": "chart",
                    "document_hints": [],
                    "needs_calculation": True,
                })
            if any(k in prompt_lower for k in ["summarize", "overview"]):
                return json.dumps({
                    "intent": "summarize_doc",
                    "query_type": "general",
                    "document_hints": [],
                    "needs_calculation": False,
                })
            return json.dumps({
                "intent": "factual",
                "query_type": "general",
                "document_hints": [],
                "needs_calculation": False,
            })

        # Verification prompt
        if "verify" in prompt_lower or "claim" in prompt_lower:
            return json.dumps({
                "supported": True,
                "support_ratio": 1.0,
                "claims": [
                    {
                        "claim": "DocuRAG is a multimodal RAG system.",
                        "supported": True,
                        "citation": "E1",
                    }
                ],
            })

        # Chart/table value extraction prompt
        if "extract_chart_values" in prompt_lower or "values" in prompt_lower:
            return json.dumps({
                "values": [
                    {"label": "Precision", "value": 0.95, "unit": "ratio"},
                    {"label": "Recall", "value": 0.92, "unit": "ratio"},
                ]
            })

        # Synthesizer prompt
        if "combine the sub-answers" in prompt_lower or "synthesizer" in prompt_lower:
            return json.dumps({
                "answer": "DocuRAG integrates hybrid retrieval [E1] and reranking [E2].",
                "citations": ["E1", "E2"],
                "answerable": True,
            })

        # Standard grounded answer generation
        return "DocuRAG is a multimodal RAG system [E1] with hybrid retrieval [E2]."


# =====================================================================
# REUSABLE PYTEST FIXTURES
# =====================================================================

@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_reranker() -> FakeReranker:
    return FakeReranker()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def minimal_pdf_bytes() -> bytes:
    """Fixture returning raw minimal valid PDF-1.4 bytes."""
    return generate_minimal_pdf_bytes()


@pytest.fixture
def minimal_png_bytes() -> bytes:
    """Fixture returning raw minimal valid PNG bytes."""
    return generate_minimal_png_bytes()


@pytest.fixture
def synthetic_text_pdf(tmp_path: Path) -> Path:
    """Fixture generating a valid text PDF with paragraphs for M1 text extraction."""
    file_path = tmp_path / "synthetic_text.pdf"
    try:
        import fitz  # PyMuPDF
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        text = (
            "DocuRAG Architecture Overview\n\n"
            "DocuRAG is a production-grade multimodal Retrieval-Augmented Generation system. "
            "It incorporates hybrid search fusing dense BGE embeddings with sparse BM25 indices. "
            "The system ensures strict claim verification and high attribution accuracy."
        )
        page.insert_text((50, 72), text, fontsize=12)
        doc.save(str(file_path))
        doc.close()
    except Exception:
        file_path.write_bytes(generate_minimal_pdf_bytes("DocuRAG Architecture Overview"))
    return file_path


@pytest.fixture
def synthetic_table_pdf(tmp_path: Path) -> Path:
    """Fixture generating a PDF containing a markdown-formatted table for M1/M2 table extraction."""
    file_path = tmp_path / "synthetic_table.pdf"
    table_content = (
        "Benchmark Performance Evaluation\n\n"
        "| Architecture | Hit@5 | Latency | Faithfulness |\n"
        "|---|---|---|---|\n"
        "| DocuRAG Linear | 0.91 | 380ms | 0.94 |\n"
        "| DocuRAG Multi-Agent | 0.95 | 520ms | 0.98 |\n"
        "| Standard Dense RAG | 0.78 | 410ms | 0.81 |\n"
    )
    try:
        import fitz
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        page.insert_text((50, 72), table_content, fontsize=11)
        doc.save(str(file_path))
        doc.close()
    except Exception:
        file_path.write_bytes(generate_minimal_pdf_bytes(table_content))
    return file_path


@pytest.fixture
def synthetic_chart_pdf(tmp_path: Path) -> Path:
    """Fixture generating a PDF with an embedded chart image for M1 figure extraction."""
    file_path = tmp_path / "synthetic_chart.pdf"
    chart_png = generate_minimal_png_bytes(width=200, height=150, rgb=(52, 152, 219))
    try:
        import fitz
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        page.insert_text((50, 50), "Figure 1: Retrieval Latency Distribution", fontsize=12)
        rect = fitz.Rect(50, 80, 250, 230)
        page.insert_image(rect, stream=chart_png)
        doc.save(str(file_path))
        doc.close()
    except Exception:
        file_path.write_bytes(generate_minimal_pdf_bytes("Figure 1: Embedded Chart Image"))
    return file_path


@pytest.fixture
def synthetic_scanned_pdf(tmp_path: Path) -> Path:
    """Fixture generating an image-only PDF with <50 chars text to trigger OCR fallback."""
    file_path = tmp_path / "synthetic_scanned.pdf"
    page_img = generate_minimal_png_bytes(width=612, height=792, rgb=(245, 245, 240))
    try:
        import fitz
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        page.insert_image(page.rect, stream=page_img)
        doc.save(str(file_path))
        doc.close()
    except Exception:
        file_path.write_bytes(generate_minimal_pdf_bytes(""))
    return file_path


@pytest.fixture
def synthetic_png_file(tmp_path: Path) -> Path:
    """Fixture generating a standalone PNG image file for single-page image ingestion."""
    file_path = tmp_path / "synthetic_diagram.png"
    file_path.write_bytes(generate_minimal_png_bytes(width=300, height=200, rgb=(46, 204, 113)))
    return file_path


@pytest.fixture
def synthetic_corpus(
    synthetic_text_pdf: Path,
    synthetic_table_pdf: Path,
    synthetic_chart_pdf: Path,
    synthetic_scanned_pdf: Path,
    synthetic_png_file: Path,
) -> Dict[str, Path]:
    """Corpus dictionary bundling all 5 synthetic test files."""
    return {
        "text": synthetic_text_pdf,
        "table": synthetic_table_pdf,
        "chart": synthetic_chart_pdf,
        "scanned": synthetic_scanned_pdf,
        "image": synthetic_png_file,
    }


@pytest.fixture
def sample_chunks() -> List[Chunk]:
    """Factory creating 5 diverse synthetic chunks (text, table, figure, scanned)."""
    return [
        Chunk(
            chunk_id="c_001",
            document_id="doc_arch",
            source="architecture.pdf",
            page=1,
            content_type="text",
            section="Overview",
            content="DocuRAG is a production-grade multimodal RAG architecture.",
            token_count=12,
        ),
        Chunk(
            chunk_id="c_002",
            document_id="doc_arch",
            source="architecture.pdf",
            page=2,
            content_type="text",
            section="Retrieval",
            content="Hybrid retrieval fuses dense BGE embeddings with sparse BM25 scores via RRF.",
            token_count=14,
        ),
        Chunk(
            chunk_id="c_003",
            document_id="doc_metrics",
            source="evaluation.pdf",
            page=3,
            content_type="table",
            section="Benchmarks",
            content="| Model | Hit@5 | Latency |\n| DocuRAG | 0.94 | 420ms |",
            token_count=16,
        ),
        Chunk(
            chunk_id="c_004",
            document_id="doc_diagram",
            source="diagrams.pdf",
            page=4,
            content_type="image",
            section="Visual",
            content="Architecture diagram illustrating router, planner, and synthesizer agents.",
            image_path="storage/figs/fig_4_1.png",
            token_count=13,
        ),
        Chunk(
            chunk_id="c_005",
            document_id="doc_scanned",
            source="scanned_notes.pdf",
            page=5,
            content_type="ocr",
            section="Handwritten",
            content="Verification checks atomic claims against cited evidence blocks.",
            ocr_used=True,
            token_count=11,
        ),
    ]


@pytest.fixture
def sample_evidence(sample_chunks: List[Chunk]) -> List[Evidence]:
    """Factory creating evidence items mapped from sample chunks."""
    return [
        Evidence(
            eid=f"E{idx+1}",
            chunk=c,
            score=0.95 - (idx * 0.1),
            norm_score=0.9 - (idx * 0.1),
            dense_score=0.88,
            sparse_score=12.4,
            rerank_score=0.95 - (idx * 0.1),
            combined_score=0.92,
        )
        for idx, c in enumerate(sample_chunks[:2])
    ]


@pytest.fixture
def sample_ask_result(sample_evidence: List[Evidence]) -> AskResult:
    """Factory creating a valid AskResult object."""
    sources = [
        Source(
            id=str(idx + 1),
            document=ev.chunk.source,
            page=ev.chunk.page,
            type=ev.chunk.content_type,
            snippet=ev.chunk.content[:60],
            score=ev.score,
            confidence=0.95,
        )
        for idx, ev in enumerate(sample_evidence)
    ]
    return AskResult(
        question="What is DocuRAG?",
        answer="DocuRAG is a production-grade multimodal RAG architecture [E1].",
        answerable=True,
        sources=sources,
        confidence=Confidence(
            score=0.95,
            label="High",
            support_ratio=1.0,
            citation_coverage=1.0,
            reason="High grounding in cited evidence",
        ),
        evidence=sample_evidence,
        timing=Timing(
            retrieval_ms=85,
            rerank_ms=120,
            generation_ms=250,
            verification_ms=90,
            total_ms=545,
            ingest_ms=0,
        ),
        verified=True,
        support_ratio=1.0,
        agent_trace=[
            TraceStep(
                agent="router",
                action="classify",
                tool=None,
                detail="Classified as factual intent",
                ms=12,
            )
        ],
    )
