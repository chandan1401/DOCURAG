# DocuRAG Interface & API Contracts (Frozen)

This document specifies the frozen data models, REST APIs, MCP tool signatures, and internal engine interfaces for **DocuRAG**. Any modification requires an entry in `CONTRACT_CHANGES.md` and Lead review.

---

## 1. Core Data Schemas (`docurag.schemas`)

All schemas inherit from Pydantic v2 `BaseModel` with `ConfigDict(populate_by_name=True, extra="ignore")` and `@model_validator(mode="before")` alias normalization.

### `Chunk`
```json
{
  "chunk_id": "c_000101",
  "document_id": "doc_001",
  "source": "financial_report.pdf",
  "page": 3,
  "content_type": "text",
  "section": "Executive Summary",
  "content": "Net revenue increased by 14% year-over-year.",
  "image_path": null,
  "page_image_path": "storage/pages/doc_001_p3.png",
  "ocr_used": false,
  "token_count": 8,
  "bbox": [54.0, 120.0, 558.0, 160.0]
}
```
*Aliases / Compatibility*: `chunk.doc_id` -> `document_id`, `chunk.doc_name` -> `source`, `chunk.text` -> `content`, `chunk.is_table` -> boolean, `chunk.is_figure` -> boolean.

### `Evidence`
```json
{
  "eid": "E1",
  "chunk": { "...": "..." },
  "score": 0.942,
  "norm_score": 0.915,
  "dense_score": 0.884,
  "sparse_score": 14.2,
  "rerank_score": 0.942,
  "combined_score": 0.925
}
```

### `Source`
```json
{
  "id": "1",
  "document": "financial_report.pdf",
  "page": 3,
  "type": "text",
  "snippet": "Net revenue increased by 14% year-over-year.",
  "thumbnail_url": "/api/pages/doc_001_p3.png",
  "score": 0.942,
  "confidence": 0.95,
  "bbox": [54.0, 120.0, 558.0, 160.0]
}
```
*Aliases / Compatibility*: `citation_id` -> `id` (string), `doc_name` -> `document`, `quote` -> `snippet`.

### `Confidence`
```json
{
  "score": 0.94,
  "label": "High",
  "support_ratio": 1.0,
  "citation_coverage": 1.0,
  "reason": "All claims directly supported by cited evidence"
}
```
*Label thresholds*: `score >= 0.75` -> "High"; `0.4 <= score < 0.75` -> "Medium"; `score < 0.4` -> "Low". When unanswerable, score capped at `0.2`.

### `Timing`
```json
{
  "retrieval_ms": 112,
  "rerank_ms": 68,
  "generation_ms": 340,
  "verification_ms": 85,
  "total_ms": 605,
  "ingest_ms": 0
}
```
*Aliases / Compatibility*: `retrieve_ms` -> `retrieval_ms`, `generate_ms` -> `generation_ms`, `verify_ms` -> `verification_ms`.

### `TraceStep` (v2)
```json
{
  "agent": "router",
  "action": "classify",
  "tool": null,
  "detail": "Classified question as multi_doc_compare",
  "ms": 18
}
```

### `AskResult`
Returned by `POST /api/query/` and `docurag.pipeline.ask()`:
```json
{
  "question": "How did Q3 revenue compare with Q2?",
  "answer": "Q3 revenue grew by 14% compared to Q2 [E1].",
  "answerable": true,
  "sources": [ { "...": "Source object" } ],
  "confidence": { "...": "Confidence object" },
  "evidence": [ { "...": "Evidence object" } ],
  "timing": { "...": "Timing object" },
  "verified": true,
  "support_ratio": 1.0,
  "agent_trace": [ { "...": "TraceStep object" } ]
}
```

### `FigureData`
```json
{
  "figure_id": "fig_001",
  "image_path": "storage/figs/fig_001.png",
  "page": 2,
  "bbox": [50.0, 100.0, 450.0, 350.0],
  "caption": "Quarterly revenue growth bar chart",
  "description": null
}
```
*Aliases*: `path` <-> `image_path`.

### `FigureDescription`
Structured output from multimodal Gemini Vision analysis (Prompt §7.1):
```json
{
  "type": "bar_chart",
  "title": "Revenue Growth 2025",
  "caption": "Quarterly comparison of revenue and margins.",
  "key_values": [{"label": "Q3", "value": "14%", "unit": "growth"}],
  "relationships": "",
  "visible_text": "Q3 Growth: 14%"
}
```

### `PageData`
Intermediate extraction representation per document page:
```json
{
  "page": 1,
  "text": "Extracted page text...",
  "ocr_used": false,
  "tables_md": ["| A | B |\n| 1 | 2 |"],
  "figures": [ { "...": "FigureData" } ],
  "page_image_path": "storage/pages/doc_001_p1.png"
}
```

### `RouteDecision`
Router Agent intent classification output:
```json
{
  "intent": "factual",
  "query_type": "general",
  "document_hints": [],
  "needs_calculation": false
}
```
*Valid Intents*: `factual`, `table_lookup`, `chart_numeric`, `multi_doc_compare`, `summarize_doc`, `out_of_scope`. Defaults to `factual`.

### `SubQuestion`
Planner Agent query decomposition unit for multi-document comparisons:
```json
{
  "sub_question": "What is the net margin in Document A?",
  "document_names": ["doc_a.pdf"],
  "document_ids": ["doc_a.pdf"]
}
```

### `GeneratedAnswer`
```json
{
  "answer": "Answer text with [E1] citations.",
  "citations": ["E1"],
  "answerable": true
}
```

### `VerifierClaim` & `VerifierResult`
```json
{
  "claims": [
    {
      "claim": "Net revenue increased by 14%.",
      "supported": true,
      "citation": "E1",
      "reason": "Direct text match"
    }
  ],
  "support_ratio": 1.0,
  "supported": true
}
```

### `Filters`
Retrieval metadata filters:
```json
{
  "document_ids": ["doc_001", "doc_002"],
  "content_types": ["table", "text"]
}
```

### `ChunkHit`
Hit dictionary returned by MCP retrieval tools:
```json
{
  "chunk_id": "c_001",
  "document": "annual_report.pdf",
  "page": 3,
  "content_type": "text",
  "snippet": "Net revenue increased by 14%...",
  "score": 0.94
}
```

---

## 2. REST API Contract (`backend/api/`)

### `POST /api/query/`
- **Request Body**:
```json
{
  "question": "What is the capital expenditure for 2025?",
  "top_k": 5,
  "mode": "auto"
}
```
*Note*: `mode` can be `"auto"` (follows `USE_AGENTS`), `"linear"`, or `"agents"`.
- **Response**: `200 OK` with JSON conforming to `AskResult`.

### `GET /api/documents/`
- **Response**: `200 OK`
```json
[
  {
    "document_id": "doc_001",
    "filename": "annual_report.pdf",
    "pages": 24,
    "chunks_count": 142,
    "status": "ready",
    "created_at": "2026-10-08T18:00:00Z"
  }
]
```

### `POST /api/documents/upload/`
- **Multipart Form**: `file` (PDF, DOCX, TXT, PNG, JPG; max 50MB)
- **Response**: `202 Accepted`
```json
{
  "document_id": "doc_002",
  "filename": "quarterly_update.pdf",
  "status": "queued"
}
```

### `GET /api/documents/<id>/progress/`
- **Response**: `200 OK`
```json
{
  "document_id": "doc_002",
  "pages_done": 12,
  "pages_total": 24,
  "status": "processing"
}
```

### `DELETE /api/documents/<id>/`
- **Response**: `204 No Content`

### Standard Error JSON Format
```json
{
  "error": {
    "code": "INVALID_QUERY",
    "message": "Question cannot be empty or whitespace.",
    "status": 400
  }
}
```
Error codes:
- `INVALID_QUERY` (400)
- `INVALID_FILE_TYPE` (400)
- `FILE_TOO_LARGE` (400)
- `DOCUMENT_NOT_FOUND` (404)
- `INGESTION_FAILED` (500)
- `QUERY_EXECUTION_ERROR` (500)

---

## 3. FastMCP Tool Signatures (`docurag.agents.mcp_servers`)

Implemented as FastMCP servers callable via `stdio` or `direct` (in-process) Python calls.

### Document Server (`document_server`)
- `list_documents() -> list[dict]`
  Returns: `[{"document_id": str, "filename": str, "pages": int, "status": str}]`
- `get_page_text(document_id: str, page: int) -> dict`
  Returns: `{"text": str, "content_types": list[str]}`
- `get_page_image(document_id: str, page: int) -> dict`
  Returns: `{"path": str}`

### Retrieval Server (`retrieval_server`)
- `hybrid_search(query: str, k: int = 6, document_ids: list[str] | None = None, content_types: list[str] | None = None) -> list[dict]`
  Returns list of `ChunkHit`: `[{"chunk_id": str, "document": str, "page": int, "content_type": str, "snippet": str, "score": float}]`
- `search_tables(query: str, k: int = 4, document_ids: list[str] | None = None) -> list[dict]`
- `search_images(query: str, k: int = 4, document_ids: list[str] | None = None) -> list[dict]`

### Analysis Server (`analysis_server`)
- `calculate(expression: str) -> dict`
  Safe AST-based arithmetic evaluator (numbers, `+ - * / ** ( )`, `abs`, `round`, `min`, `max`).
  Returns: `{"result": float | None, "error": str | None}`
- `extract_chart_values(chunk_id: str, question: str) -> dict`
  Returns: `{"values": [{"label": str, "value": float, "unit": str}]}`

---

## 4. Pipeline Engine Public API (`docurag.pipeline`)

```python
def ask(question: str, top_k: int = 5, use_agents: bool | None = None) -> AskResult: ...
def _ask_linear(question: str, top_k: int = 5) -> AskResult: ...
def ingest_file(file_path: str, collection_id: str = "default") -> dict: ...
def ingest_directory(dir_path: str, collection_id: str = "default") -> dict: ...
```

---

## 5. Core Engine Subpackages Public Interface Stubs

### Ingestion (`docurag.ingest`)
- `load_pdf(path: Path | str, document_id: str) -> list[PageData]`
- `load_image(path: Path | str, document_id: str) -> list[PageData]`
- `extract_pages(path: Path | str, document_id: str) -> list[PageData]`
- `iter_pages(path: Path | str, document_id: str = "") -> Iterator[PageData]`
- `ocr_page(image: Any) -> tuple[str, float]`
- `extract_figures(path: Path | str, page_number: int | None = None) -> list[FigureData]`
- `describe_figure(image_path: Path | str) -> FigureDescription`
- `transcribe_page(image_path: Path | str) -> str`
- `classify_page(page: Any) -> "text" | "scanned" | "chart_like" | "mixed"`

### Chunking (`docurag.chunk`)
- `chunk_page(page: PageData, document_id: str, source: str) -> list[Chunk]`
- `chunk_document(pages: list[PageData], document_id: str, source: str) -> list[Chunk]`

### Indexing (`docurag.index`)
- `Embedder(model_name: str = ..., device: str | None = ...)`
  - `encode(sentences: list[str] | str) -> np.ndarray` (preserves 2D shape `(0, 384)` on empty)
  - `embed_query(q: str) -> np.ndarray`
  - `embed_chunks(chunks: list[Chunk]) -> np.ndarray`
- `ChunkStore(storage_dir: str | Path | None = ...)`
  - `add_document(doc_id: str, chunks: list[Chunk], embeddings: np.ndarray | None = ...) -> None`
  - `remove_document(doc_id: str) -> None`
  - `get(chunk_id: str) -> Chunk`
  - `all_chunks() -> list[Chunk]`
  - `save(path: str | Path | None = ...) -> None`
  - `load(root: str | Path) -> ChunkStore`
  - `stats() -> dict[str, Any]`
- `FAISSIndex(dim: int = 384)`
  - `build(chunks: list[Chunk], embeddings: np.ndarray) -> None`
  - `search(query_embedding: np.ndarray, top_k: int = 30) -> list[tuple[Chunk, float]]`
- `BM25Index()`
  - `build(chunks: list[Chunk]) -> None`
  - `search(query: str, top_k: int = 30) -> list[tuple[Chunk, float]]`

### Retrieval (`docurag.retrieve`)
- `hybrid_search(query: str, store: Any = None, embedder: Any = None, *, filters: Filters | None = None, dense_k: int = 30, bm25_k: int = 30, rrf_k: int = 60) -> list[tuple[Chunk, float]]`
- `retrieve(query: str, store: Any = None, embedder: Any = None, reranker: Any = None, *, final_k: int = 6, filters: Filters | None = None, query_type: str = "general", expand_neighbor_pages: bool = False) -> list[Evidence]`
- `retrieve_for_documents(query: str, document_ids: list[str], store: Any = None, embedder: Any = None, reranker: Any = None, *, final_k: int = 4) -> list[Evidence]`
- `rerank(query: str, candidates: list[tuple[Chunk, float]] | list[Chunk], reranker: Any = None, top_n: int = 6) -> list[tuple[Chunk, float]]`

### Generation & Verification (`docurag.generate`)
- `generate_answer(question: str, evidence: list[Evidence], instructions: str | None = None) -> GeneratedAnswer`
- `synthesize_comparison(question: str, sub_answers: list[dict[str, Any]] | str) -> GeneratedAnswer`
- `verify_answer(answer: GeneratedAnswer | str, evidence: list[Evidence]) -> VerifierResult`
- `compute_confidence(evidence: list[Evidence], support_ratio: float, answer: GeneratedAnswer | str) -> Confidence`
- Prompts: `FIGURE_DESCRIPTION_PROMPT`, `GROUNDED_ANSWER_PROMPT`, `VERIFIER_PROMPT`
