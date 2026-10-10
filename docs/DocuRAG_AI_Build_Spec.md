# DocuRAG — AI Build Spec

> **Audience:** an AI coding agent (Claude Code, Cursor, etc.) implementing this repo end to end.
> **Human owner:** Adi. Hackathon evaluation is **13 Oct 2026**; build window ends 12 Oct.
> Read this whole file before writing code. Work milestone by milestone. Do not skip ahead.

---

## 0. Operating Rules for the Agent

1. **Follow milestones in order (M0 → M9).** Each has acceptance criteria and a verify command. A milestone is done only when its verify command passes.
2. **Commit after every milestone** with message `M<n>: <title>`.
3. **Core engine first.** `docurag/` must never import Django. Django (`backend/`) only imports `docurag.pipeline`.
4. **Do not add features, libraries, or abstractions not listed here.** If something is ambiguous, pick the simplest option, note it in `DECISIONS.md`, and continue.
5. **Never hardcode secrets.** Read `GEMINI_API_KEY` etc. from environment variables. Provide `.env.example`. Never commit `.env`.
6. **Every external call (LLM, vision, OCR) must be wrapped** with retry (3 tries, exponential backoff) and a disk cache keyed by content hash, so re-runs are free and rate limits don't break ingestion.
7. **Fail soft in ingestion:** one bad page must not fail the document. Log it, mark the page, continue.
8. **Must run in Google Colab (T4 GPU) and locally on CPU.** Detect CUDA with `torch.cuda.is_available()`; never assume a GPU.
9. **Write tests as you go** (pytest). Use small fixture files in `tests/fixtures/`. Mock LLM calls in unit tests; real-API tests are marked `@pytest.mark.live` and skipped by default.
10. **Keep a running `PROGRESS.md`:** milestone, status, anything the human must do (e.g., supply API key, add documents).
11. **Tiering:** M0–M7 are **MUST**. M8–M9 are **STRETCH** and start only after M7's verify passes. If time is short, skip stretch entirely.

---

## 1. What We Are Building

**DocuRAG** — a multimodal RAG system that ingests PDFs and images, retrieves relevant text, tables and visual context, and generates grounded answers with **sources, confidence, and response-time information**.

Must handle:
- Paraphrased queries (semantic match, not keyword only)
- Scanned PDFs (OCR)
- Tables (searchable as markdown)
- Charts, diagrams, embedded images (vision-derived searchable text + original image sent at answer time)
- Multi-document answers
- Refusal when evidence is insufficient

**Non-goals:** user accounts/auth, multi-tenant isolation, streaming responses, fine-tuning, fancy frontend framework, production deployment.

---

## 2. Environment & Stack (fixed — do not substitute)

| Concern | Choice |
|---|---|
| Python | 3.10+ |
| PDF | `pymupdf` (fitz), `pdfplumber` for table fallback |
| OCR | `easyocr` (GPU if available); Gemini vision fallback for low-confidence pages |
| Vision + answer LLM | Google Gemini via `google-genai` (model name from `GEMINI_MODEL`, default a current Flash model) |
| Text embeddings | `sentence-transformers` → `BAAI/bge-small-en-v1.5` (config-switchable to `bge-m3`) |
| Vector index | `faiss-cpu`, `IndexFlatIP` on L2-normalised vectors |
| Keyword | `rank_bm25` |
| Reranker | `sentence_transformers.CrossEncoder("BAAI/bge-reranker-base")` |
| Backend | Django 5 + Django REST Framework |
| Frontend | Django templates + Tailwind via CDN + vanilla JS `fetch` (no build step) |
| Agents (M8) | `langgraph`, `mcp` (FastMCP) |
| Tests | `pytest` |
| Colab tunnel | `cloudflared` (or ngrok via `NGROK_AUTHTOKEN`) |

`requirements.txt` must pin major versions. Split into `requirements.txt` (core + Django) and `requirements-agents.txt` (M8).

---

## 3. Repository Layout (create exactly this)

```
docurag/                         # core engine — NO django imports
  __init__.py
  config.py
  schemas.py
  cache.py                       # content-hash disk cache + retry decorator
  timing.py
  ingest/
    __init__.py
    pdf_loader.py
    ocr.py
    vision.py
    chunker.py
  index/
    __init__.py
    embedder.py
    vector_store.py
    bm25_store.py
    store.py                     # ChunkStore: persistence of chunks + both indexes
  retrieve/
    __init__.py
    hybrid.py
    rerank.py
  generate/
    __init__.py
    prompts.py
    answer.py
    verify.py
    confidence.py
  agents/                        # M8 only
    __init__.py
    graph.py
    mcp_servers/
      document_server.py
      retrieval_server.py
  pipeline.py                    # public API: ingest_file(), ask()
backend/
  manage.py
  config/{settings.py,urls.py,wsgi.py}
  api/{models.py,serializers.py,views.py,urls.py,tasks.py,apps.py}
  templates/{base.html,library.html,ask.html,history.html}
  static/{app.js}
notebooks/
  01_ingest_and_index.ipynb
  02_retrieval_and_answer.ipynb
  03_eval.ipynb
  04_colab_serve.ipynb           # runs Django + tunnel in Colab
data/{raw/,eval/gold_qa.json,SOURCES.md}
storage/{uploads/,pages/,figs/,index/,cache/}   # gitignored
tests/{fixtures/,test_*.py}
.env.example  .gitignore  requirements.txt  requirements-agents.txt
README.md  PROGRESS.md  DECISIONS.md
```

---

## 4. Configuration (`docurag/config.py`)

Single `Settings` dataclass loaded from env with defaults:

| Env var | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | — (required for live) | Gemini key |
| `GEMINI_MODEL` | `gemini-flash-latest`-style current Flash model | Vision + answer model |
| `DOCURAG_STORAGE` | `./storage` | Root storage dir (Colab: Drive path) |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Text embedder |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder |
| `OCR_MIN_CHARS` | `50` | Below this per-page char count → treat as scanned |
| `RENDER_DPI` | `200` | Page render DPI for OCR/thumbnails |
| `MIN_IMG_PX` | `150` | Skip embedded images smaller than this (either side) |
| `CHUNK_TOKENS` | `400` | Target chunk size |
| `CHUNK_OVERLAP` | `50` | Overlap tokens |
| `DENSE_K` / `BM25_K` | `30` / `30` | Candidates per retriever |
| `RERANK_IN` / `FINAL_K` | `20` / `6` | Rerank input size / final evidence count |
| `RRF_K` | `60` | RRF constant |
| `SUPPORT_THRESHOLD` | `0.7` | Verifier retry threshold |
| `USE_AGENTS` | `false` | Use LangGraph path (M8) |

---

## 5. Data Contracts (`docurag/schemas.py`)

Use `pydantic` v2 models (or dataclasses with `.to_dict()`).

```python
ContentType = Literal["text", "table", "image", "ocr"]

class Chunk(BaseModel):
    chunk_id: str                 # "c_000231"
    document_id: str              # "d_004"
    source: str                   # original filename
    page: int                     # 1-indexed
    content_type: ContentType
    section: str | None = None
    content: str                  # text that is embedded + shown as snippet
    image_path: str | None = None       # figure crop, if content_type == "image"
    page_image_path: str | None = None  # full page render for the viewer
    ocr_used: bool = False

class Evidence(BaseModel):
    eid: str                      # "E1".."En" — assigned per query
    chunk: Chunk
    score: float                  # reranker score (raw)
    norm_score: float             # 0..1

class Source(BaseModel):
    id: str; document: str; page: int; type: ContentType
    snippet: str; thumbnail_url: str | None; score: float

class Confidence(BaseModel):
    score: float                  # 0..1
    label: Literal["High", "Medium", "Low"]
    support_ratio: float
    citation_coverage: float

class Timing(BaseModel):
    retrieval_ms: int; rerank_ms: int; generation_ms: int
    verification_ms: int; total_ms: int

class AskResult(BaseModel):
    answer: str
    answerable: bool
    sources: list[Source]
    confidence: Confidence
    timing: Timing
    agent_trace: list[str] = []
```

`AskResult.model_dump()` is exactly the JSON returned by `POST /api/query/`.

---

## 6. Milestones

### M0 — Scaffold
**Tasks:** create layout (section 3), `requirements.txt`, `.gitignore` (storage/, .env, __pycache__), `.env.example`, `config.py`, `schemas.py`, `cache.py`, `timing.py`, empty `PROGRESS.md`/`DECISIONS.md`, `README.md` stub.

- `cache.py`: `disk_cache(namespace)` decorator keyed on SHA-256 of args (bytes args hashed by content); stores JSON under `storage/cache/<namespace>/`. `with_retry(fn, tries=3, base=1.5)` decorator.
- `timing.py`: `class StageTimer` with `with timer.stage("retrieval"): ...` and `.ms(name)`, `.total_ms()`.

**Verify:** `pytest tests/test_scaffold.py` — tests that config loads defaults, cache returns stored value on second call without invoking the function, retry retries then raises, timer records > 0 ms.

---

### M1 — PDF & Image Ingestion
**File:** `ingest/pdf_loader.py`, `ingest/ocr.py`, `ingest/vision.py`

```python
def load_pdf(path: Path, document_id: str) -> list[PageData]: ...
def load_image(path: Path, document_id: str) -> list[PageData]: ...   # single-page doc

class PageData(BaseModel):
    page: int
    text: str
    ocr_used: bool
    tables_md: list[str]
    figures: list[FigureData]       # saved PNG paths + page bbox
    page_image_path: str
```

**Behaviour:**
1. For each PDF page: extract text with PyMuPDF (`page.get_text("text")`).
2. If `len(text.strip()) < OCR_MIN_CHARS` → render at `RENDER_DPI` → `ocr_page(image) -> (text, mean_conf)`. If `mean_conf < 0.5`, call `vision.transcribe_page(image)` instead.
3. Always save a page render to `storage/pages/<doc>_p<N>.png` (used for thumbnails and the viewer).
4. Tables: `page.find_tables()`; convert each to markdown. If PyMuPDF finds none but text looks tabular, optionally try pdfplumber. Skip empty tables.
5. Embedded images: `page.get_images(full=True)`; skip if either side < `MIN_IMG_PX`, or aspect ratio > 8 or < 0.125, or repeated on > 50% of pages (logos). Save to `storage/figs/`.
6. `load_image`: treat a standalone image as one page: OCR it, and also treat the full image as a figure.
7. Wrap per-page logic in try/except; on failure return a `PageData` with empty text and log.

**`vision.py`:**
```python
def describe_figure(image_path: Path) -> FigureDescription: ...   # cached, retried
def transcribe_page(image_path: Path) -> str: ...
```
`describe_figure` prompt is in section 7.1; parse JSON robustly (strip code fences; on parse failure fall back to `{"caption": raw_text}`). Flatten into text:
```
[{type}] {title}. {caption}. Values: label=value unit; ... Relationships: {relationships}. Text: {visible_text}
```

**Verify:** `pytest tests/test_ingest.py` using fixtures: (a) a text PDF → text extracted, `ocr_used=False`; (b) the same PDF rasterised to images-only PDF → `ocr_used=True` and key words recovered; (c) a PDF with a table → at least one markdown table with `|`; (d) a PDF with a chart image → one figure saved; (e) a standalone PNG → one page. `describe_figure` is mocked in unit tests.

---

### M2 — Chunking
**File:** `ingest/chunker.py`

```python
def chunk_document(pages: list[PageData], document_id: str, source: str) -> list[Chunk]: ...
```

**Rules:**
- Text → paragraph-aware chunks of about `CHUNK_TOKENS` tokens (approximate tokens as `len(text.split()) * 1.3`), `CHUNK_OVERLAP` overlap. Detect headings heuristically (short line, Title Case/ALL CAPS, no terminal period) and set `section` for following chunks.
- Table → one chunk per table (`content_type="table"`); if > 1500 tokens, split by row groups repeating the header row. Prefix with nearest heading.
- Figure → one chunk (`content_type="image"`) with flattened vision text + any caption text found on the page near the figure (lines starting with "Figure"/"Fig."/"Chart"/"Table").
- OCR'd pages produce `content_type="ocr"` text chunks.
- `chunk_id` globally unique via a counter persisted in the store.
- `page_image_path` filled on every chunk.
- Each chunk's `content` stays clean. The embedding text adds a header: `f"Doc: {source} | Page: {page} | Section: {section}\n{content}"` (built in the embedder, not stored).

**Verify:** `pytest tests/test_chunker.py` — chunk sizes within bounds, tables never split mid-row, every chunk has page ≥ 1 and a source, figure chunks carry `image_path`.

---

### M3 — Embeddings & Indexes
**Files:** `index/embedder.py`, `vector_store.py`, `bm25_store.py`, `store.py`

```python
class Embedder:
    def __init__(self, model_name, device=None): ...
    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray   # float32, L2-normalised
    def embed_query(self, q: str) -> np.ndarray
    # bge models: prefix queries with "Represent this sentence for searching relevant passages: "

class ChunkStore:
    def add_document(self, doc_id, chunks, embeddings) -> None
    def remove_document(self, doc_id) -> None      # rebuild indexes without it
    def get(self, chunk_id) -> Chunk
    def all_chunks(self) -> list[Chunk]
    def save(self) / load(cls, root)               # chunks.jsonl + faiss.index + id map
    def stats(self) -> dict                        # counts per doc and content_type
```

- Batch embedding (batch_size 32 on GPU, 8 on CPU), progress logging.
- BM25: tokenise lowercase on `\w+`, keep digits and hyphenated IDs (`M-204` stays searchable by keeping both `m-204` and `m`, `204` tokens).
- Persist to `DOCURAG_STORAGE/index/`. `load` must work on a fresh Colab runtime pointing at Drive.
- Thread-safe `add_document` (a simple `threading.Lock`), since Django will ingest in a background thread.

**Verify:** `pytest tests/test_index.py` — add chunks, save, load, search returns same top ids; remove_document removes its chunks from both indexes; BM25 finds an exact ID token that dense search ranks poorly.

---

### M4 — Hybrid Retrieval + Rerank
**Files:** `retrieve/hybrid.py`, `retrieve/rerank.py`

```python
def hybrid_search(query, store, embedder, *, filters: Filters | None = None,
                  dense_k=..., bm25_k=..., rrf_k=...) -> list[tuple[Chunk, float]]
def rerank(query, candidates, reranker, top_n) -> list[tuple[Chunk, float]]
def retrieve(query, store, embedder, reranker, *, final_k, filters=None) -> list[Evidence]
```

**Behaviour:**
1. Dense top-`DENSE_K` and BM25 top-`BM25_K` (apply `filters`: `document_ids`, `content_types`).
2. RRF: `score(c) = Σ 1/(RRF_K + rank_i(c))`; keep top `RERANK_IN`.
3. Cross-encoder rerank on `(query, chunk.content)`; keep top `FINAL_K`.
4. **Modality diversity rule:** if the query contains any of `chart, graph, figure, plot, trend, diagram, image, picture, table` (case-insensitive) and the final set has no `image`/`table` chunk, replace the lowest-ranked item with the best-scoring chunk of that type from the candidate pool (if one exists).
5. Normalise reranker scores to `norm_score` in 0..1 via sigmoid on the raw logit.
6. Assign `eid` = `E1..En` in final rank order.

**Verify:** `pytest tests/test_retrieve.py` on a tiny synthetic corpus: paraphrase query ("excessive heat" vs doc "thermal generation") finds the right chunk in top-3; exact ID query works; diversity rule inserts a figure chunk for a "chart" query.

---

### M5 — Grounded Generation, Verification, Confidence, Timing
**Files:** `generate/prompts.py`, `answer.py`, `verify.py`, `confidence.py`, `pipeline.py`

```python
def generate_answer(question, evidence: list[Evidence]) -> GeneratedAnswer   # {answer, citations, answerable}
def verify_answer(answer: GeneratedAnswer, evidence) -> VerifierResult       # {support_ratio, per_claim: [...]}
def compute_confidence(evidence, support_ratio, answer) -> Confidence

# pipeline.py — the ONLY public API of the engine
def ingest_file(path: Path, *, filename: str | None = None) -> IngestReport
def ask(question: str, *, filters: Filters | None = None) -> AskResult
def get_engine() -> Engine           # lazy singleton: loads models + store once
```

**`generate_answer`:**
- Build evidence blocks: `[E1] {source} p.{page} ({content_type}):\n{content}`.
- For up to 3 evidence items with `image_path`, attach the actual image bytes to the Gemini request so the model reads the original chart/diagram.
- Use the prompt in 7.2; request JSON; parse robustly (strip fences; one repair retry on invalid JSON; final fallback → `answerable=false`).
- Validate that every cited id exists in evidence; drop unknown ids.

**`verify_answer`:** second LLM call using prompt 7.3; returns `support_ratio = supported_claims / total_claims` (1.0 if the answer is a refusal).

**Retry logic in `ask()`:**
```
evidence = retrieve(final_k=FINAL_K)
gen = generate_answer(...)
ver = verify_answer(...)
if gen.answerable and ver.support_ratio < SUPPORT_THRESHOLD:
    evidence = retrieve(final_k=FINAL_K*2, expand_neighbor_pages=True)   # one retry only
    gen = generate_answer(...); ver = verify_answer(...)
```
If `answerable` is false at the end → `confidence.score` capped at 0.2, label `Low`.

**Confidence formula (document it in code comments as a heuristic):**
```
top = max(e.norm_score for e in cited_evidence)  # or all evidence if no citations
coverage = sentences_with_citation / total_sentences   # 0 if none
score = 0.4*top + 0.4*support_ratio + 0.2*coverage
label = High if >= 0.80, Medium if >= 0.50, else Low
```

**Timing:** wrap retrieval, rerank, generation, verification in `StageTimer`; return all fields in `Timing`.

**Sources:** one `Source` per cited evidence only (not all retrieved), `snippet` = first 300 chars, `thumbnail_url` = `/media/pages/<file>` path (relative; the backend resolves).

**Verify:**
- `pytest tests/test_generate.py` with mocked LLM: JSON parsing (clean, fenced, broken-then-repaired), unknown citation ids dropped, refusal path sets low confidence, retry triggers once when support is low, timing fields populated.
- `pytest -m live tests/test_live_smoke.py` (needs `GEMINI_API_KEY`): ingest the fixture PDF, ask one question, assert non-empty answer with at least one source.

---

### M6 — Notebooks & Demo Corpus (core engine proof)
**Tasks:**
- `notebooks/01_ingest_and_index.ipynb`: mount Drive, `pip install -r requirements.txt`, set secrets from Colab `userdata`, ingest every file in `data/raw/`, print `store.stats()` and 3 sample chunks per content type.
- `notebooks/02_retrieval_and_answer.ipynb`: load store, run 6 sample questions (one per category in section 10), display answer, sources, confidence, timing.
- Add `scripts/ingest_dir.py` CLI: `python scripts/ingest_dir.py data/raw` (idempotent — skips already ingested filenames by content hash).
- Create `data/SOURCES.md` template (the human fills in the files; list which categories are still missing: text, table-heavy, chart-heavy, diagram, scanned PDF, standalone images).

**Verify:** `python scripts/ingest_dir.py data/raw && python -m docurag.pipeline "test question"` prints a JSON `AskResult`. Update `PROGRESS.md` with what documents the human still needs to add.

---

### M7 — Django Backend + Frontend (MUST)
**Models (`api/models.py`):**
```python
class Document(models.Model):
    filename, file (FileField), file_type, pages (int, null), status (queued|processing|ready|failed),
    error (text), chunk_counts (JSONField), uploaded_at
class QueryLog(models.Model):
    question, answer, answerable, confidence (float), total_ms (int),
    timing (JSON), sources (JSON), agent_trace (JSON), created_at
```
Chunks live in the engine's `ChunkStore` (not duplicated in the Django DB).

**Endpoints (DRF):**

| Method | Path | Behaviour |
|---|---|---|
| POST | `/api/documents/` | multipart `file`; validate extension (`.pdf .png .jpg .jpeg .webp`) and size ≤ 50 MB; create `Document(status=queued)`; start background thread running `pipeline.ingest_file`; return 202 + document JSON |
| GET | `/api/documents/` | list with status, pages, chunk_counts |
| GET | `/api/documents/<id>/` | detail |
| DELETE | `/api/documents/<id>/` | remove from store + DB + files |
| GET | `/api/documents/<id>/pages/<n>/` | serve page PNG |
| POST | `/api/query/` | `{question, document_ids?, content_types?}` → `AskResult` JSON; also writes `QueryLog` |
| GET | `/api/history/` | last 50 queries |
| GET | `/api/health/` | model-loaded flags, chunk counts, device |

- Reject empty/whitespace questions with 400. Cap question length at 1000 chars.
- Background ingestion: `threading.Thread(daemon=True)`; update `status` and `error`; engine's store lock handles concurrency. Do not introduce Celery.
- Serve `storage/pages` and `storage/figs` via `MEDIA_URL=/media/`.
- Settings: `ALLOWED_HOSTS` and `CSRF_TRUSTED_ORIGINS` read from env (comma-separated) so a tunnel URL can be injected. Use `DEBUG` from env. For API views, exempt CSRF (or use DRF `SessionAuthentication`-free config) so the vanilla JS frontend works behind a tunnel.
- Engine singleton loads once at startup (`apps.py ready()` or lazy on first request; lazy preferred to keep `manage.py` commands fast).

**Frontend (templates + `static/app.js`, Tailwind CDN):**
1. **Library page** (`/`): drag-and-drop upload; table of documents with status badge (polls `/api/documents/` every 2 s while any doc is queued/processing), pages, counts of text/table/image chunks, delete button.
2. **Ask page** (`/ask/`): textarea + Ask button + loading state; renders
   - answer text with `[E#]` converted into clickable chips,
   - Sources panel: cards with document, page, type icon (📄 text/ocr, 📊 table, 🖼️ figure), snippet, thumbnail,
   - Confidence badge (colour by label) with the percentage,
   - Response time pill with expandable per-stage breakdown,
   - "Insufficient evidence" styled state when `answerable=false`.
3. **Viewer drawer:** clicking a chip/source opens a side drawer showing the full page image (`/api/documents/<id>/pages/<n>/`) with the cited snippet below.
4. **History page** (`/history/`): list of past queries; click to re-display the stored result.
5. Optional filter dropdown on Ask page: limit to specific documents.

Keep the UI clean and responsive; no external JS frameworks.

**Verify:**
- `pytest backend/tests` (Django `APIClient` with the engine mocked): upload → 202; list shows doc; query returns the contract shape; empty question → 400; bad extension → 400.
- Manual smoke (document the steps in README): `python backend/manage.py migrate && python backend/manage.py runserver`, upload a PDF, ask, click a source.

---

### M8 — Agent Layer: LangGraph + MCP (STRETCH)
Start only after M7 passes. Install from `requirements-agents.txt`.

**MCP servers (FastMCP, stdio):** thin wrappers over engine functions, no new logic.
- `document_server.py`: `list_documents()`, `get_page_text(document_id, page)`, `get_page_image(document_id, page)`.
- `retrieval_server.py`: `hybrid_search(query, k, document_ids?, content_types?)`, `search_images(query, k)`, `search_tables(query, k)`.
Each tool returns JSON-serialisable dicts with `source`, `page`, `content_type`, `snippet`, `score`.

**LangGraph (`agents/graph.py`):** state = `{question, plan, evidence, draft, verification, retries, trace}`.
Nodes: `analyze_question` → `retrieve` → `vision_read` (conditional) → `generate` → `verify` → (`retrieve` once more | `finalize`).
- `analyze_question`: rule-based or one cheap LLM call; outputs `{needs_table, needs_image, multi_doc, out_of_scope}`.
- `retrieve` calls the MCP retrieval tools via an in-process MCP client (or direct tool functions if MCP client wiring costs > 1 hour; note the choice in `DECISIONS.md`).
- Each node appends its name (and tool names called) to `trace`.
- `finalize` builds the same `AskResult` as the linear path, including `agent_trace`.

`pipeline.ask()` dispatches to the graph when `USE_AGENTS=true`, else linear path. Both must return an identical `AskResult` shape.

**UI:** if `agent_trace` is non-empty, show a collapsible "Agent trace" list on the Ask page.

**Verify:** `pytest tests/test_agents.py` (mocked LLM): graph reaches `finalize`; low support triggers exactly one extra retrieve; trace lists nodes; `AskResult` validates. MCP servers start with `python -m docurag.agents.mcp_servers.retrieval_server` without error.

---

### M9 — Evaluation Harness (STRETCH, high value)
**File:** `scripts/eval.py`, `notebooks/03_eval.ipynb`, `data/eval/gold_qa.json`

Gold file schema:
```json
[{"id":"q01","type":"paraphrase|table|chart|scanned|diagram|multidoc|unanswerable",
  "question":"...","expected_answer":"...","gold_sources":[{"document":"X.pdf","page":12}]}]
```
(The human writes the questions after the corpus is final; the agent provides 5 placeholder examples and the loader.)

**Metrics (per type and overall):**
- `hit@5`: any gold (document, page) in the final evidence.
- `answer_correct`: LLM-judge (prompt 7.4) comparing answer to `expected_answer` → 0/1.
- `faithfulness`: mean verifier `support_ratio`.
- `refusal_accuracy`: on `unanswerable` items, fraction with `answerable=false`.
- `latency`: median and p95 `total_ms`.
- **Ablation:** run retrieval-only hit@5 for `bm25`, `dense`, `hybrid`, `hybrid+rerank` and print a markdown table.

Output `eval_results.md` + `eval_results.json`.

**Verify:** `python scripts/eval.py --gold data/eval/gold_qa.json` produces both files with all metrics.

---

## 7. Prompts (use verbatim; keep in `prompts.py`)

### 7.1 Figure description (vision)
```
You are indexing a figure from a document so it can be found by text search.
Describe the image and return ONLY valid JSON with this shape:
{
  "type": "bar_chart|line_chart|pie_chart|table|diagram|photo|scanned_text|other",
  "title": "<title if visible, else empty>",
  "caption": "<2-3 sentence description of what the figure shows>",
  "key_values": [{"label": "<x/category/series>", "value": "<number or text>", "unit": "<unit or empty>"}],
  "relationships": "<for diagrams/flowcharts: A -> B -> C; else empty>",
  "visible_text": "<all legible text in the image>"
}
Read numbers exactly as shown. If a value is unreadable, omit it. Do not guess.
```

### 7.2 Grounded answer
```
You answer questions using ONLY the evidence blocks below. 
Rules:
1. Use only facts stated in the evidence (text, tables, or the attached images). Do not use outside knowledge.
2. After every claim, cite the evidence ids in square brackets, e.g. [E1] or [E2][E4].
3. If the evidence does not contain enough information, set "answerable": false and say what is missing. Do not guess.
4. For numbers from charts/tables, quote them exactly and state units.
5. If sources disagree, say so and cite both.
6. Be concise (max ~150 words unless the question needs more).

Return ONLY valid JSON:
{"answer": "<text with [E#] citations>", "citations": ["E1","E3"], "answerable": true}

QUESTION: {question}

EVIDENCE:
{evidence_blocks}
```

### 7.3 Verifier
```
You are a strict fact-checker. Given an ANSWER and the EVIDENCE it cites, split the answer into atomic claims 
and decide for each whether the cited evidence directly supports it.
Return ONLY valid JSON:
{"claims": [{"claim": "<text>", "supported": true|false, "reason": "<short>"}]}
Mark a claim unsupported if it adds facts not present in the evidence, even if plausible.

ANSWER: {answer}

EVIDENCE:
{cited_evidence_blocks}
```

### 7.4 Eval judge
```
Question: {question}
Expected answer: {expected}
Model answer: {answer}
Does the model answer convey the same key facts as the expected answer (numbers must match)? 
Return ONLY JSON: {"correct": true|false, "reason": "<short>"}
```

---

## 8. Testing Strategy

- **Unit (default, offline):** M0–M5 with mocked LLM/vision and tiny fixtures. Must run in < 60 s on CPU. Use a lightweight embedder stub (`FakeEmbedder` with hash-based vectors) for index/retrieval tests where model downloads are not needed; keep one slow test marked `@pytest.mark.slow` using the real embedder.
- **Live (`-m live`):** one end-to-end smoke test with a real Gemini key.
- **Fixtures to generate programmatically** (so no binary files need to be supplied): build a text PDF, a table PDF, a chart PDF (matplotlib → PNG → embed), and a scanned-style PDF (rasterise text PDF to image-only PDF) using PyMuPDF/matplotlib in `tests/conftest.py`.
- CI-style command: `pytest -q` (default) must pass before every commit.

---

## 9. Colab Notes (put in README + `04_colab_serve.ipynb`)

1. Mount Drive; set `DOCURAG_STORAGE=/content/drive/MyDrive/docurag_storage`.
2. Load secrets from `google.colab.userdata` into env (`GEMINI_API_KEY`, optional `NGROK_AUTHTOKEN`).
3. `pip install -r requirements.txt` (restart runtime if prompted).
4. Run `python backend/manage.py migrate` then `runserver 0.0.0.0:8000` as a background process.
5. Start `cloudflared tunnel --url http://localhost:8000` (or ngrok); parse the public URL; export it into `ALLOWED_HOSTS` and `CSRF_TRUSTED_ORIGINS` **before** starting Django (start the tunnel first, then Django).
6. Print the public URL.
Use `faiss-cpu` (corpus is small); GPU is used only by the embedder, reranker and EasyOCR.

---

## 10. Required Question Categories (for notebooks 02/03 and the demo)

| Category | Example shape |
|---|---|
| Paraphrase | Query uses different wording than the source |
| Table | "What was X in <row/column>?" |
| Chart | Numeric value or trend only visible in a figure |
| Scanned | Fact only available through OCR |
| Diagram | Component order / connection |
| Multi-document | Needs two or more sources |
| Unanswerable | Must return `answerable=false` |

---

## 11. Definition of Done

**MUST (M0–M7)**
- [ ] `pytest -q` passes offline
- [ ] PDF and image upload via UI; background ingestion with status polling
- [ ] Scanned PDF → OCR'd → searchable
- [ ] Tables → markdown chunks; figures → vision chunks with values
- [ ] Hybrid retrieval + reranker + modality-diversity rule
- [ ] Grounded answers with `[E#]` citations mapped to document + page
- [ ] Clickable sources open the page image
- [ ] Confidence (score + label) and per-stage timing on every answer
- [ ] Refusal path (`answerable=false`) works and is styled
- [ ] Query history persisted
- [ ] Runs in Colab behind a tunnel (documented, tested steps)
- [ ] README with architecture diagram, setup, run, and demo instructions

**STRETCH**
- [ ] M8 agent graph + MCP servers + agent trace in UI
- [ ] M9 eval harness + ablation table

---

## 12. Things the Human Must Provide (agent: list these in `PROGRESS.md` when reached)

1. `GEMINI_API_KEY` (and optional `NGROK_AUTHTOKEN`).
2. 8–12 public documents in `data/raw/` covering: text-heavy, table-heavy, chart-heavy, technical diagram, a scanned PDF, and 2–3 standalone images; fill `data/SOURCES.md`.
3. 25–30 gold Q&A items in `data/eval/gold_qa.json` (M9).
4. Final decision on Gemini model name if the default is unavailable.

---

## 13. Kickoff Prompt (paste this into the coding agent)

```
Read DocuRAG_AI_Build_Spec.md fully. Implement it milestone by milestone starting at M0.
Rules: follow the Operating Rules in section 0; keep docurag/ free of Django imports; mock all LLM
calls in tests; commit after each milestone; update PROGRESS.md after each milestone with status and
anything I need to provide. Do not start M8 until M7's verify passes. After each milestone, run its
verify command and show me the result before continuing. If anything is ambiguous, choose the
simplest option and record it in DECISIONS.md rather than asking.
Start with M0 now.
```
