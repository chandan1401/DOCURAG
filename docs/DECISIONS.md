# Architectural & Engineering Decisions

## 1. Schema Harmonization between Base Spec and PROJECT.md
- **Context**: `DocuRAG_AI_Build_Spec.md` §5 specified `Chunk` fields (`chunk_id`, `document_id`, `source`, `page`, `content_type`, `section`, `content`), while `PROJECT.md` referenced `doc_id`, `doc_name`, `text`, `token_count`, `is_table`, `is_figure`, `bbox`.
- **Decision**: Implemented canonical Base Spec §5 attributes as primary fields and added property getters (`chunk.doc_id`, `chunk.doc_name`, `chunk.text`, `chunk.is_table`, `chunk.is_figure`) along with alias mapping in `__init__`. This provides 100% mutual compatibility across all subsequent worker modules without breaking Pydantic serialization.

## 2. Multi-Agent Trace Schema Upgrade (`TraceStep` v2)
- **Context**: Base Spec §5 initially used `agent_trace: list[str]`. Multi-Agent Spec §A3 upgraded this to structured `TraceStep` (`agent`, `action`, `tool`, `detail`, `ms`).
- **Decision**: Adopted `TraceStep` v2 across all schemas and API responses. When in linear mode, `agent_trace` defaults to an empty list `[]`. In agent mode, it stores the full execution sequence.

## 3. Atomic Disk Caching with Corruption Self-Healing
- **Context**: In multi-threaded ingestion or query execution, concurrent writes could corrupt cache files.
- **Decision**: Cache files are written to a unique PID/timestamp temporary file on the same filesystem volume, then atomically moved via `os.replace`. If a cache read encounters `json.JSONDecodeError`, the corrupted file is automatically unlinked and the value recomputed.

## 4. Cumulative Stage Timing
- **Context**: Verifier retry loops in M5 execute retrieval and generation stages multiple times.
- **Decision**: `StageTimer` accumulates milliseconds across repeated stage invocations (`self._stages[name] += elapsed`), ensuring that reported timing metrics accurately represent total system work.

## 5. Hermetic Test Infrastructure
- **Context**: Running unit tests offline in <60 seconds without GPU or external network access.
- **Decision**: Implemented `FakeEmbedder` (deterministic 384-dimensional unit vector using SHA-256 projections), `FakeReranker` (token overlap Jaccard scoring), and `FakeLLM` (scripted structured JSON generator). When `DOCURAG_FAKES=1` is set, all pipeline steps execute deterministically without external API calls.
