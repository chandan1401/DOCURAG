# DocuRAG — Multi-Agent Spec

> **Companion to `DocuRAG_AI_Build_Spec.md`** (call it the *Base Spec*). The Base Spec stays the source of truth for schemas, prompts, config, endpoints and acceptance tests. This file adds two things:
>
> - **Part A — Build agents:** how to split the build across several coding agents working in parallel, each owning one workflow, with strict file ownership and handoffs.
> - **Part B — Runtime agents:** the multi-agent architecture *inside* DocuRAG, where specialist agents handle different workflows (ingestion, question answering, multi-document comparison, chart/table analysis, verification). This replaces the single "stretch" graph in Base Spec M8 and makes it a first-class feature.
>
> Deadline: build ends **12 Oct**, evaluation **13 Oct 2026**. Core pipeline (Base Spec M0–M7) must work **without** Part B. Runtime agents are an upgrade layer behind a flag (`USE_AGENTS`).

---

# PART A — BUILD AGENTS (parallel coding workflow)

## A1. Why split, and the one rule that makes it work

Parallel agents only help if they cannot collide. The rule: **contracts first, then ownership.** One Lead agent freezes the interfaces (schemas, function signatures, fakes, fixtures). After that, each agent codes only inside its own files, against stubs, and never waits on another agent's real implementation.

## A2. Agent roster

| # | Agent | Owns (only edits these) | Base Spec milestones | Depends on |
|---|---|---|---|---|
| 0 | **Lead / Integrator** | `docurag/config.py`, `schemas.py`, `cache.py`, `timing.py`, `pipeline.py`, stubs, `tests/conftest.py`, `CONTRACTS.md`, `PROGRESS.md`, merges | M0, parts of M5/M6 | — |
| 1 | **Ingestion Agent** | `docurag/ingest/*` | M1, M2 | Phase 0 |
| 2 | **Retrieval Agent** | `docurag/index/*`, `docurag/retrieve/*` | M3, M4 | Phase 0 |
| 3 | **Generation Agent** | `docurag/generate/*` | M5 | Phase 0 |
| 4 | **Runtime-Agents Agent** | `docurag/agents/*` (Part B) | M8 (expanded) | Phase 0 |
| 5 | **Backend Agent** | `backend/api/*`, `backend/config/*` | M7 (API) | Phase 0 |
| 6 | **Frontend Agent** | `backend/templates/*`, `backend/static/*` | M7 (UI) | Phase 0 (API contract) |
| 7 | **QA & Eval Agent** | `tests/*` (except `conftest.py`), `scripts/eval.py`, `data/eval/*`, `notebooks/*`, `README.md` | M6, M9, test suites | Phase 0 |

A single human can run these as separate sessions (git worktrees) or one orchestrating tool can spawn them as sub-agents. If only one agent is available, run them **sequentially in the order 0 → 1 → 2 → 3 → 5 → 6 → 4 → 7**; the ownership rules still apply.

## A3. Phase 0 — Contract Freeze (Lead only, serial, ~2–3 hours)

Nothing else starts until this passes. The Lead produces:

1. **Repo scaffold** per Base Spec §3 and M0.
2. **`schemas.py`** per Base Spec §5, with this change (v2): replace `agent_trace: list[str]` with
   ```python
   class TraceStep(BaseModel):
       agent: str          # "router", "retrieval", ...
       action: str         # "classify", "hybrid_search", ...
       tool: str | None = None
       detail: str = ""    # short human-readable note
       ms: int = 0
   # AskResult.agent_trace: list[TraceStep] = []
   ```
3. **Interface stubs** (signatures + docstrings + `raise NotImplementedError`, or a working fake) for every public function in Base Spec M1–M5:
   `load_pdf, load_image, describe_figure, chunk_document, Embedder, ChunkStore, hybrid_search, rerank, retrieve, generate_answer, verify_answer, compute_confidence, ingest_file, ask`.
4. **Fakes** usable by everyone: `FakeEmbedder` (hash-based vectors), `FakeLLM` (returns scripted JSON for answer/verify/vision prompts), `FakeReranker` (token-overlap score). Selected by env `DOCURAG_FAKES=1`.
5. **Fixture factory** in `tests/conftest.py`: generates text PDF, table PDF, chart PDF, scanned-style PDF, standalone PNG, and a 5-chunk tiny corpus.
6. **`CONTRACTS.md`**: frozen API JSON (query response, document JSON), the `TraceStep` format, error codes, and the Part B tool signatures (§B6).
7. **Branch/merge setup:** `main` protected by "tests pass"; one branch per agent: `agent/ingestion`, `agent/retrieval`, `agent/generation`, `agent/runtime`, `agent/backend`, `agent/frontend`, `agent/qa`.

**Verify:** with `DOCURAG_FAKES=1`, `python -c "from docurag.pipeline import ask; print(ask('hello'))"` returns a valid `AskResult` (from fakes), and `pytest -q` passes.

## A4. Parallel phase rules (all agents)

1. **Edit only files you own.** Need a change elsewhere? Append a request to `CONTRACT_CHANGES.md` (who, file, exact change, why) and keep working against the stub. The Lead applies approved changes on `main`; you rebase.
2. **Contracts are frozen.** Signatures in `CONTRACTS.md` don't change without a Lead-approved entry. Adding optional parameters with defaults is allowed; renaming or removing is not.
3. **Code against stubs/fakes** until the real dependency lands on `main`; then switch tests to the real one.
4. **Every PR must include:** passing unit tests for your area, an update to `PROGRESS.md` (your section only), and the verify command output.
5. **No new dependencies** without a `CONTRACT_CHANGES.md` entry (the Lead edits `requirements*.txt` — it is a shared file).
6. **LLM calls in tests are always mocked** (`FakeLLM`). Live tests only by the QA Agent.
7. **Rebase on `main` before opening a PR.** Small PRs, merged in the order in §A6.

## A5. Per-agent briefs

### Agent 1 — Ingestion
Implement Base Spec M1 + M2 exactly. Extra requirements:
- Provide `ingest/triage.py: classify_page(page) -> Literal["text","scanned","chart_like","mixed"]` (rule-based: text density, image area ratio). The Runtime-Agents Agent calls this in the ingestion workflow (§B4), so keep it pure and deterministic.
- Expose per-page results incrementally via a generator `iter_pages(path)` in addition to `load_pdf()`, so the backend can report progress.
- **Done when:** Base Spec M1 and M2 verify commands pass on real (non-fake) code with `describe_figure` mocked.

### Agent 2 — Retrieval
Implement Base Spec M3 + M4. Extra requirements:
- `retrieve(...)` accepts an optional `query_type` hint (`"general"|"table"|"chart"`) that biases the modality-diversity rule, used by the Router (§B3).
- Add `retrieve_for_documents(query, document_ids, ...)` to support per-document sub-queries in the comparison workflow (§B5.3).
- **Done when:** Base Spec M3 and M4 verify commands pass; hit@5 on the tiny synthetic corpus ≥ 0.9 for paraphrase queries.

### Agent 3 — Generation
Implement Base Spec M5. Extra requirements:
- `generate_answer` accepts an optional `instructions: str` appended to the system prompt (used by specialist agents).
- Provide `synthesize_comparison(question, sub_answers)` for the comparison workflow (prompt in §B7.2).
- Provide `extract_chart_values(question, evidence)` (vision-assisted, returns structured values) for the Chart/Table Analyst (§B5.4).
- **Done when:** Base Spec M5 verify passes with mocked LLM; one live smoke test passes when a key is available.

### Agent 4 — Runtime Agents
Implement Part B in full (`docurag/agents/*`). Starts after Phase 0; uses stubs for ingestion/retrieval/generation until their PRs land. **Done when:** Part B acceptance tests (§B9) pass with `DOCURAG_FAKES=1`.

### Agent 5 — Backend
Implement Base Spec M7 API half. Extra requirements:
- `POST /api/query/` accepts optional `"mode": "auto" | "linear" | "agents"`; `auto` follows `USE_AGENTS`.
- `GET /api/documents/<id>/progress/` returns per-page ingestion progress (uses `iter_pages`).
- `agent_trace` persists as JSON in `QueryLog`.
- **Done when:** Base Spec M7 backend tests pass using the engine in fakes mode.

### Agent 6 — Frontend
Implement Base Spec M7 UI half against `CONTRACTS.md` (develop with a JSON mock server or the fakes-mode backend). Extra requirements:
- **Agent Trace panel:** vertical timeline of `TraceStep`s (agent badge, action, tool, ms), collapsible, shown when `agent_trace` is non-empty. Colour-code per agent.
- **Mode toggle** on the Ask page: Auto / Linear / Agents.
- **Ingestion progress bar** per document using the progress endpoint.
- **Done when:** the full flow (upload → progress → ask → answer, sources, confidence, timing, trace, viewer drawer) works against the fakes-mode backend.

### Agent 7 — QA & Eval
- Own all test suites beyond each agent's unit tests: **integration tests** (`tests/integration/`) that run ingest→ask end to end in fakes mode and, when `-m live`, with real Gemini.
- Implement Base Spec M6 notebooks and M9 eval harness. Add per-mode comparison: run the gold set with `linear` and `agents` and report both (this shows whether the agents actually help).
- Write the README (architecture diagram including the agent graph, setup, Colab steps).
- Maintain `data/SOURCES.md` and list what documents the human must still provide.
- **Done when:** `pytest -q` passes on `main`; `python scripts/eval.py` runs for both modes.

## A6. Merge order and integration checkpoints

```
Phase 0 (Lead) ──► main
        │
        ├── agent/ingestion  ─┐
        ├── agent/retrieval  ─┼─► merge 1: ingestion → retrieval → generation  ─► CHECKPOINT 1
        ├── agent/generation ─┘      (real linear pipeline works end to end, no agents)
        │
        ├── agent/backend ─┬─► merge 2: backend → frontend                      ─► CHECKPOINT 2
        └── agent/frontend ┘      (full web demo, linear mode)
        │
        └── agent/runtime ───► merge 3: runtime agents                          ─► CHECKPOINT 3
                                      (agent mode, trace panel, eval both modes)
        agent/qa merges continuously (tests, notebooks, README)
```

- **Checkpoint 1:** `python scripts/ingest_dir.py data/raw && python -m docurag.pipeline "<question>"` returns a real grounded `AskResult`.
- **Checkpoint 2:** web UI demo runs in Colab behind a tunnel with the linear path. **This is the minimum shippable demo.**
- **Checkpoint 3:** `mode=agents` returns the same `AskResult` shape plus a populated trace, and the eval harness reports both modes.

## A7. Schedule (8–12 Oct)

| Day | Lead | Agents 1–3 | Agents 4–6 | Agent 7 |
|---|---|---|---|---|
| Thu 8 | Phase 0 contract freeze | start on stubs | start on stubs | fixtures, README skeleton |
| Fri 9 | review/merge; Checkpoint 1 | land M1–M5 | backend API; UI shell; runtime agents against fakes | integration tests |
| Sat 10 | merge 2; Checkpoint 2 | bug fixes; live-test tuning | UI complete; runtime agents wired to real functions | notebooks, live tests |
| Sun 11 | Checkpoint 3 | tuning from eval | trace panel; agent fallbacks | eval both modes, ablation |
| Mon 12 | freeze, tag `v1.0-demo`, rehearse | bug fixes only | bug fixes only | final eval, backup video, README |

**Cut order if behind:** Agents 4's comparison workflow → chart analyst → trace panel polish → mode toggle. Never cut Checkpoint 2.

---

# PART B — RUNTIME AGENTS (inside DocuRAG)

## B1. Principle

A multi-agent system is justified only where it beats the linear pipeline. In DocuRAG, **different question types need different workflows**, and routing to the right specialist is the value. Everything else stays deterministic code.

| Component | LLM-driven agent? | Reason |
|---|---|---|
| Router (question classification) | **Yes** (one cheap call, rule-based fast path) | Needs semantic intent understanding |
| Retrieval Agent | **Partly** (chooses tools/filters; search itself is deterministic) | Picks modality and scope |
| Planner (decomposition) | **Yes** | Splits multi-document questions |
| Chart/Table Analyst | **Yes** (vision + calculator tool) | Reads figures, does arithmetic |
| Synthesizer | **Yes** | Combines sub-answers |
| Verifier | **Yes** | Independent fact-check |
| Ingestion Triage | **No** (rules) | Deterministic, cheaper, faster |
| Page workers (OCR, table, figure) | **No** (functions; vision call only for figures) | No decisions to make |

## B2. Agent map

```
                         ┌─────────────────┐
              question ─►│   ROUTER AGENT  │
                         └────────┬────────┘
        ┌───────────────┬─────────┼──────────────┬────────────────┐
        ▼               ▼         ▼              ▼                ▼
   FACTUAL         TABLE/CHART   MULTI-DOC      SUMMARIZE        OUT-OF-SCOPE
   workflow        workflow      COMPARE        DOC workflow     (refuse early)
        │               │        workflow           │
        ▼               ▼             ▼             ▼
   Retrieval       Retrieval     Planner ─► N×Retrieval   Retrieval(doc-scoped)
   Agent           Agent         Agents (parallel)        + Synthesizer
        │               │             │
        │          Analyst Agent      ▼
        │         (vision+calc)   Synthesizer
        └───────────────┴─────────────┘
                        ▼
                 ANSWER (generate)
                        ▼
                 VERIFIER AGENT ──fail──► Retrieval Agent (expanded, once)
                        ▼ pass
                 FINALIZE (citations, confidence, timing, trace)
```

Ingestion has its own small graph (§B4), independent of the query graph.

## B3. Router Agent

**Input:** question, list of available documents (names only).
**Output:**
```json
{"intent": "factual|table_lookup|chart_numeric|multi_doc_compare|summarize_doc|out_of_scope",
 "query_type": "general|table|chart",
 "document_hints": ["Maintenance_Report.pdf"],
 "needs_calculation": false}
```
- **Fast path (no LLM):** keyword rules first — `compare|versus|vs|difference between|across` → `multi_doc_compare`; `summari[sz]e|overview of` + a document name → `summarize_doc`; `chart|graph|plot|trend|figure` → `chart_numeric`; `table|row|column` → `table_lookup`. Use the LLM only if rules are ambiguous or match more than one intent.
- **Out-of-scope** = clearly not about the documents (chit-chat, general knowledge with no document link). Router returns it only with high confidence; otherwise default to `factual` and let retrieval + verifier handle refusal. Rationale: an over-eager router that refuses valid questions is worse than one that over-routes.
- Prompt: §B7.1.

## B4. Ingestion workflow (LangGraph or plain function graph)

```
file ─► TRIAGE (deterministic) ─┬─ text page ─────► text extractor ─┐
                                ├─ scanned page ──► OCR worker ─────┤
                                ├─ chart_like ────► figure worker ──┼─► table worker ─► CHUNKER ─► INDEXER
                                └─ mixed ─────────► all of above ───┘
```
- Triage uses `ingest/triage.py: classify_page`.
- Workers run **per page in a thread pool (max 4 workers)**; the vision calls inside the figure worker are rate-limited by a shared semaphore (max 2 concurrent) and cached.
- Progress events (`pages_done / pages_total`) feed `GET /api/documents/<id>/progress/`.
- Deterministic by design; this workflow's "agentic" part is the per-page routing, and its trace is shown in the library page ("page 7: scanned → OCR; page 12: chart → vision").

## B5. Query workflows

### B5.1 Retrieval Agent (shared by all workflows)
- **Tools (MCP):** `hybrid_search`, `search_tables`, `search_images`, `get_page_text`, `list_documents`.
- **Policy:** start with `hybrid_search` (filters from `document_hints`). If `query_type` is `table`/`chart`, also call `search_tables`/`search_images`. If top rerank score is below `LOW_SCORE` (default 0.35 normalised), call once more with query expansion (LLM or rule-based synonyms), then proceed with the best available. Max **3 tool calls** per invocation.
- **Output:** `list[Evidence]` (same as Base Spec).

### B5.2 Factual workflow
`Router → Retrieval Agent → generate_answer → Verifier → finalize` (the linear pipeline with a router and a tool-choosing retriever). Retry once on verifier failure.

### B5.3 Multi-document comparison workflow
1. **Planner Agent** decomposes the question into 2–4 sub-questions, each optionally scoped to a document (prompt §B7.3). Output: `[{"sub_question": "...", "document_ids": [..] | null}]`.
2. **N × Retrieval Agent** run in parallel (thread pool, max 4), each using `retrieve_for_documents`.
3. Each sub-question gets `generate_answer` (short, cited).
4. **Synthesizer** (`synthesize_comparison`) merges sub-answers, preserving every `[E#]` citation (eids re-numbered globally; keep a mapping).
5. Verifier checks the final answer against the union of cited evidence.
- Hard caps: ≤ 4 sub-questions, ≤ 1 retry overall.

### B5.4 Chart / Table Analyst workflow
1. Retrieval Agent with `query_type=chart|table`.
2. **Analyst Agent** receives the evidence images/tables and calls `extract_chart_values` → structured `[{"label","value","unit"}]`.
3. If `needs_calculation`, it calls the **calculator tool** (`analysis_server.calculate(expression)`) for differences, percentages, ratios. Calculator must be a **safe evaluator** (parse with `ast`, allow only numbers and `+ - * / ** ( )`, `abs`, `round`, `min`, `max`; no names, attributes, or calls beyond those). Never use `eval`.
4. `generate_answer` is called with the analyst's numbers appended as extra evidence block `[E_calc]`, so arithmetic is cited and checkable.
5. Verifier runs as usual.

### B5.5 Summarize-document workflow
Retrieval scoped to one document (`document_ids=[id]`), pull top-12 diverse chunks (by page spread), then `generate_answer` with `instructions="Produce a structured summary with page citations."`. No planner.

### B5.6 Verifier Agent (all workflows)
Independent from the generator: uses Base Spec prompt 7.3, ideally a different `temperature`/call. Fail → one expanded retrieval and regeneration; second fail → return the answer with `confidence.label="Low"` and a caveat, or `answerable=false` if the evidence is clearly insufficient.

## B6. MCP tool servers (FastMCP, stdio)

Thin wrappers over engine functions; no business logic duplicated. All tools return JSON-serialisable dicts. These signatures are **frozen in `CONTRACTS.md`**.

| Server | Tool | Signature |
|---|---|---|
| `document_server` | `list_documents` | `() -> [{"document_id","filename","pages","status"}]` |
| | `get_page_text` | `(document_id: str, page: int) -> {"text": str, "content_types": [..]}` |
| | `get_page_image` | `(document_id: str, page: int) -> {"path": str}` |
| `retrieval_server` | `hybrid_search` | `(query: str, k: int = 6, document_ids: list[str] \| None = None, content_types: list[str] \| None = None) -> [ChunkHit]` |
| | `search_tables` | `(query: str, k: int = 4, document_ids=None) -> [ChunkHit]` |
| | `search_images` | `(query: str, k: int = 4, document_ids=None) -> [ChunkHit]` |
| `analysis_server` | `calculate` | `(expression: str) -> {"result": float \| None, "error": str \| None}` |
| | `extract_chart_values` | `(chunk_id: str, question: str) -> {"values": [{"label","value","unit"}]}` |

`ChunkHit = {"chunk_id","document","page","content_type","snippet","score"}`.

**Wiring:** agents call tools through a single `ToolClient` class. Implementation A: real in-process MCP client over stdio. Implementation B: direct Python calls to the same functions. Choose via `MCP_MODE=stdio|direct` (default `direct` for demo reliability; `stdio` demonstrated in the README and a test). Both must produce identical results; record any gaps in `DECISIONS.md`.

## B7. Prompts (keep in `docurag/agents/prompts.py`)

### B7.1 Router
```
Classify the user's question about a document collection. Available documents: {doc_names}.
Return ONLY JSON:
{"intent": "factual|table_lookup|chart_numeric|multi_doc_compare|summarize_doc|out_of_scope",
 "query_type": "general|table|chart",
 "document_hints": [<exact document names mentioned or clearly implied, else []>],
 "needs_calculation": true|false}
Use "out_of_scope" ONLY if the question clearly has nothing to do with documents (e.g. greetings, 
general trivia). When unsure between intents, choose "factual".
QUESTION: {question}
```

### B7.2 Synthesizer (comparison)
```
Combine the sub-answers below into one answer to the original question. 
Keep every citation marker like [E3] attached to the claim it supports. Do not add facts that are not in the sub-answers. 
If sub-answers conflict, state the conflict. If a sub-question was unanswerable, say what is missing.
Return ONLY JSON: {"answer": "...", "citations": ["E1","E2"], "answerable": true|false}
ORIGINAL QUESTION: {question}
SUB-ANSWERS:
{sub_answers}
```

### B7.3 Planner
```
Break the question into 2-4 independent sub-questions that can each be answered by searching the documents. 
If the question mentions specific documents, scope the sub-question to them. 
Return ONLY JSON: {"sub_questions": [{"sub_question": "...", "document_names": ["..."] }]}
Available documents: {doc_names}
QUESTION: {question}
```

### B7.4 Analyst
```
From the supplied table/figure evidence, extract the values needed to answer the question. 
Return ONLY JSON: {"values":[{"label":"","value":"","unit":"","source":"E#"}],
                   "calculation": "<arithmetic expression using only numbers, or empty>"}
Read numbers exactly as shown. Never estimate unreadable values.
QUESTION: {question}
```

(Answer generation and verifier prompts: Base Spec §7.2 and §7.3.)

## B8. State, limits and fallbacks

**LangGraph state:**
```python
class AgentState(TypedDict):
    question: str
    route: RouteDecision | None
    sub_questions: list[SubQuestion]
    evidence: list[Evidence]
    analyst_values: list[dict]
    draft: GeneratedAnswer | None
    verification: VerifierResult | None
    retries: int
    trace: list[TraceStep]
    timer: StageTimer
```

**Limits (enforced in code, configurable):**
- Total LLM calls per question ≤ 12; total tool calls ≤ 10; wall-clock timeout 45 s.
- Retries ≤ 1; sub-questions ≤ 4; parallel workers ≤ 4.
- Each node records a `TraceStep` including its duration.

**Fallbacks (every agent path must degrade, never crash):**
- Router JSON invalid → intent `factual`.
- Planner invalid or empty → treat as `factual`.
- Any agent exception or limit hit → log, record a trace step `{"agent":"orchestrator","action":"fallback"}`, run the **linear pipeline** (`pipeline._ask_linear`) and return its result.
- `USE_AGENTS=false` → linear pipeline only.

## B9. Acceptance tests for Part B (`tests/test_agents.py`, fakes mode)

1. Router: five scripted questions map to the five intents (fast-path rules, no LLM call made).
2. Router returns invalid JSON → falls back to `factual`.
3. Factual workflow: trace order is `router → retrieval → generate → verifier → finalize`.
4. Low verifier support → exactly one extra retrieval; trace shows it; second failure yields `Low` confidence.
5. Comparison workflow: planner yields 3 sub-questions → 3 parallel retrievals → synthesizer; final citations map back to original chunks; caps respected for a planner returning 8 sub-questions (truncated to 4).
6. Analyst workflow: `calculate("21 - 12")` → 9 appears in the answer evidence as `[E_calc]`.
7. Safe calculator rejects `__import__('os')`, attribute access, and names; accepts valid arithmetic.
8. Any agent raising an exception → result equals the linear pipeline's result and trace includes the fallback step.
9. `AskResult` from agent mode validates against the same schema as linear mode, with `agent_trace` populated.
10. Ingestion graph: a mixed fixture PDF yields the expected per-page routing (text / scanned / chart) and progress events.
11. `MCP_MODE=stdio` and `direct` return identical results for `hybrid_search` on the tiny corpus.

## B10. Evaluation of the agent upgrade

The QA Agent runs the gold set in `linear` and `agents` modes and reports per question type: correctness, faithfulness, latency (median, p95), LLM calls per question. **If agent mode does not beat linear on at least comparison and chart/table questions, say so in the README and demo with linear as default.** Report honestly; the point is to show the comparison, not to force a win.

---

# Definition of Done (multi-agent additions)

- [ ] Phase 0 contracts frozen and merged; `CONTRACTS.md` complete
- [ ] Each build agent worked only in its owned files (check with `git log --stat` per branch)
- [ ] Checkpoints 1, 2, 3 passed in order
- [ ] Router + factual, table/chart, comparison, summarize workflows implemented
- [ ] Ingestion triage routes pages and reports progress
- [ ] MCP servers run (`stdio`) and `direct` mode matches
- [ ] Safe calculator with tests
- [ ] Limits, timeouts and linear fallback verified by tests
- [ ] Trace panel and mode toggle in the UI
- [ ] Eval report comparing linear vs agents
- [ ] `v1.0-demo` tag cut on 12 Oct

---

# Kickoff Prompts

**Lead (run first, alone):**
```
Read DocuRAG_AI_Build_Spec.md and DocuRAG_Multi_Agent_Spec.md. You are the Lead/Integrator (Agent 0).
Execute Phase 0 (A3) only: scaffold, schemas with TraceStep v2, interface stubs, fakes (FakeEmbedder, FakeLLM,
FakeReranker), fixture factory, CONTRACTS.md, branches. Do not implement real ingestion, retrieval, generation,
backend, frontend, or agents. Run the Phase 0 verify command and show me the output. Then update PROGRESS.md.
```

**Worker agents (one session each, after Phase 0 is merged):**
```
Read DocuRAG_AI_Build_Spec.md, DocuRAG_Multi_Agent_Spec.md and CONTRACTS.md. You are Agent <N>: <ROLE>.
Work on branch agent/<name>. Edit ONLY the files listed for your role in section A2. Code against the stubs and
fakes (DOCURAG_FAKES=1) until real dependencies land on main. Contract changes go in CONTRACT_CHANGES.md, never
directly. Implement your brief in section A5 (and Part B if you are the Runtime-Agents Agent). Run your verify
commands, update only your section of PROGRESS.md, rebase on main, and open a PR. Stop and report if you are blocked.
```

**Single-agent fallback (if you only have one AI session):**
```
Read both specs. Play the roles in the order 0 → 1 → 2 → 3 → 5 → 6 → 4 → 7, committing per role and respecting
file ownership. Pass Checkpoints 1 and 2 before starting Part B. Show verify output after each role.
```
