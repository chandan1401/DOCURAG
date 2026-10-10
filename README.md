# DocuRAG

> Production-Grade Multimodal Retrieval-Augmented Generation Engine with Multi-Agent Orchestration.

DocuRAG ingests complex multimodal documents (PDFs, DOCX, TXT, scanned images), extracts structured text and tables, indexes them using hybrid dense (BGE) and sparse (BM25) vector stores, performs Reciprocal Rank Fusion (RRF) and Cross-Encoder reranking, and synthesizes grounded answers with inline citations, confidence scoring, and stage timing.

---

## Architecture Overview

```
[ User / Web UI / CLI ]
           │
           ▼
[ Pipeline Router (`USE_AGENTS` flag) ]
     ├── False ──► [ Linear Pipeline: Ingest ➔ Chunk ➔ Hybrid Retrieve ➔ Rerank ➔ Generate ➔ Verify ]
     └── True  ──► [ Multi-Agent Graph: Router ➔ Planner ➔ Analyst/Retrieval ➔ Synthesizer ➔ Verifier ]
                         │ (Circuit Breaker Fallback)
                         └──► [ Linear Pipeline Fallback ]
```

---

## Quickstart

### 1. Installation
```bash
pip install -r requirements.txt
pip install -r requirements.txt
# For multi-agent runtime (M8):
pip install -r requirements-agents.txt
```

### 2. Environment Configuration
Copy `.env.example` to `.env` and set your Gemini API key:
```bash
cp .env.example .env
```

### 3. Hermetic Offline Testing
Run the offline test suite in milliseconds without GPU or network access:
```powershell
$env:DOCURAG_FAKES="1"; pytest tests/test_scaffold.py -v
```

### 4. CLI Execution
Query DocuRAG directly from the command line:
```powershell
$env:DOCURAG_FAKES="1"; python -m docurag.pipeline "What is DocuRAG?"
```
