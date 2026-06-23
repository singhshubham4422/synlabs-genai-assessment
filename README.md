# SynLabs GenAI Assessment


> A two-part applied AI engineering assessment covering cost-efficient RAG infrastructure and LLM-as-Judge evaluation pipelines — built end-to-end with Gemini 3.5 Flash on the Google Antigravity agent harness.

---

## Repository Structure

```
synlabs-genai-assessment/
├── problem1/       Cost-Efficient RAG Application
└── problem2/       LLM-as-Judge Evaluation Pipeline
```

---

---

# Problem 1 — Cost-Efficient RAG Application

## Overview

A production-ready Retrieval-Augmented Generation (RAG) service backed by a self-hosted ChromaDB vector store, evaluated against managed alternatives at scale. The system ingests PDF, HTML, and Markdown documents, embeds them locally, and serves grounded answers via a FastAPI endpoint — with full retrieval and answer quality metrics.

The core thesis: a self-hosted embedded vector store is a credible, measurable alternative to managed vector DBs for most small-to-medium workloads, at a fraction of the cost.

---

## Tech Stack

| Component | Choice | Justification |
|---|---|---|
| Vector Store | ChromaDB (embedded) | Zero infra cost, pip-installable, metadata filtering, idempotent upsert via SHA256 IDs |
| Embeddings | `all-MiniLM-L6-v2` | Free, 384-dim, runs locally via sentence-transformers, no API cost |
| LLM | Gemini 3.5 Flash | Fast, cost-efficient, agent-optimized via Antigravity harness |
| HTTP Framework | FastAPI + Uvicorn | Async, lightweight, production-ready |
| PDF Parsing | pypdf | Lightweight, no external dependencies |
| HTML Parsing | BeautifulSoup4 | Standard, robust |
| Config | python-dotenv | All secrets via `.env`, nothing hardcoded |

---

## Setup & Quick Start

```bash
cd problem1
cp .env.example .env          # Add your GEMINI_API_KEY
pip install -r requirements.txt
python ingest.py --path ./docs
uvicorn rag_service:app --host 0.0.0.0 --port 8000
```

### Query the Service

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RLHF?", "top_k": 5}'
```

### Run Evaluation

```bash
python eval_harness.py --service-url http://localhost:8000
python cost_comparison.py
```

---

## Configuration

All configuration is via environment variables. No hardcoded values anywhere.

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | Required. Gemini API key |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `TOP_K` | `5` | Number of chunks to retrieve per query |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model name |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Local path for ChromaDB persistence |
| `COLLECTION_NAME` | `rag_docs` | ChromaDB collection name |

---

## Ingestion Pipeline

### Supported Formats
- `.pdf` — parsed via pypdf, page by page
- `.html` / `.htm` — parsed via BeautifulSoup4, text extracted
- `.md` — read directly as plain text

### Chunking Strategy
Sliding window by character count. Default: 512 chars per chunk, 64-char overlap. Configurable via `.env`.

### Idempotent Re-ingest
Each chunk is assigned an ID = `sha256(chunk_text)`. ChromaDB upserts on this ID. Re-running ingest on the same corpus never creates duplicate vectors.

### Metadata Stored Per Chunk
```json
{
  "source": "filename.pdf",
  "chunk_index": 3,
  "file_type": "pdf"
}
```
Metadata filtering is supported on the `/query` endpoint via the `filter` parameter.

---

## API Reference

### `POST /query`

**Request:**
```json
{
  "question": "What is attention in transformers?",
  "top_k": 5,
  "filter": { "source": "llm_fundamentals.md" }
}
```

**Response:**
```json
{
  "answer": "Attention is a mechanism that allows the model to... [Source: llm_fundamentals.md, chunk 3]",
  "chunks_used": [
    { "text": "...", "source": "llm_fundamentals.md", "chunk_index": 3, "score": 0.21 }
  ],
  "latency_ms": 2788.5,
  "tokens_used": { "input": 412, "output": 138 },
  "chunk_count": 5
}
```

**No-hallucination guard:** If all retrieved chunk distances exceed 0.85, the LLM is never called. The response returns `"No relevant context found."` immediately.

### `GET /health`
```json
{ "status": "ok" }
```

### Per-Query Logging (stdout)
```
2026-06-23T10:42:11 | What is RLHF? | 2788ms | chunks=5 | in=412 | out=138
```
