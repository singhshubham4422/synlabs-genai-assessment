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

---

## Document Corpus

Three Markdown files used as the evaluation corpus:

| File | Topic Coverage |
|---|---|
| `docs/llm_fundamentals.md` | Transformers, attention, tokenization, embeddings, RLHF |
| `docs/rag_architecture.md` | Chunking strategies, vector stores, retrieval metrics, grounding, hallucination, HyDE, reranking |
| `docs/vector_db_comparison.md` | FAISS, ChromaDB, Pinecone, Weaviate, pgvector, cost, latency, ANN algorithms |

---

## Evaluation Results

Evaluated over a fixed set of 20 questions covering factual lookup, multi-hop reasoning, out-of-scope queries, and ambiguous queries.

### Retrieval Metrics

| Metric | Score | Interpretation |
|---|---|---|
| **Hit Rate @ k** | **0.90** | 90% of queries retrieved at least one relevant chunk |
| **Recall @ k** | **0.8575** | 85.75% of all relevant chunks were retrieved |
| **MRR** | **0.85** | First relevant chunk appeared at rank 1.18 on average |
| **nDCG @ k** | **0.8438** | Strong ranking quality; relevant chunks near the top |

### Answer Quality Metrics

| Metric | Score | Interpretation |
|---|---|---|
| **Faithfulness** | **0.955** | LLM almost never added facts beyond retrieved chunks |
| **Answer Relevance** | **0.955** | Answers directly addressed the question asked |
| **Exact Match (EM)** | 0.10 | Expected low — generative answers rarely match gold strings |
| **F1 Score** | 0.2092 | Token overlap low by design; generative != extractive |

> **Note on EM/F1:** These metrics are designed for extractive QA (e.g. SQuAD). For generative RAG, low EM/F1 is structurally expected and does not indicate poor quality. The faithfulness and relevance scores (both 0.955) are the meaningful quality signals here.

### Latency

| Metric | Value |
|---|---|
| **p50 (median)** | **2788.5 ms** |
| **p95** | **3312.9 ms** |

Latency breakdown: ~50ms local embedding + ~2700ms Gemini API round trip. Acceptable for a QA service; reducible with async batching or caching repeated queries.

---

## Cost Comparison

### Assumptions
- Vector dimensionality: 384 (float32 = 4 bytes each)
- Storage per vector: ~1.7 KB (vector + metadata)
- Query volume: 50,000 queries/month
- ChromaDB: self-hosted on a $10/mo VPS (2 vCPU, 4 GB RAM) — holds up to ~2M vectors
- Pinecone: s1.x1 pod at $0.096/hr, each pod holds ~5M vectors
- Weaviate: Standard cloud tier, public pricing

| Scale | ChromaDB (self-hosted) | Pinecone (s1.x1) | Weaviate Cloud | Notes |
|---|---|---|---|---|
| **100K vectors** | **$10/mo** | $70/mo | $25/mo | Chroma runs on basic VPS; Pinecone requires 1 pod minimum |
| **1M vectors** | **$10/mo** | $70/mo | $25/mo | Chroma fits in memory on 4 GB VPS; Pinecone 1 pod |
| **10M vectors** | **$20/mo** | $140/mo | $185/mo | Chroma upgraded to 8 GB VPS; Pinecone needs 2 pods |

ChromaDB is **7x cheaper** than Pinecone at 1M vectors and **7–9x cheaper** at 10M.

### Trade-offs Accepted with ChromaDB
- No built-in replication or high availability
- No multi-region support
- Ops burden falls on your team (backups, scaling)
- No managed SLA

---

## Design Decisions

**Why ChromaDB over FAISS?** ChromaDB has built-in metadata filtering, persistence without a server process, and a clean Python API. FAISS requires manual index serialization and has no native metadata support.

**SHA256 chunk IDs** enable upsert semantics — re-ingesting the same file is a no-op, not a duplicate insertion.

**No-hallucination guard** skips the LLM entirely when retrieval confidence is low (distance > 0.85). The system admits uncertainty rather than fabricating an answer.

**LLM-as-judge** for faithfulness and relevance uses a separate Gemini call with a strict prompt returning only a float 0–1, minimizing token usage.

---

## Discussion

**When would you switch back to a managed vector DB?**
At >5M vectors with multi-region requirements, uptime SLAs, or a team without ops bandwidth, the $60–100/mo premium for Pinecone or Weaviate becomes worth it. The break-even point is roughly when the engineering time to maintain the self-hosted setup costs more than the managed service premium.

**Was retrieval or generation the weak link?**
Retrieval was slightly weaker (Hit Rate 0.90, MRR 0.85) while generation was near-perfect (Faithfulness 0.955). The system occasionally failed to retrieve the right chunk for multi-hop questions requiring context from two documents simultaneously. A reranker or HyDE would address this.

**What single change would most improve answer quality?**
Adding a reranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) between retrieval and generation. The bi-encoder embedding model used here optimizes for recall, not precision. A cross-encoder reranker would improve the quality of the top-3 chunks passed to the LLM.
