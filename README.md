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
