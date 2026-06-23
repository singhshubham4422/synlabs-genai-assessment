# synlabs-genai-assessment

## Structure
  problem1/   Cost-Efficient RAG Application (ChromaDB + Gemini 3.5 Flash)
  problem2/   Coming soon

## Problem 1 — Quick Start
  cd problem1
  cp .env.example .env      # add your GEMINI_API_KEY
  pip install -r requirements.txt
  python ingest.py --path ./docs
  uvicorn rag_service:app --host 0.0.0.0 --port 8000
