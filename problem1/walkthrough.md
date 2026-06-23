# Walkthrough — Cost-Efficient RAG System

A complete production-ready RAG system has been built, tested, and validated.

## Deliverables Built

1. [`.env.example`](file:///C:/Users/sshub/Videos/Assesment/Problem_1/.env.example): Template environment config with embedding model (`all-MiniLM-L6-v2`), database persistent location (`./chroma_db`), and default threshold values.
2. [`requirements.txt`](file:///C:/Users/sshub/Videos/Assesment/Problem_1/requirements.txt): List of exact Python dependencies.
3. [`config.py`](file:///C:/Users/sshub/Videos/Assesment/Problem_1/config.py): Exposes config options dynamically from env variables.
4. [`docs/`](file:///C:/Users/sshub/Videos/Assesment/Problem_1/docs): Fixture corpus containing:
   - [`llm_fundamentals.md`](file:///C:/Users/sshub/Videos/Assesment/Problem_1/docs/llm_fundamentals.md)
   - [`rag_architecture.md`](file:///C:/Users/sshub/Videos/Assesment/Problem_1/docs/rag_architecture.md)
   - [`vector_db_comparison.md`](file:///C:/Users/sshub/Videos/Assesment/Problem_1/docs/vector_db_comparison.md)
5. [`ingest.py`](file:///C:/Users/sshub/Videos/Assesment/Problem_1/ingest.py): Chunking (sliding window) & embedding loader with SHA256 duplicate guards and cosine space configuration.
6. [`rag_service.py`](file:///C:/Users/sshub/Videos/Assesment/Problem_1/rag_service.py): FastAPI app running queries, implementing threshold checks (<= 0.85), refusal on no relevant context, and API key safety fallbacks.
7. [`eval_questions.json`](file:///C:/Users/sshub/Videos/Assesment/Problem_1/eval_questions.json): 20 robust evaluation questions spanning multiple categories (factual, multi-hop, ambiguous, out-of-scope).
8. [`eval_harness.py`](file:///C:/Users/sshub/Videos/Assesment/Problem_1/eval_harness.py): Evaluation harness checking Hit Rate, Recall, MRR, nDCG, Exact Match, Token F1, and LLM-as-judge scores.
9. [`cost_comparison.py`](file:///C:/Users/sshub/Videos/Assesment/Problem_1/cost_comparison.py): Computes VPS vs managed database costs and generates the markdown report.
10. [`README.md`](file:///C:/Users/sshub/Videos/Assesment/Problem_1/README.md): Setup, configuration, architecture decision log, and final discussion.

---

## Evaluation Harness Run Results

The RAG pipeline was evaluated against all 20 questions in the test dataset. Here are the aggregate results:

```json
{
  "hit_rate_at_k": 0.9,
  "recall_at_k": 0.8575,
  "mrr": 0.85,
  "ndcg_at_k": 0.8438,
  "faithfulness": 0.9550,
  "answer_relevance": 0.9550,
  "em": 0.1,
  "f1": 0.2092,
  "p50_latency_ms": 2788.5,
  "p95_latency_ms": 3312.9
}
```

---

## Cost Comparison Results

The cost analysis of self-hosting ChromaDB on a VPS vs managed alternatives Pinecone and Weaviate was calculated and exported to [`cost_comparison.md`](file:///C:/Users/sshub/Videos/Assesment/Problem_1/cost_comparison.md):

| Scale | ChromaDB (self-hosted) | Pinecone (s1.x1) | Weaviate Cloud | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **100K vectors** | $10/mo | $70/mo | $25/mo | Chroma runs on basic VPS; Pinecone requires 1 pod minimum; Weaviate standard base tier. |
| **1M vectors** | $10/mo | $70/mo | $25/mo | Chroma easily fits in memory on 4GB VPS; Pinecone 1 pod; Weaviate standard base tier. |
| **10M vectors** | $20/mo (upgrade VPS) | $140/mo (2 pods) | $185/mo | Chroma upgraded to 8GB VPS ($20/mo); Pinecone requires 2 pods; Weaviate standard tier scales up. |
