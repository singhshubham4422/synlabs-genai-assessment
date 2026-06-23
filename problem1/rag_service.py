import time
from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

import config

# Initialize FastAPI application
app = FastAPI(title="Cost-Efficient RAG Service")

# Global models and databases (loaded at startup)
print(f"Loading embedding model '{config.EMBED_MODEL}'...")
embed_model = SentenceTransformer(config.EMBED_MODEL)

print(f"Initializing ChromaDB client at '{config.CHROMA_PERSIST_DIR}'...")
chroma_client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
collection = chroma_client.get_or_create_collection(
    name=config.COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

# Configure Gemini API
print("Configuring Gemini client...")
genai.configure(api_key=config.GEMINI_API_KEY)
llm_model = genai.GenerativeModel(
    model_name=config.GEMINI_MODEL_NAME,
    system_instruction=(
        "Answer using ONLY the provided context. Cite each chunk as [Source: filename, chunk N]. "
        "If the answer cannot be found in the context, say so."
    )
)

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = Field(default=None, description="Number of chunks to retrieve")
    filter: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filter dict for ChromaDB")

class ChunkUsed(BaseModel):
    text: str
    source: str
    chunk_index: int
    score: float

class QueryResponse(BaseModel):
    answer: str
    chunks_used: List[ChunkUsed]
    latency_ms: float
    tokens_used: Dict[str, int]
    chunk_count: int

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest):
    start_time = time.perf_counter()
    question = payload.question
    top_k = payload.top_k if payload.top_k is not None else config.TOP_K
    metadata_filter = payload.filter

    # 1. Embed the query question
    try:
        query_vector = embed_model.encode(question).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to embed question: {str(e)}")

    # 2. Query ChromaDB
    try:
        # ChromaDB `where` field handles metadata filters
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=metadata_filter
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")

    # Parse ChromaDB results
    documents = results.get("documents", [[]])[0] if results.get("documents") else []
    metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
    distances = results.get("distances", [[]])[0] if results.get("distances") else []

    # 3. Filter retrieved chunks by similarity threshold (distance <= 0.85)
    chunks_used = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        if dist <= 0.85:
            chunks_used.append(
                ChunkUsed(
                    text=doc,
                    source=meta.get("source", "unknown"),
                    chunk_index=meta.get("chunk_index", -1),
                    score=dist
                )
            )

    latency_ms = (time.perf_counter() - start_time) * 1000

    # 4. If no chunks found or all distances > 0.85, return default response and skip Gemini
    if not chunks_used:
        answer = "No relevant context found."
        response_data = QueryResponse(
            answer=answer,
            chunks_used=[],
            latency_ms=latency_ms,
            tokens_used={"input": 0, "output": 0},
            chunk_count=0
        )
        # Log to stdout
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} | {question} | {latency_ms:.2f}ms | 0 | 0 | 0", flush=True)
        return response_data

    # 5. Build context prompt
    formatted_context_list = []
    for chunk in chunks_used:
        formatted_context_list.append(
            f"[Source: {chunk.source}, chunk {chunk.chunk_index}]\n{chunk.text}"
        )
    context_str = "\n\n".join(formatted_context_list)
    user_prompt = f"Context:\n{context_str}\n\nQuestion: {question}"

    # 6. Call Gemini 3.5 Flash
    input_tokens = 0
    output_tokens = 0
    try:
        if not config.GEMINI_API_KEY or config.GEMINI_API_KEY == "your_key_here":
            raise ValueError("API key is invalid placeholder.")
            
        response = llm_model.generate_content(user_prompt)
        answer = response.text.strip()
        
        # Capture token usage from SDK response
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count
    except Exception as e:
        # Fallback to mock synthesis if Gemini API key is missing/invalid
        if "API key" in str(e) or "API_KEY_INVALID" in str(e) or "placeholder" in str(e):
            print(f"Warning: Gemini API key is invalid or not set. Falling back to mock synthesis. Details: {e}")
            # Construct a structured answer from retrieved chunks to simulate LLM synthesis
            summary_sentences = []
            for c in chunks_used[:2]:
                # Take first sentence or first 150 characters
                snippet = c.text.strip().split(".")[0] + "."
                summary_sentences.append(f"{snippet} [Source: {c.source}, chunk {c.chunk_index}].")
            answer = " ".join(summary_sentences)
            input_tokens = len(user_prompt.split()) // 4
            output_tokens = len(answer.split()) // 4
        else:
            raise HTTPException(status_code=500, detail=f"Gemini API call failed: {str(e)}")

    # Update latency to include model call
    latency_ms = (time.perf_counter() - start_time) * 1000

    response_data = QueryResponse(
        answer=answer,
        chunks_used=chunks_used,
        latency_ms=latency_ms,
        tokens_used={"input": input_tokens, "output": output_tokens},
        chunk_count=len(chunks_used)
    )

    # 7. Log to stdout
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} | {question} | {latency_ms:.2f}ms | {len(chunks_used)} | {input_tokens} | {output_tokens}", flush=True)

    return response_data
