# Retrieval-Augmented Generation (RAG) Architecture

Retrieval-Augmented Generation (RAG) is a design pattern used to enrich LLM prompts with relevant, verified external knowledge retrieved from a corpus. This solves the knowledge cutoff problem and mitigates hallucinations by grounding LLM responses in factual content. A typical RAG pipeline consists of ingestion, retrieval, and synthesis.

## Chunking Strategies

A crucial step in ingestion is chunking—splitting long source files into smaller, coherent text segments. If chunks are too large, they dilute specific information and risk exceeding LLM context windows. If chunks are too small, they lose critical context.

Common chunking strategies include:
- **Character/Word-based Chunking**: Splitting text into fixed lengths (e.g., 512 characters) with an overlap (e.g., 64 characters) to ensure text spanning across boundaries is captured in both chunks.
- **Sentence-based Chunking**: Splitting text by sentences using punctuation markers, preserving semantic units better than arbitrary character counts.
- **Semantic Chunking**: Analyzing embedding differences between successive sentences and splitting when a significant shift in topic is detected.

The sliding window approach (fixed-size chunks with overlap) is the most standard, cost-effective baseline. The overlap guarantees that key concepts on boundaries are not cut in half, maintaining structural integrity across chunks.

## Vector Databases and Retrieval Metrics

Once chunked, each segment is embedded into a vector space and loaded into a vector database. During query execution, the user's question is embedded using the same encoder, and a nearest-neighbor search (using metrics like Cosine Similarity or Euclidean Distance) retrieves the top $k$ chunks.

To evaluate retrieval quality, several metrics are calculated over test datasets:
- **Hit Rate @ k**: The fraction of queries where at least one retrieved chunk contains the correct information.
- **Recall @ k**: The ratio of retrieved relevant chunks to the total number of relevant chunks in the database for a query.
- **Mean Reciprocal Rank (MRR)**: Evaluates the position of the first relevant chunk in the results. If the first relevant chunk is at rank $r$, the score is $1/r$.
- **Normalized Discounted Cumulative Gain (nDCG @ k)**: Evaluates ranking quality, giving higher weight to highly relevant chunks placed at the top of the search results.

## Advanced RAG: Grounding, HyDE, and Reranking

RAG helps ground responses by providing direct quotes or reference points to the LLM. "Grounding" is the process of ensuring that every claim in the LLM's generated response is directly backed by the retrieved context. Hallucination occurs when the model introduces external facts or contradicts the provided context. To counter this, strict system prompts enforce that if the answer is missing from the context, the model must explicitly state that the answer is not available.

Advanced architectures improve retrieval using:
1. **Hypothetical Document Embeddings (HyDE)**: The user query is sent to an LLM to generate a fake "hypothetical" answer. This hypothetical answer is embedded and used to query the vector store. This is effective because search queries are often brief and differ from document syntax, whereas the hypothetical answer shares the same syntax and vocabulary as the corpus.
2. **Reranking**: Bi-encoders (like standard sentence-transformers) are fast but lose token-to-token comparison detail. After retrieving the top 20 or 50 chunks via a bi-encoder, a more powerful Cross-Encoder (Reranker) scores the query against each chunk. The cross-encoder processes the query and document together, producing highly accurate similarity scores, though at a higher computational latency.
