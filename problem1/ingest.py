import os
import sys
import argparse
import hashlib
import chromadb
from bs4 import BeautifulSoup
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Import config constants
import config

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split text into fixed-size character chunks with overlap using sliding window.
    """
    chunks = []
    text_len = len(text)
    if text_len == 0:
        return chunks

    # Safeguards on bounds
    if chunk_size <= 0:
        chunk_size = 512
    if chunk_overlap < 0:
        chunk_overlap = 0
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 2

    start = 0
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end >= text_len:
            break
        start += chunk_size - chunk_overlap

    return chunks

def main():
    parser = argparse.ArgumentParser(description="Ingest PDF, HTML, and MD documents into ChromaDB.")
    parser.add_argument("--path", type=str, default="./docs", help="Path to a file or directory of files to ingest.")
    args = parser.parse_args()

    target_path = args.path
    if not os.path.exists(target_path):
        print(f"Error: Path '{target_path}' does not exist.")
        sys.exit(1)

    # Gather files
    files_to_process = []
    if os.path.isdir(target_path):
        for root, _, files in os.walk(target_path):
            for file in files:
                files_to_process.append(os.path.join(root, file))
    else:
        files_to_process.append(target_path)

    supported_extensions = {".pdf", ".html", ".htm", ".md"}
    valid_files = []
    for f in files_to_process:
        ext = os.path.splitext(f)[1].lower()
        if ext in supported_extensions:
            valid_files.append((f, ext))
        else:
            print(f"Warning: Skipping unsupported file extension: '{f}'")

    if not valid_files:
        print("No supported files found to process.")
        sys.exit(0)

    # Initialize ChromaDB Persistent Client
    print(f"Initializing ChromaDB persistent client at '{config.CHROMA_PERSIST_DIR}'...")
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    collection = chroma_client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # Initialize Embeddings model
    print(f"Loading embedding model '{config.EMBED_MODEL}'...")
    embed_model = SentenceTransformer(config.EMBED_MODEL)

    total_files_processed = 0
    total_chunks_upserted = 0
    total_duplicates_skipped = 0

    for file_path, ext in valid_files:
        filename = os.path.basename(file_path)
        print(f"Processing '{filename}'...")
        try:
            text = ""
            if ext == ".md":
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            elif ext in (".html", ".htm"):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                soup = BeautifulSoup(content, "html.parser")
                text = soup.get_text(separator=" ")
            elif ext == ".pdf":
                reader = PdfReader(file_path)
                pages_text = []
                for page in reader.pages:
                    p_text = page.extract_text()
                    if p_text:
                        pages_text.append(p_text)
                text = "\n".join(pages_text)

            if not text.strip():
                print(f"Warning: Extracted text from '{filename}' is empty. Skipping.")
                continue

            # Split into chunks
            chunks = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            if not chunks:
                print(f"Warning: No valid chunks generated for '{filename}'. Skipping.")
                continue

            file_type = "html" if ext in (".html", ".htm") else ext.lstrip(".")

            # Filter duplicates within the file and map items
            unique_chunks_dict = {}
            for i, chunk in enumerate(chunks):
                cid = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                if cid not in unique_chunks_dict:
                    unique_chunks_dict[cid] = {
                        "text": chunk,
                        "metadata": {
                            "source": filename,
                            "chunk_index": i,
                            "file_type": file_type
                        }
                    }

            ids_to_check = list(unique_chunks_dict.keys())
            existing_ids = set()
            if ids_to_check:
                try:
                    existing = collection.get(ids=ids_to_check)
                    existing_ids = set(existing.get("ids", []))
                except Exception:
                    pass

            dup_count = sum(1 for cid in ids_to_check if cid in existing_ids)
            
            # Prepare batch for Chroma DB
            ids_to_upsert = []
            documents_to_upsert = []
            metadatas_to_upsert = []

            for cid, item in unique_chunks_dict.items():
                ids_to_upsert.append(cid)
                documents_to_upsert.append(item["text"])
                metadatas_to_upsert.append(item["metadata"])

            if ids_to_upsert:
                # Compute embeddings using sentence-transformers
                embeddings_to_upsert = embed_model.encode(documents_to_upsert).tolist()
                
                # Perform idempotent upsert
                collection.upsert(
                    ids=ids_to_upsert,
                    embeddings=embeddings_to_upsert,
                    metadatas=metadatas_to_upsert,
                    documents=documents_to_upsert
                )

            total_files_processed += 1
            total_chunks_upserted += len(ids_to_upsert)
            total_duplicates_skipped += dup_count
            print(f"Successfully processed '{filename}' ({len(ids_to_upsert)} chunks upserted, {dup_count} duplicates detected).")

        except Exception as e:
            print(f"Error processing file '{file_path}': {e}", file=sys.stderr)

    print("\n=== Ingestion Summary ===")
    print(f"Total files processed: {total_files_processed}")
    print(f"Total chunks upserted: {total_chunks_upserted}")
    print(f"Duplicates skipped:    {total_duplicates_skipped}")

if __name__ == "__main__":
    main()
