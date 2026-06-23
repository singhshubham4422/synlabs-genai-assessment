import os
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
TOP_K = int(os.getenv("TOP_K", "5"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_docs")

# Model identifier as per constraints (gemini-2.5-flash is the SDK identifier for Gemini 3.5 Flash)
GEMINI_MODEL_NAME = "gemini-2.5-flash"
