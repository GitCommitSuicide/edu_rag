"""
config.py — Central Configuration

All configurable values for the RAG pipeline live here.
Loads sensitive values (API keys) from .env automatically.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# ── Directories ───────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

# Folder containing the source PDFs (your study material)
DATASET_DIR = BASE_DIR / "dataset"

# Folder containing the source data (alias used by rag_engine / app.py)
DATA_DIR = BASE_DIR / "data"

# ChromaDB persistent storage location (used by ingest.py)
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# ── Chunking Settings ─────────────────────────────────────────────
CHUNK_SIZE = 256          # tokens per chunk
CHUNK_OVERLAP = 64        # overlap between adjacent chunks

# ── Embedding Model ──────────────────────────────────────────────
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# ── ChromaDB ──────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "rag_documents"

# ── LLM Settings ──────────────────────────────────────────────────
LLM_MODEL_NAME = "cogito-2.1:671b-cloud"
LLM_TEMPERATURE = 0.1

# ── Retrieval ─────────────────────────────────────────────────────
SIMILARITY_TOP_K = 4
SIMILARITY_CUTOFF = 0.80

# ── Ollama Settings ───────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
REQUEST_TIMEOUT = 120.0

