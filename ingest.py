"""
ingest.py — Ingestion Pipeline

This script reads all PDFs from the dataset/ folder, splits them into
chunks, generates embeddings, and stores everything in a ChromaDB
vector database for later retrieval.

Run this once (or whenever you add new PDFs):
    python ingest.py
"""

import sys
import time
import shutil
import chromadb

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import (
    DATASET_DIR,
    CHROMA_DB_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBED_MODEL_NAME,
    CHROMA_COLLECTION_NAME,
)


def ingest_documents():
    """
    Full ingestion pipeline:
    1. Load PDFs from dataset/
    2. Split into chunks with sentence-aware boundaries
    3. Generate embeddings using HuggingFace model
    4. Store in ChromaDB (persisted to disk)
    """

    print("=" * 60)
    print("  RAG Document Ingestion Pipeline")
    print("=" * 60)

    # ── Step 0: Clean previous index if it exists ─────────────────
    if CHROMA_DB_DIR.exists():
        print(f"\n🗑️  Removing old vector database at: {CHROMA_DB_DIR}")
        shutil.rmtree(CHROMA_DB_DIR)

    # ── Step 1: Load PDFs ─────────────────────────────────────────
    print(f"\n📂 Loading documents from: {DATASET_DIR}")
    start = time.time()

    if not DATASET_DIR.exists():
        print(f"❌ Dataset directory not found: {DATASET_DIR}")
        sys.exit(1)

    reader = SimpleDirectoryReader(
        input_dir=str(DATASET_DIR),
        recursive=True,
        required_exts=[".pdf"],
    )
    documents = reader.load_data()

    print(f"   ✅ Loaded {len(documents)} pages from PDFs")
    elapsed = time.time() - start
    print(f"   ⏱️  Time: {elapsed:.1f}s")

    # Show which files were loaded
    file_names = set()
    for doc in documents:
        fname = doc.metadata.get("file_name", "Unknown")
        file_names.add(fname)
    print(f"\n📄 Files loaded ({len(file_names)}):")
    for name in sorted(file_names):
        print(f"   • {name}")

    # ── Step 2: Initialize embedding model ────────────────────────
    print(f"\n🧠 Loading embedding model: {EMBED_MODEL_NAME}")
    start = time.time()
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    elapsed = time.time() - start
    print(f"   ✅ Model loaded in {elapsed:.1f}s")

    # ── Step 3: Set up ChromaDB vector store ──────────────────────
    print(f"\n💾 Setting up ChromaDB at: {CHROMA_DB_DIR}")
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    chroma_collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME
    )
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # ── Step 4: Chunk, embed, and index ───────────────────────────
    print(f"\n✂️  Chunking documents (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print("📊 Creating embeddings and building index... (this may take a few minutes)")
    start = time.time()

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter],
        show_progress=True,
    )

    elapsed = time.time() - start
    print(f"\n   ✅ Index built in {elapsed:.1f}s")
    print(f"   📦 Total chunks stored: {chroma_collection.count()}")

    # ── Done ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅ Ingestion complete!")
    print(f"  Vector database saved to: {CHROMA_DB_DIR}")
    print("  You can now run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    ingest_documents()
