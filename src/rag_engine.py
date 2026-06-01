"""
RAG Engine — LlamaIndex + Ollama
Handles: loading documents, building index, querying
"""

import os
import sys
from pathlib import Path

# Add project root to path so we can import config
sys.path.insert(0, str(Path(__file__).parent.parent))


from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    Document
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

import config as cfg_file

# ──────────────
# Configuration
# ──────────────
class RAGConfig:
    DATA_DIR        = "./dataset"        # folder with your PDFs / TXT / DOCX

    # Pull defaults from config.py so you only change the model in one place
    LLM_MODEL       = cfg_file.LLM_MODEL_NAME          
    EMBED_MODEL     = cfg_file.EMBED_MODEL_NAME

    CHUNK_SIZE      = cfg_file.CHUNK_SIZE
    CHUNK_OVERLAP   = cfg_file.CHUNK_OVERLAP
    TOP_K           = cfg_file.SIMILARITY_TOP_K
    TEMPERATURE     = cfg_file.LLM_TEMPERATURE
    REQUEST_TIMEOUT = cfg_file.REQUEST_TIMEOUT
    CHROMA_DB_DIR   = cfg_file.CHROMA_DB_DIR
    CHROMA_COLLECTION = cfg_file.CHROMA_COLLECTION_NAME


def configure_settings(cfg: RAGConfig = RAGConfig()):
    """
    Global settings for LlamaIndex: what LLM and Embedder to use.
    """
    Settings.llm = Ollama(
        model=cfg.LLM_MODEL,
        request_timeout=cfg.REQUEST_TIMEOUT,
        temperature=cfg.TEMPERATURE,
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=cfg.EMBED_MODEL,
    )
    Settings.node_parser = SentenceSplitter(
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP,
    )

def get_index(cfg: RAGConfig = RAGConfig()) -> VectorStoreIndex:
    """
    Load the index from the ChromaDB vector store built by ingest.py.
    """
    chroma_dir = Path(cfg.CHROMA_DB_DIR)

    if chroma_dir.exists():
        print(f"[RAG] Loading existing ChromaDB index from '{cfg.CHROMA_DB_DIR}' ...")
        chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
        chroma_collection = chroma_client.get_or_create_collection(
            name=cfg.CHROMA_COLLECTION
        )
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=Settings.embed_model,
        )

    # Build new index
    data_dir = Path(cfg.DATA_DIR)
    if not data_dir.exists() or not any(data_dir.iterdir()):
        raise FileNotFoundError(f"No data found. Please add files to {cfg.DATA_DIR}")

    print(f"[RAG] LLM      : {cfg.LLM_MODEL}")
    print(f"[RAG] Embedder : {cfg.EMBED_MODEL}")
    print(f"[RAG] Chunk    : {cfg.CHUNK_SIZE} tokens  (overlap {cfg.CHUNK_OVERLAP})")
    print(f"[RAG] Reading documents from '{cfg.DATA_DIR}' ...")
    
    documents = []
    for filepath in data_dir.glob("**/*"):
        if filepath.is_file():
            try:
                reader = SimpleDirectoryReader(input_files=[str(filepath)])
                docs = reader.load_data()
                documents.extend(docs)
                print(f"  - Loaded: {filepath.name}")
            except Exception as e:
                print(f"  ! Error loading {filepath.name}: {e}. Skipping.")

    if not documents:
        raise ValueError("No valid documents could be loaded.")

    print(f"[RAG] Building Chroma vector index from {len(documents)} document chunks ...")
    
    chroma_dir.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
    chroma_collection = chroma_client.get_or_create_collection(
        name=cfg.CHROMA_COLLECTION
    )
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=Settings.embed_model
    )
    return index

def query(question: str, index: VectorStoreIndex, cfg: RAGConfig = RAGConfig()) -> dict:
    """
    Retrieve relevant chunks and generate an answer.
    """
    query_engine = index.as_query_engine(
        similarity_top_k=cfg.TOP_K,
        streaming=False,
    )

    response = query_engine.query(question)

    sources = []
    for node in response.source_nodes:
        sources.append({
            "file"  : node.metadata.get("file_name", "unknown"),
            "score" : round(node.score or 0.0, 4),
            "text"  : node.text[:300] + ("…" if len(node.text) > 300 else ""),
        })

    return {
        "answer" : str(response),
        "sources": sources,
    }
