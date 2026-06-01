"""
query_engine.py — Query Engine Module

Provides a reusable function to load the persisted ChromaDB index
and create a LlamaIndex query engine backed by the Ollama LLM.

Used by app.py (Streamlit UI) or can be run standalone.
"""

import sys
import chromadb

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import (
    CHROMA_DB_DIR,
    EMBED_MODEL_NAME,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    SIMILARITY_TOP_K,
    SIMILARITY_CUTOFF,
    CHROMA_COLLECTION_NAME,
    OLLAMA_BASE_URL,
    REQUEST_TIMEOUT,
)


# System prompt instructs the LLM to answer based on retrieved context only
SYSTEM_PROMPT = """You are a helpful study assistant that answers questions based ONLY on the provided document context.

Rules:
1. Answer the question using ONLY the information from the retrieved document chunks.
2. If the context does not contain enough information to answer, say: "I don't have enough information in the loaded documents to answer this question."
3. Be concise but thorough. Use bullet points or numbered lists when appropriate.
4. Always mention which document/source the information comes from when possible.
5. Do NOT make up or hallucinate information that is not in the context.
"""


def get_query_engine():
    """
    Load the persisted ChromaDB vector store and return a
    configured LlamaIndex query engine with Ollama LLM.

    Returns:
        query_engine: A LlamaIndex query engine ready to answer questions.

    Raises:
        SystemExit: If the vector database doesn't exist (run ingest.py first).
    """

    # ── Validate prerequisites ────────────────────────────────────
    if not CHROMA_DB_DIR.exists():
        print("❌ Vector database not found!")
        print("   Run 'python ingest.py' first to index your documents.")
        sys.exit(1)

    # ── Configure Ollama LLM ──────────────────────────────────────
    llm = Ollama(
        model=LLM_MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        request_timeout=REQUEST_TIMEOUT,
        temperature=LLM_TEMPERATURE,
    )
    Settings.llm = llm

    # ── Load embedding model ──────────────────────────────────────
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.embed_model = embed_model

    # ── Connect to persisted ChromaDB ─────────────────────────────
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    chroma_collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME
    )
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # ── Rebuild index from vector store ───────────────────────────
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    # ── Create retriever with custom settings ─────────────────────
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=SIMILARITY_TOP_K,
    )

    # ── Postprocessor to filter low-relevance chunks ──────────────
    postprocessor = SimilarityPostprocessor(
        similarity_cutoff=SIMILARITY_CUTOFF,
    )

    # ── Create and return the query engine ────────────────────────
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[postprocessor],
    )

    return query_engine


def get_index_stats():
    """
    Return basic stats about the persisted vector database.

    Returns:
        dict with keys: 'exists', 'chunk_count', 'collection_name'
    """
    if not CHROMA_DB_DIR.exists():
        return {"exists": False, "chunk_count": 0, "collection_name": None}

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    chroma_collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME
    )

    return {
        "exists": True,
        "chunk_count": chroma_collection.count(),
        "collection_name": CHROMA_COLLECTION_NAME,
    }


if __name__ == "__main__":
    print("Loading query engine...")
    engine = get_query_engine()
    stats = get_index_stats()
    print(f"✅ Ready! Index has {stats['chunk_count']} chunks.\n")

    while True:
        try:
            question = input("You: ").strip()
            if question.lower() in ["exit", "quit"]:
                print("Bye!")
                break
            if not question:
                continue

            print("Thinking...")
            response = engine.query(question)
            print(f"\n[Answer]\n{response}\n")

            if response.source_nodes:
                print("--- Sources ---")
                for node in response.source_nodes:
                    fname = node.metadata.get("file_name", "unknown")
                    score = round(node.score or 0.0, 4)
                    print(f"  - {fname} (score: {score})")
            print()

        except KeyboardInterrupt:
            print("\nBye!")
            break
