# Local RAG Chat Assistant

A Retrieval-Augmented Generation (RAG) system built with **LlamaIndex** and **Ollama**. This project allows you to index local documents and interact with them using a strictly context-bound conversational assistant. It provides both a command-line interface and a modern, chat-based web interface via Streamlit.

## 🚀 Features

- **Local & Private:** Powered completely by local open-source models using Ollama and HuggingFace embeddings.
- **Strict Context Enforcement:** The assistant is explicitly programmed to *only* answer using the information provided in the ingested documents. It will safely decline to answer if the information is unavailable, preventing hallucinations or reliance on outside general knowledge.
- **Persistent Storage:** Uses **ChromaDB** to persist vector embeddings for extremely fast startup and retrieval.
- **Multi-turn Chat Context:** The Streamlit web UI maintains conversation history, seamlessly passing previous turns to the LLM for accurate follow-up questions.
- **Source Citations:** View exactly which document chunks were used to generate an answer, alongside their relevance scores.

## 📂 Project Structure

- `dataset/` - Directory to place your input documents (PDFs, TXT, DOCX, etc.).
- `config.py` - Master configuration file containing model names, chunking strategies, and database paths.
- `ingest.py` - Pipeline script that reads the `dataset/`, generates embeddings, and saves the vector index into the `chroma_db/` directory.
- `src/rag_engine.py` - Core LlamaIndex abstraction module handling LlamaIndex settings, index loading, and query operations.
- `streamlit_app.py` - The primary chat-based web UI using Streamlit.
- `app.py` - A basic console-based chat application.
- `query_engine.py` - Standalone module to configure the RetrieverQueryEngine and perform quick console-based ad-hoc testing.

## 🛠️ Usage

### 1. Ingest Documents
Before asking questions, you must ingest your documents to build the ChromaDB index.
Place your files in the `dataset/` folder, make sure Ollama is running, and then run:
```bash
python ingest.py
```

### 2. Run the Streamlit Web UI (Recommended)
Launch the interactive web application which features chat memory, clear controls, and source citations:
```bash
streamlit run streamlit_app.py
```

### 3. Run the Console App
If you prefer a lightweight terminal experience:
```bash
python app.py
```

## ⚙️ Configuration
You can tweak the models, chunk size, top-K retrieval count, temperatures, and database paths inside `config.py`. 
By default, the LLM is managed locally by **Ollama**, ensuring complete data privacy and offline capability.
