import streamlit as st
import sys
import os

# Ensure the src directory is in the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rag_engine import configure_settings, get_index, RAGConfig

st.set_page_config(page_title="RAG Chat Assistant", page_icon="🤖", layout="wide")

st.title("🤖 RAG Chat Assistant")
st.markdown("Ask questions based on your documents! The assistant remembers the context of the conversation.")

@st.cache_resource(show_spinner=False)
def init_rag():
    with st.spinner("Initializing RAG Engine..."):
        cfg = RAGConfig()
        configure_settings(cfg)
        try:
            index = get_index(cfg)
            return index, cfg
        except Exception as e:
            st.error(f"Failed to load index: {e}")
            st.stop()

index, cfg = init_rag()

SYSTEM_PROMPT = """You are a strict assistant. You must answer questions based ONLY on the provided document context.

Rules:
1. Answer the question using ONLY the information from the retrieved context.
2. If the context does not contain the answer, say: "I'm sorry, I don't have enough information in the loaded documents to answer this question."
3. Do NOT make up or hallucinate information.
4. NEVER use your outside general knowledge to answer questions.
"""

if "chat_engine" not in st.session_state:
    try:
        # condense_plus_context is excellent for maintaining context over multiple turns
        st.session_state.chat_engine = index.as_chat_engine(
            chat_mode="condense_plus_context", 
            system_prompt=SYSTEM_PROMPT,
            similarity_top_k=cfg.TOP_K,
            verbose=True
        )
    except ValueError:
        # Fallback to condense_question if condense_plus_context is not supported in this version
        st.session_state.chat_engine = index.as_chat_engine(
            chat_mode="condense_question",
            system_prompt=SYSTEM_PROMPT,
            verbose=True
        )

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("⚙️ Chat Settings")
    st.markdown("Context is saved automatically for follow-up questions.")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_engine.reset()
        st.rerun()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is your question?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.chat_engine.chat(prompt)
                st.markdown(response.response)
                
                # Show sources
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    with st.expander("📄 Sources Context Used"):
                        for node in response.source_nodes:
                            filename = node.metadata.get('file_name', 'unknown')
                            score = round(node.score or 0.0, 4)
                            st.markdown(f"- **{filename}** (Relevance Score: {score})")
                            st.caption(node.text[:300] + "...")
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response.response})
            except Exception as e:
                st.error(f"An error occurred during chat: {e}")
