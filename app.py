"""
app.py
------
Streamlit front-end for the Chat with PDF app.

Run with:
    streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv
import os

from src.rag_pipeline import process_pdf

# ── Load environment variables (.env file) ─────────────────────────────────────
load_dotenv()


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄",
    layout="centered",
)


# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Chat bubbles */
    .user-bubble {
        background: #1e3a5f;
        border-radius: 12px 12px 2px 12px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #e8eaf6;
        font-size: 15px;
        max-width: 80%;
        margin-left: auto;
    }
    .assistant-bubble {
        background: #1a1d27;
        border: 1px solid #2a2d3e;
        border-radius: 12px 12px 12px 2px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #c8cad8;
        font-size: 15px;
        max-width: 80%;
    }
    .source-badge {
        background: #12151f;
        border: 1px solid #2a2d3e;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 11px;
        color: #6b7280;
        display: inline-block;
        margin: 2px;
    }
    .stat-card {
        background: #1a1d27;
        border: 1px solid #2a2d3e;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ───────────────────────────────────────────────
def init_session():
    defaults = {
        "qa_chain": None,
        "chat_history": [],     # List of {"role": ..., "content": ..., "sources": ...}
        "pdf_processed": False,
        "chunk_count": 0,
        "pdf_name": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()

# -- Read API key from Streamlit secrets or local .env ----------
# No input field shown — keeps key secure in production
api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key:
    st.sidebar.error("⚠️ OpenAI API key not configured. Add it to Streamlit secrets.")
    st.stop()
    
# ── Sidebar: API key + PDF upload ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.divider()
    st.markdown("## 📄 Upload PDF")

    uploaded_file = st.file_uploader(
        label="Choose a PDF file",
        type="pdf",
        help="Max recommended size: ~50 pages for fast processing.",
    )

    process_btn = st.button(
        "⚡ Process PDF",
        use_container_width=True,
        disabled = not uploaded_file,
    )

    # ── Process the PDF when button clicked ───────────────────────────────────
    if process_btn:
        with st.spinner("Reading PDF, chunking text, and building FAISS index..."):
            try:
                qa_chain, chunk_count = process_pdf(uploaded_file, api_key)

                # Store in session state so it persists across reruns
                st.session_state.qa_chain = qa_chain
                st.session_state.chunk_count = chunk_count
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.pdf_processed = True
                st.session_state.chat_history = []  # Reset chat for new PDF

                st.success("✅ PDF processed!")

            except Exception as e:
                st.error(f"Error processing PDF: {e}")

    # ── Stats panel (shown after processing) ──────────────────────────────────
    if st.session_state.pdf_processed:
        st.divider()
        st.markdown("### 📊 Index Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div style="font-size:22px; color:#60a5fa;">
                    {st.session_state.chunk_count}
                </div>
                <div style="font-size:11px; color:#6b7280;">Chunks</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div style="font-size:22px; color:#34d399;">FAISS</div>
                <div style="font-size:11px; color:#6b7280;">Index</div>
            </div>""", unsafe_allow_html=True)

        st.caption(f"📎 {st.session_state.pdf_name}")

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown("# 📄 Chat with PDF")
st.caption("Upload a PDF, process it, then ask anything about its content.")
st.divider()

# ── Empty state ───────────────────────────────────────────────────────────────
if not st.session_state.pdf_processed:
    st.info("👈 Upload a PDF and click **Process PDF** in the sidebar to get started.")
    st.stop()

# ── Chat history display ───────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">🧑 {msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="assistant-bubble">🤖 {msg["content"]}</div>',
            unsafe_allow_html=True,
        )
        # Show source page numbers if available
        if msg.get("sources"):
            source_html = " ".join(
                f'<span class="source-badge">📄 Page {s}</span>'
                for s in msg["sources"]
            )
            st.markdown(source_html, unsafe_allow_html=True)

# ── Chat input ─────────────────────────────────────────────────────────────────
user_query = st.chat_input("Ask a question about your PDF...")

if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    with st.spinner("Thinking..."):
        try:
            from langchain_core.messages import HumanMessage, AIMessage

            # Build LangChain message history
            lc_history = []
            for msg in st.session_state.chat_history[:-1]:
                if msg["role"] == "user":
                    lc_history.append(HumanMessage(content=msg["content"]))
                else:
                    lc_history.append(AIMessage(content=msg["content"]))

            # Invoke — chain now returns string directly
            answer = st.session_state.qa_chain.invoke({
                "input": user_query,
                "chat_history": lc_history,
            })

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": [],
            })

        except Exception as e:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"⚠️ Error: {e}",
                "sources": [],
            })

    st.rerun()
