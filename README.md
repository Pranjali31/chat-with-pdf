# 📄 Chat with PDF — RAG App

A production-style Retrieval-Augmented Generation (RAG) app that lets you upload any PDF and have a conversation with its contents — powered by LangChain, FAISS, OpenAI, and Streamlit.

---
# Open README.md and add at the top:

## 🚀 Live Demo
👉 (https://chat-with-pdf-7uetemfdtgmmsxjhzsjxsm.streamlit.app/)

## 🏗️ Architecture

```
PDF Upload
    │
    ▼
┌─────────────────────────────────────────────────┐
│  STEP 1 — Load                                  │
│  PyPDFLoader reads the PDF page by page         │
│  → List of Document objects                     │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│  STEP 2 — Chunk                                 │
│  RecursiveCharacterTextSplitter breaks pages    │
│  into overlapping ~800-char chunks              │
│  → Smaller, semantically coherent pieces        │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│  STEP 3 — Embed & Index                         │
│  OpenAI text-embedding-3-small converts each    │
│  chunk into a 1536-dim vector                   │
│  FAISS stores all vectors in an in-memory index │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│  STEP 4 — Retrieve & Generate (at query time)   │
│  User question → embed → FAISS similarity       │
│  search → top 4 chunks retrieved               │
│  → Injected into GPT-3.5-turbo prompt           │
│  → Answer grounded in your document            │
└─────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone / download the project

```bash
cd chat-with-pdf
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your OpenAI API key

```bash
cp .env.example .env
# Edit .env and paste your OpenAI key
```

### 5. Run the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 📁 Project Structure

```
chat-with-pdf/
│
├── app.py                  # Streamlit UI — all pages, chat, sidebar
├── src/
│   └── rag_pipeline.py     # RAG logic — load, chunk, embed, index, chain
│
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
└── README.md
```

---

## ⚙️ Key Configuration (rag_pipeline.py)

| Constant | Default | What it controls |
|---|---|---|
| `CHUNK_SIZE` | `800` | Characters per chunk. Smaller = more precise retrieval |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks to avoid losing context at borders |
| `RETRIEVER_TOP_K` | `4` | How many chunks are sent to the LLM per question |
| `LLM_MODEL` | `gpt-3.5-turbo` | Swap for `gpt-4o` for better reasoning |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Smaller/faster; use `text-embedding-3-large` for higher accuracy |

---

## 💡 How RAG Works (Plain English)

Traditional LLMs have a fixed context window and no access to your files. RAG solves this by:

1. **Pre-processing**: Your PDF is broken into chunks and each chunk is turned into a vector (a list of numbers that captures its meaning).
2. **At query time**: Your question is also turned into a vector, and the system finds the chunks with the most similar vectors — these are the most relevant pieces of your document.
3. **Augmented prompt**: Those chunks + your question are sent to the LLM together. The LLM answers *only from that context*, which keeps answers grounded and reduces hallucination.

---

