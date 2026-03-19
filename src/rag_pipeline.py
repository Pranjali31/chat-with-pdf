"""
src/rag_pipeline.py
-------------------
Handles all RAG (Retrieval-Augmented Generation) logic:
  - Load and split PDF into chunks
  - Embed chunks using OpenAI Embeddings
  - Store/retrieve chunks using FAISS vector store
  - Build a conversational RAG chain using LangChain 1.x LCEL syntax
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import tempfile
import os


# ── Constants ──────────────────────────────────────────────────────────────────

CHUNK_SIZE = 800          # Characters per chunk (tune for your use case)
CHUNK_OVERLAP = 100       # Overlap keeps context at chunk boundaries
RETRIEVER_TOP_K = 4       # Number of chunks retrieved per query
LLM_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-3-small"


# ── Step 1: Load PDF ───────────────────────────────────────────────────────────

def load_pdf(uploaded_file) -> list:
    """
    Save the Streamlit UploadedFile to a temp path and load it with PyPDFLoader.
    Returns a list of LangChain Document objects (one per page).
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    os.unlink(tmp_path)  # Clean up temp file

    return documents


# ── Step 2: Split into Chunks ──────────────────────────────────────────────────

def split_documents(documents: list) -> list:
    """
    Split pages into smaller chunks for more precise retrieval.

    RecursiveCharacterTextSplitter tries to split on paragraphs → sentences →
    words, keeping chunks semantically coherent.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],  # Try these separators in order
    )
    chunks = splitter.split_documents(documents)
    return chunks


# ── Step 3: Embed & Index with FAISS ──────────────────────────────────────────

def build_vector_store(chunks: list, api_key: str) -> FAISS:
    """
    Embed each chunk with OpenAI and store them in a FAISS index.

    FAISS (Facebook AI Similarity Search) stores vectors in memory and enables
    fast nearest-neighbour lookups — perfect for small-to-medium document sets.
    """
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=api_key,
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


# ── Step 4: Build Conversational RAG Chain (LangChain 1.x LCEL) ───────────────

def build_qa_chain(vector_store: FAISS, api_key: str):
    """
    LangChain 1.x pure LCEL chain — no legacy chains needed.
    Uses RunnablePassthrough to pipe components together explicitly.
    """
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        openai_api_key=api_key,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_TOP_K},
    )

    # Prompt that injects retrieved context + chat history + question
    prompt = ChatPromptTemplate.from_messages([
        ("system",
            "You are a precise document assistant. "
            "Answer using ONLY the context below. "
            "If the answer isn't in the context, say so honestly.\n\n"
            "Context:\n{context}"
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Format retrieved documents into a single string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # LCEL chain: retrieve → format → prompt → llm → parse
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["input"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# ── Main entry point ───────────────────────────────────────────────────────────

def process_pdf(uploaded_file, api_key: str):
    """
    Full pipeline: PDF → chunks → embeddings → FAISS → RAG chain.
    Returns the ready-to-use chain and the total chunk count.
    """
    documents = load_pdf(uploaded_file)
    chunks = split_documents(documents)
    vector_store = build_vector_store(chunks, api_key)
    qa_chain = build_qa_chain(vector_store, api_key)

    return qa_chain, len(chunks)

