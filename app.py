
import os
import tempfile
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage


load_dotenv()

st.set_page_config(page_title="PDF RAG Chatbot (Groq)", layout="wide")


# -----------------------------
# Caching heavy resources
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    # Downloaded once, cached by Streamlit
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def get_llm(model_name: str, temperature: float):
    # Uses GROQ_API_KEY from environment
    return ChatGroq(model=model_name, temperature=temperature)


def build_vectorstore_from_pdf(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()  # each page is a Document with metadata {source, page}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings()
    vs = FAISS.from_documents(chunks, embeddings)
    return vs


def format_sources(docs) -> List[str]:
    sources = []
    for d in docs:
        src = d.metadata.get("source", "uploaded_pdf")
        page = d.metadata.get("page", None)
        if page is not None:
            sources.append(f"{os.path.basename(src)} (page {page + 1})")
        else:
            sources.append(f"{os.path.basename(src)}")
    # de-duplicate while preserving order
    seen = set()
    out = []
    for s in sources:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def rag_answer(
    question: str,
    vectorstore: FAISS,
    chat_history: List[Tuple[str, str]],
    k: int,
    model_name: str,
    temperature: float,
):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(question)

    context = "\n\n".join(
        [f"[Page {d.metadata.get('page', '?') + 1}] {d.page_content}" for d in docs]
    )

    history_text = ""
    # keep a short history to reduce prompt size
    for u, a in chat_history[-6:]:
        history_text += f"User: {u}\nAssistant: {a}\n"

    system_prompt = f"""
You are a helpful assistant answering questions ONLY using the provided PDF context.
If the answer is not in the context, say: "I don't know based on the provided document."
Be concise and accurate.

PDF Context:
{context}

Conversation so far:
{history_text}
""".strip()

    llm = get_llm(model_name=model_name, temperature=temperature)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ]
    response = llm.invoke(messages).content

    return response, docs


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("PDF RAG Chatbot (Groq + FAISS)")
st.write("Upload a PDF, build the index, then chat with it using Retrieval-Augmented Generation (RAG).")

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox(
        "Groq model",
        options=[
            "llama-3.1-8b-instant",
            "llama3-70b-8192",
            "mixtral-8x7b-32768",
        ],
        index=0,
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    k = st.slider("Top-k chunks to retrieve", 1, 8, 4, 1)
    chunk_size = st.slider("Chunk size", 300, 2000, 1000, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 150, 10)

    st.divider()
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

    col_a, col_b = st.columns(2)
    with col_a:
        build_btn = st.button("Build / Rebuild Index", use_container_width=True)
    with col_b:
        reset_btn = st.button("Reset Chat", use_container_width=True)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": "..."}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (user, assistant)

if reset_btn:
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.rerun()

# Build index when button pressed
if build_btn:
    if not uploaded_pdf:
        st.error("Please upload a PDF first.")
    else:
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            st.error("Missing GROQ_API_KEY. Put it in your .env file.")
        else:
            with st.spinner("Saving PDF and building vector index..."):
                # Save uploaded file to a temporary path
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_pdf.read())
                    tmp_path = tmp.name

                try:
                    st.session_state.vectorstore = build_vectorstore_from_pdf(
                        tmp_path,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                    st.success("Index built. You can now chat with the PDF.")
                finally:
                    # Remove temp file
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

# Show existing chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
user_question = st.chat_input("Ask a question about the uploaded PDF...")

if user_question:
    if st.session_state.vectorstore is None:
        st.error("Please upload a PDF and click 'Build / Rebuild Index' first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving from PDF and generating answer..."):
                answer, retrieved_docs = rag_answer(
                    question=user_question,
                    vectorstore=st.session_state.vectorstore,
                    chat_history=st.session_state.chat_history,
                    k=k,
                    model_name=model_name,
                    temperature=temperature,
                )
                st.markdown(answer)

                sources = format_sources(retrieved_docs)
                with st.expander("Sources (retrieved pages)"):
                    if sources:
                        st.write("\n".join([f"- {s}" for s in sources]))
                    else:
                        st.write("No sources found.")

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_history.append((user_question, answer))