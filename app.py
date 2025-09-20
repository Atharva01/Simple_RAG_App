import streamlit as st
from pathlib import Path
from rag.config import AppConfig
from rag.ingest import load_pdfs, split_docs, build_faiss
from rag.store import save_faiss, load_faiss
from rag.chain import build_chain
from rag.utils import save_uploaded_files

st.set_page_config(page_title="Simple RAG (Ollama + FAISS)", layout="wide")
cfg = AppConfig()

st.title("🔎 Simple RAG — Ollama + FAISS — Streamlit")


@st.cache_resource(show_spinner=False)
def get_db_cached():
    return load_faiss(cfg.paths.vs_dir)


@st.cache_resource(show_spinner=False)
def get_ask_fn_cached():
    db = get_db_cached()
    if db is None:
        return None
    return build_chain(db, cfg)


with st.sidebar:
    st.header("Ingestion")
    uploaded = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Build / Update Index") and uploaded:
        pdf_paths = save_uploaded_files(uploaded, cfg.paths.docs_dir)
        with st.spinner("Loading → splitting → embedding → indexing…"):
            docs = load_pdfs(pdf_paths)
            chunks = split_docs(docs, cfg)
            db = load_faiss(cfg.paths.vs_dir)
            if db is None:
                db = build_faiss(chunks, cfg)
            else:
                db.add_documents(chunks)
            save_faiss(db, cfg.paths.vs_dir)
        st.success("Index saved ✅")
        st.cache_resource.clear()  # refresh caches

db = get_db_cached()
if db is None:
    st.info("No vector store found. Upload PDFs and click “Build / Update Index”.")
else:
    ask = get_ask_fn_cached()
    q = st.text_input("Ask a question about your documents")
    if st.button("Ask") and q.strip():
        with st.spinner("Thinking…"):
            result = ask(q.strip())
        st.subheader("Answer")
        st.write(result["answer"])
        with st.expander("Retrieved Context"):
            for i, doc in enumerate(result["context"]):
                st.markdown(
                    f"**Chunk {i+1}** — {doc.metadata.get('source', '')}")
                snippet = doc.page_content[:1200]
                st.write(snippet + ("…" if len(doc.page_content) > 1200 else ""))
