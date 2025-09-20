from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from .llm import get_embeddings
from .config import AppConfig


def load_pdfs(pdf_paths: List[Path]):
    docs = []
    for p in pdf_paths:
        loader = PyPDFLoader(str(p))
        docs.extend(loader.load())
    return docs


def split_docs(docs, cfg: AppConfig):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.rag.chunk_size, chunk_overlap=cfg.rag.chunk_overlap
    )
    return splitter.split_documents(docs)


def build_faiss(docs, cfg: AppConfig) -> FAISS:
    emb = get_embeddings(cfg)
    return FAISS.from_documents(docs, emb)
