from langchain_ollama.llms import OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings
from .config import AppConfig


def get_llm(cfg: AppConfig) -> OllamaLLM:
    return OllamaLLM(model=cfg.models.llm)


def get_embeddings(cfg: AppConfig) -> OllamaEmbeddings:
    return OllamaEmbeddings(model=cfg.models.embed)
