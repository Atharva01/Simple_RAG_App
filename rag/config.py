from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Paths:
    root: Path = Path(os.getenv("RAG_ROOT", ".")).resolve()
    docs_dir: Path = root / "data" / "docs"
    vs_dir: Path = root / "data" / "vectorstore"


@dataclass(frozen=True)
class Models:
    llm: str = os.getenv("LLM_MODEL", "gpt-oss:20b")
    embed: str = os.getenv("EMBED_MODEL", "snowflake-arctic-embed2:568m")


@dataclass(frozen=True)
class RAGConfig:
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    n_docs: int = int(os.getenv("TOP_K", "4"))
    strip_think_tags: bool = False


@dataclass(frozen=True)
class AppConfig:
    paths: Paths = Paths()
    models: Models = Models()
    rag: RAGConfig = RAGConfig()
