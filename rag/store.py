from pathlib import Path
from langchain_community.vectorstores import FAISS


def save_faiss(db: FAISS, dir_path: Path):
    dir_path.mkdir(parents=True, exist_ok=True)
    db.save_local(str(dir_path))


def load_faiss(dir_path: Path) -> FAISS | None:
    if not dir_path.exists():
        return None
    try:
        return FAISS.load_local(str(dir_path), allow_dangerous_deserialization=True)
    except Exception:
        return None
