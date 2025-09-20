import re
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from .llm import get_llm
from .config import AppConfig

PROMPT_TMPL = """Answer the following question based only on the provided context.
Think step by step before providing a detailed answer.
Please generate normal text. No markdown!
<context>
{context}
</context>
Question: {input}
"""


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def build_chain(db: FAISS, cfg: AppConfig):
    llm = get_llm(cfg)
    prompt = ChatPromptTemplate.from_template(PROMPT_TMPL)
    doc_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever(search_kwargs={"k": cfg.rag.n_docs})
    chain = create_retrieval_chain(retriever, doc_chain)

    def ask(q: str) -> dict:
        resp = chain.invoke({"input": q})
        ans = resp.get("answer", "")
        if cfg.rag.strip_think_tags:
            ans = strip_think(ans)
        return {"answer": ans, "context": resp.get("context", [])}

    return ask
