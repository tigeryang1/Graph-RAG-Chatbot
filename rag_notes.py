import hashlib
from pathlib import Path
from typing import Any

from graph_rag_auth import get_api_key
from graph_rag_config import DEFAULT_EMBEDDING_MODEL

try:
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError:  # pragma: no cover - optional until installed
    FAISS = None
    Document = Any
    GoogleGenerativeAIEmbeddings = None
    RecursiveCharacterTextSplitter = None


KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent / "knowledge_base"

_cached_vector_store = None
_cached_kb_fingerprint: str | None = None


def _kb_fingerprint() -> str:
    """Hash based on file paths, sizes, and modification times."""
    entries: list[str] = []
    for path in list_note_files():
        stat = path.stat()
        entries.append(f"{path}:{stat.st_size}:{stat.st_mtime_ns}")
    return hashlib.md5("|".join(entries).encode()).hexdigest()


def list_note_files() -> list[Path]:
    if not KNOWLEDGE_BASE_DIR.exists():
        return []
    return sorted(KNOWLEDGE_BASE_DIR.glob("*.txt"))


def load_note_documents() -> list[Document]:
    if RecursiveCharacterTextSplitter is None:
        raise ModuleNotFoundError(
            "langchain-text-splitters is required to build the local note index."
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    docs: list[Document] = []
    for path in list_note_files():
        content = path.read_text(encoding="utf-8")
        for index, chunk in enumerate(splitter.split_text(content)):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "name": path.name,
                        "path": str(path),
                        "chunk_index": index,
                    },
                )
            )
    return docs


def build_note_vector_store(documents: list[Document]):
    if FAISS is None or GoogleGenerativeAIEmbeddings is None:
        raise ModuleNotFoundError(
            "langchain-community and langchain-google-genai are required to build the local note index."
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model=DEFAULT_EMBEDDING_MODEL,
        google_api_key=get_api_key(),
    )
    return FAISS.from_documents(documents, embedding=embeddings)


def get_vector_store():
    """Return the cached FAISS vector store, rebuilding only when knowledge_base files change."""
    global _cached_vector_store, _cached_kb_fingerprint

    fingerprint = _kb_fingerprint()
    if _cached_vector_store is not None and fingerprint == _cached_kb_fingerprint:
        return _cached_vector_store

    documents = load_note_documents()
    if not documents:
        _cached_vector_store = None
        _cached_kb_fingerprint = fingerprint
        return None

    _cached_vector_store = build_note_vector_store(documents)
    _cached_kb_fingerprint = fingerprint
    return _cached_vector_store


def retrieve_supporting_docs(question: str, max_docs: int = 3) -> list[dict[str, str | int]]:
    vector_store = get_vector_store()
    if vector_store is None:
        return []

    matches = vector_store.similarity_search_with_score(question, k=max_docs)

    hits: list[dict[str, str | int]] = []
    for doc, score in matches:
        snippet = " ".join(doc.page_content.split())[:220]
        hits.append(
            {
                "name": doc.metadata.get("name", "unknown"),
                "path": doc.metadata.get("path", ""),
                "score": int(round(float(score))),
                "snippet": snippet,
            }
        )
    return hits


def build_document_context(doc_hits: list[dict[str, str | int]]) -> str:
    if not doc_hits:
        return "No supporting account notes matched the question."

    lines = []
    for item in doc_hits:
        lines.append(f"[{item['name']}] {item['snippet']}")
    return "\n".join(lines)
