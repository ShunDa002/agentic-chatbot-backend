from pathlib import Path
import hashlib
import sqlite3
import threading
from typing import List
from dotenv import load_dotenv
import os
import certifi
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import docx2txt
from langchain_openai import OpenAIEmbeddings

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
Path("uploads").mkdir(exist_ok=True)
Path("chroma_db").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)
INGESTION_DB = Path("data/rag_ingestions.sqlite")
_INGESTION_LOCK = threading.Lock()


def _init_ingestion_db() -> None:
    with sqlite3.connect(INGESTION_DB) as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS rag_ingestions (
            thread_id TEXT NOT NULL, content_sha256 TEXT NOT NULL,
            filename TEXT NOT NULL, chunks INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (thread_id, content_sha256))""")


_init_ingestion_db()

embeddings = OpenAIEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    api_key=os.environ["MODAL_EMBEDDING_API_KEY"],
    base_url=os.environ["MODAL_EMBEDDING_BASE_URL"],
    chunk_size=32,
    check_embedding_ctx_length=False,
    model_kwargs={"encoding_format": "float"},
    request_timeout=120,
    max_retries=2,
)
vectorstore = Chroma(
    collection_name="agentic_chatbot_docs",
    embedding_function=embeddings,
    persist_directory="chroma_db",
)


def read_file_text(file_path: str) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "\n".join(
            (page.extract_text() or "") for page in PdfReader(file_path).pages
        )
    if suffix == ".docx":
        return docx2txt.process(file_path)
    if suffix in {".txt", ".md", ".py", ".csv"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError("Unsupported file type. Upload PDF, DOCX, TXT, MD, PY, or CSV.")


def add_document_to_rag(file_path: str, thread_id: str, source_name: str | None = None):
    path = Path(file_path)
    display_name = Path(source_name).name if source_name else path.name
    content_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    with _INGESTION_LOCK, sqlite3.connect(INGESTION_DB) as connection:
        existing = connection.execute(
            "SELECT filename, chunks FROM rag_ingestions WHERE thread_id=? AND content_sha256=?",
            (thread_id, content_sha256),
        ).fetchone()
        if existing:
            return {
                "filename": display_name,
                "original_filename": existing[0],
                "chunks": existing[1],
                "sha256": content_sha256,
                "duplicate": True,
                "status": "already_ingested",
            }
        text = read_file_text(str(path))
        if not text.strip():
            raise ValueError("No text could be extracted from this file.")
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=700, chunk_overlap=100
        ).split_text(text)
        docs = [
            Document(
                page_content=c,
                metadata={
                    "thread_id": thread_id,
                    "source": display_name,
                    "content_sha256": content_sha256,
                    "chunk_index": i,
                },
            )
            for i, c in enumerate(chunks)
        ]
        ids = [f"{thread_id}:{content_sha256}:{i}" for i in range(len(docs))]
        vectorstore.add_documents(docs, ids=ids)
        connection.execute(
            "INSERT INTO rag_ingestions (thread_id,content_sha256,filename,chunks) VALUES (?,?,?,?)",
            (thread_id, content_sha256, display_name, len(docs)),
        )
        connection.commit()
    return {
        "filename": display_name,
        "chunks": len(docs),
        "sha256": content_sha256,
        "duplicate": False,
        "status": "ingested",
    }


def retrieve_documents(query: str, thread_id: str, k: int = 6) -> list[Document]:
    if not query.strip():
        return []
    return vectorstore.similarity_search(query, k=k, filter={"thread_id": thread_id})


def format_retrieved_documents(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[Source {i}: {d.metadata.get('source','uploaded document')}, chunk {d.metadata.get('chunk_index','?')}]\n{d.page_content}"
        for i, d in enumerate(docs, 1)
    )


def retrieve_from_rag(query: str, thread_id: str, k: int = 6) -> str:
    docs = retrieve_documents(query, thread_id, k)
    return (
        format_retrieved_documents(docs)
        if docs
        else "No relevant uploaded document content found."
    )


def has_uploaded_documents(thread_id: str) -> bool:
    with sqlite3.connect(INGESTION_DB) as connection:
        return (
            connection.execute(
                "SELECT 1 FROM rag_ingestions WHERE thread_id=? LIMIT 1", (thread_id,)
            ).fetchone()
            is not None
        )
