import json
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embeddings import get_embeddings


load_dotenv()

DB_DIRECTORY = Path("chroma_db")
DB_STORES_DIRECTORY = DB_DIRECTORY / "stores"
ACTIVE_DB_POINTER = DB_DIRECTORY / "active_db.json"
COLLECTION_NAME = "coursemate"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DEFAULT_EMPTY_RESPONSE = "I could not find the answer in the document."

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use only the provided context to answer the user's question.
If the answer is not present in the context, say:
"I could not find the answer in the document."
Keep the answer concise and grounded in the document.
""",
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
""",
        ),
    ]
)


@dataclass(frozen=True)
class IngestionResult:
    document_name: str
    page_count: int
    chunk_count: int


def require_mistral_api_key() -> None:
    if not os.getenv("MISTRAL_API_KEY"):
        raise RuntimeError("Missing MISTRAL_API_KEY. Add it to your environment or .env file.")


def _ensure_db_root() -> None:
    DB_DIRECTORY.mkdir(parents=True, exist_ok=True)
    DB_STORES_DIRECTORY.mkdir(parents=True, exist_ok=True)


def _read_active_db_pointer() -> Path | None:
    if not ACTIVE_DB_POINTER.exists():
        return None

    try:
        payload = json.loads(ACTIVE_DB_POINTER.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    relative_path = payload.get("active_db")
    if not relative_path:
        return None

    active_path = Path(relative_path)
    if not active_path.is_absolute():
        active_path = (DB_DIRECTORY / active_path).resolve()

    return active_path


def get_active_database_directory() -> Path | None:
    active_path = _read_active_db_pointer()
    if active_path and active_path.exists():
        return active_path

    legacy_db_present = (DB_DIRECTORY / "chroma.sqlite3").exists()
    if legacy_db_present:
        return DB_DIRECTORY

    return None


def database_exists(persist_directory: Path = DB_DIRECTORY) -> bool:
    if persist_directory != DB_DIRECTORY:
        return persist_directory.exists() and any(persist_directory.iterdir())

    active_path = get_active_database_directory()
    return active_path is not None and any(active_path.iterdir())


def _set_active_database_directory(db_path: Path) -> None:
    _ensure_db_root()
    payload = {
        "active_db": str(db_path.resolve().relative_to(DB_DIRECTORY.resolve())),
    }
    ACTIVE_DB_POINTER.write_text(json.dumps(payload), encoding="utf-8")


def _create_new_store_directory() -> Path:
    _ensure_db_root()
    return DB_STORES_DIRECTORY / f"db_{uuid4().hex}"


def cleanup_inactive_stores(keep_recent: int = 3) -> None:
    active_store = get_active_database_directory()
    if not DB_STORES_DIRECTORY.exists():
        return

    candidate_stores = sorted(
        [path for path in DB_STORES_DIRECTORY.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    retained = 0
    for store_path in candidate_stores:
        if active_store and store_path.resolve() == active_store.resolve():
            retained += 1
            continue

        if retained < keep_recent:
            retained += 1
            continue

        try:
            for nested in sorted(store_path.rglob("*"), reverse=True):
                if nested.is_file():
                    nested.unlink(missing_ok=True)
                elif nested.is_dir():
                    nested.rmdir()
            store_path.rmdir()
        except OSError:
            continue


def _split_documents(pdf_path: str, document_name: str) -> tuple[list[Document], int]:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    for index, chunk in enumerate(chunks):
        chunk.metadata.setdefault("source", document_name)
        chunk.metadata["chunk_index"] = index

    return chunks, len(docs)


def ingest_pdf(
    pdf_path: str,
    document_name: str,
    replace_existing: bool = True,
    persist_directory: Path = DB_DIRECTORY,
) -> IngestionResult:
    chunks, page_count = _split_documents(pdf_path, document_name)
    target_directory = persist_directory

    if persist_directory == DB_DIRECTORY:
        target_directory = _create_new_store_directory()
        target_directory.mkdir(parents=True, exist_ok=True)

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(target_directory),
        collection_name=COLLECTION_NAME,
    )

    if replace_existing and persist_directory == DB_DIRECTORY:
        _set_active_database_directory(target_directory)
        cleanup_inactive_stores()

    return IngestionResult(
        document_name=document_name,
        page_count=page_count,
        chunk_count=len(chunks),
    )


def load_vectorstore(persist_directory: Path = DB_DIRECTORY) -> Chroma:
    resolved_directory = persist_directory
    if persist_directory == DB_DIRECTORY:
        active_directory = get_active_database_directory()
        if active_directory is None:
            raise FileNotFoundError("No vector database found. Upload and index a PDF first.")
        resolved_directory = active_directory

    if not database_exists(resolved_directory):
        raise FileNotFoundError("No vector database found. Upload and index a PDF first.")

    return Chroma(
        persist_directory=str(resolved_directory),
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME,
    )


def build_retriever(persist_directory: Path = DB_DIRECTORY):
    return load_vectorstore(persist_directory).as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5,
        },
    )


def build_llm() -> ChatMistralAI:
    require_mistral_api_key()
    return ChatMistralAI(model="mistral-small-2506", temperature=0)


def answer_question(question: str, persist_directory: Path = DB_DIRECTORY) -> tuple[str, list[Document]]:
    clean_question = question.strip()
    if not clean_question:
        raise ValueError("Question cannot be empty.")

    retriever = build_retriever(persist_directory)
    docs = retriever.invoke(clean_question)
    if not docs:
        return DEFAULT_EMPTY_RESPONSE, []

    context = "\n\n".join(doc.page_content for doc in docs)
    response = build_llm().invoke(
        PROMPT.invoke({"context": context, "question": clean_question})
    )
    return response.content.strip() or DEFAULT_EMPTY_RESPONSE, docs
