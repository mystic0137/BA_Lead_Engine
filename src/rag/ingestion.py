import hashlib
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from src.config import CHROMA_DB_PATH, EMBEDDING_MODEL, POLICIES_DIR

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_document(path: Path) -> str:
    if path.suffix == ".pdf":
        return load_pdf(path)
    elif path.suffix in (".md", ".txt"):
        return load_markdown(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    chunk_size = chunk_size or CHUNK_SIZE
    overlap = overlap or CHUNK_OVERLAP
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def doc_id(text: str, source: str, idx: int) -> str:
    h = hashlib.md5(f"{source}:{idx}:{text[:50]}".encode()).hexdigest()[:8]
    return f"{Path(source).stem}_{idx}_{h}"


def build_vectorstore() -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_PATH),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name="ba_policies",
        metadata={"hnsw:space": "cosine"},
    )


def ingest(force: bool = False) -> None:
    policy_files = (
        list(POLICIES_DIR.glob("**/*.pdf")) +
        list(POLICIES_DIR.glob("**/*.md")) +
        list(POLICIES_DIR.glob("**/*.txt"))
    )
    if not policy_files:
        raise FileNotFoundError(f"No policy documents found in {POLICIES_DIR}")
    logger.info("Found %d policy documents", len(policy_files))
    model = SentenceTransformer(str(EMBEDDING_MODEL))
    collection = build_vectorstore()
    existing = set(collection.get()["ids"])
    for doc_path in policy_files:
        logger.info("Ingesting: %s", doc_path.name)
        text = load_document(doc_path)
        chunks = chunk_text(text)
        ids, documents, embeddings, metadatas = [], [], [], []
        for i, chunk in enumerate(chunks):
            chunk_id = doc_id(chunk, str(doc_path), i)
            if not force and chunk_id in existing:
                continue
            embedding = model.encode(chunk).tolist()
            ids.append(chunk_id)
            documents.append(chunk)
            embeddings.append(embedding)
            metadatas.append({
                "source": doc_path.name,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })
        if ids:
            collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            logger.info("Stored %d chunks from %s", len(ids), doc_path.name)
        else:
            logger.info("Skipped %s — already ingested", doc_path.name)
    logger.info("Ingestion complete. Total chunks: %d", collection.count())
