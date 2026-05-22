import hashlib
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from src.config import CHROMA_DB_PATH, POLICIES_DIR, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def _load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_document(path: Path) -> str:
    if path.suffix == ".pdf":
        return _load_pdf(path)
    elif path.suffix in (".md", ".txt"):
        return _load_markdown(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Sliding window chunker. Splits on whitespace boundaries to avoid
    cutting mid-sentence. Overlap preserves context across chunk edges.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def _doc_id(text: str, source: str, idx: int) -> str:
    h = hashlib.md5(f"{source}:{idx}:{text[:50]}".encode()).hexdigest()[:8]
    return f"{Path(source).stem}_{idx}_{h}"


def build_vectorstore() -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name="ba_policies",
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def ingest(force: bool = False) -> None:
    """
    Load all documents from POLICIES_DIR, chunk, embed, and store in ChromaDB.
    Set force=True to re-ingest even if documents already exist.
    """
    policy_files = list(POLICIES_DIR.glob("**/*.pdf")) + \
                   list(POLICIES_DIR.glob("**/*.md")) + \
                   list(POLICIES_DIR.glob("**/*.txt"))

    if not policy_files:
        raise FileNotFoundError(f"No policy documents found in {POLICIES_DIR}")

    logger.info("Found %d policy documents", len(policy_files))

    model = SentenceTransformer(str(EMBEDDING_MODEL))
    collection = build_vectorstore()

    existing = set(collection.get()["ids"])

    for doc_path in policy_files:
        logger.info("Ingesting: %s", doc_path.name)
        text = _load_document(doc_path)
        chunks = _chunk_text(text)

        ids, documents, embeddings, metadatas = [], [], [], []

        for i, chunk in enumerate(chunks):
            doc_id = _doc_id(chunk, str(doc_path), i)
            if not force and doc_id in existing:
                continue

            embedding = model.encode(chunk).tolist()
            ids.append(doc_id)
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest()