import logging
from functools import lru_cache

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.config import CHROMA_DB_PATH
from src.rag.ingest import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

TOP_K = 3


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection("ba_policies")


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Embed query and return top_k most relevant policy chunks.
    Returns list of dicts with 'text', 'source', 'score'.
    """
    model = _get_model()
    collection = _get_collection()

    embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "score": round(1 - distance, 4),  # cosine distance → similarity
        })

    logger.debug("Retrieved %d chunks for query: %s", len(chunks), query[:60])
    return chunks