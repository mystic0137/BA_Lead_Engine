import logging
import chromadb

from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from src.config import (
    CHROMA_DB_PATH, EMBEDDING_MODEL, get_settings
)
from src.inference.engine import InferenceEngine

logger = logging.getLogger(__name__)


class MLState:
    def __init__(self):
        self.engine = None
        self.embedding_model = None
        self.chroma_collection = None

    def is_ready(self):
        return self.engine is not None

    def is_rag_ready(self):
        return (
            self.embedding_model is not None
            and self.chroma_collection is not None
        )

    def initialize(self):
        if self.engine is not None:
            return
        self.engine = InferenceEngine()
        logger.info("Inference engine loaded")

    def initialize_rag(self):
        if self.is_rag_ready():
            return
        logger.info("Loading RAG components...")
        try:
            self.embedding_model = SentenceTransformer(str(EMBEDDING_MODEL))
            logger.info("Embedding model loaded")
        except Exception as e:
            logger.error("Failed to load embedding model: %s", e)
            self.embedding_model = None
            raise RuntimeError(f"Failed to load embedding model: {e}") from e

        try:
            chroma_client = chromadb.PersistentClient(
                path=str(CHROMA_DB_PATH),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self.chroma_collection = chroma_client.get_collection("ba_policies")
            logger.info("ChromaDB collection loaded")
        except Exception as e:
            logger.error("Failed to load ChromaDB collection: %s", e)
            self.chroma_collection = None
            raise RuntimeError(f"Failed to load ChromaDB collection: {e}") from e

    def clear(self):
        self.engine = None
        self.embedding_model = None
        self.chroma_collection = None


ml = MLState()
