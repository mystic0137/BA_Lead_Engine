#handles state of the entire app
import json
import logging
import onnxruntime as rt
from functools import lru_cache

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from groq import Groq

from src.config import (
    XGBOOST_ONNX_PATH, XGBOOST_CONFIG_PATH,
    CHROMA_DB_PATH, EMBEDDING_MODEL, settings
)
from src.inference.engine import InferenceEngine
from src.analytics.finance import BACostCalculator

logger = logging.getLogger(__name__)


class MLState:
    def __init__ (self):
        self.engine = None
        self.embedding_model = None
        self.chroma_collection = None
        self.groq_client = None
    
    def is_ready(self):
        return self.engine is not None
    
    def is_rag_ready(self):
        return (
            self.embedding_model is not None
            and self.chroma_collection is not None
            and self.groq_client is not None
        )
    
    def initialize(self):
        if self.engine is not None:
            return
        
        self.engine = InferenceEngine()
        
        logger.info("Loading RAG components...")
        try:
            self.embedding_model = SentenceTransformer(str(EMBEDDING_MODEL))
            logger.info("Embedding model loaded")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
        
        try:
            chroma_client = chromadb.PersistentClient(
                path=str(CHROMA_DB_PATH),
                settings=Settings(anonymized_telemetry=False),
            )
            self.chroma_collection = chroma_client.get_collection("ba_policies")
            logger.info("ChromaDB collection loaded")
        except Exception as e:
            logger.error(f"Failed to load ChromaDB collection: {e}")
        
        try:
            api_key = settings.GROQ_API_KEY.get_secret_value()
            self.groq_client = Groq(api_key=api_key)
            logger.info("Groq client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
    
    def clear(self):
        self.engine = None
        self.embedding_model = None
        self.chroma_collection = None
        self.groq_client = None

ml = MLState()