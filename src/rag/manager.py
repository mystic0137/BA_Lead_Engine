import logging
import os

from src.config import (
    ACTIVE_SYSTEM_PROMPT_ID, FINETUNING_DIR
)
from src.rag.prompts import (
    build_user_prompt, get_system_prompt, build_query, format_chunks, parse_email
)
from src.rag.ingestion import ingest as _ingest
from src.rag.feedback import Label, save_feedback, load_feedback, feedback_stats
from src.rag.llm_client import generate_chat
from core.state import ml

logger = logging.getLogger(__name__)


class RAGManager:
    def __init__(self):
        os.makedirs(FINETUNING_DIR, exist_ok=True)
        self.feedback_log = FINETUNING_DIR / "feedback_log.jsonl"
        self.sft_log = FINETUNING_DIR / "sft_log.jsonl"
        self.dpo_log = FINETUNING_DIR / "dpo_log.jsonl"
        self._retrieval_cache: dict[str, list[dict]] = {}
        self._cache_max = 128

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        cache_key = f"{query}::top_{top_k}"
        cached = self._retrieval_cache.get(cache_key)
        if cached is not None:
            logger.debug("RAG cache hit for query: %s", query[:60])
            return cached

        if not ml.is_rag_ready():
            raise RuntimeError("RAG components not initialized")
        embedding = ml.embedding_model.encode(query).tolist()
        results = ml.chroma_collection.query(
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
                "score": round(1 - distance, 4),
            })
        logger.debug("Retrieved %d chunks for query: %s", len(chunks), query[:60])

        if len(self._retrieval_cache) >= self._cache_max:
            self._retrieval_cache.clear()
        self._retrieval_cache[cache_key] = chunks
        return chunks

    def generate_email(self, lead: dict) -> dict:
        query = build_query(lead)
        chunks = self.retrieve(query)
        policy_context = format_chunks(chunks)
        system_prompt = get_system_prompt()
        user_prompt = build_user_prompt(lead, policy_context=policy_context)
        result = generate_chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        subject, body = parse_email(result["raw"])
        return {
            "subject": subject,
            "body": body,
            "provider": result["provider"],
            "model": result["model"],
            "retrieved_sources": [c["source"] for c in chunks],
            "system_prompt_id": ACTIVE_SYSTEM_PROMPT_ID,
            "tokens_input": result["tokens_input"],
            "tokens_output": result["tokens_output"],
            "latency_ms": result["latency_ms"],
        }

    def ingest(self, force: bool = False) -> None:
        _ingest(force)

    def save_feedback(self, **kwargs) -> None:
        kwargs["feedback_log"] = self.feedback_log
        kwargs["sft_log"] = self.sft_log
        kwargs["dpo_log"] = self.dpo_log
        save_feedback(**kwargs)

    def load_feedback(self) -> list[dict]:
        return load_feedback(self.feedback_log)

    def feedback_stats(self) -> dict:
        return feedback_stats(self.feedback_log, self.sft_log, self.dpo_log)
