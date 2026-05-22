import json
import logging
import os
from datetime import datetime
from enum import Enum

from src.config import FINETUNING_DIR
from src.rag.prompt_builder import build_user_prompt

logger = logging.getLogger(__name__)

os.makedirs(FINETUNING_DIR, exist_ok=True)

FEEDBACK_LOG = FINETUNING_DIR / "feedback_log.jsonl"
SFT_LOG = FINETUNING_DIR / "sft_log.jsonl"
DPO_LOG = FINETUNING_DIR / "dpo_log.jsonl"


class Label(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EDITED = "edited"
    NEUTRAL = "neutral"


def _resolve_label(
    accepted: bool | None, was_edited: bool, rating: int
) -> tuple[str, str | None]:
    """
    Resolve label and flag contradictions between explicit action and star rating.
    Returns (label, contradiction_note | None).
    """
    if accepted is None:
        return Label.NEUTRAL, None

    if accepted and not was_edited:
        if rating < 3:
            return Label.ACCEPTED, "low_rating_but_accepted"
        return Label.ACCEPTED, None

    if accepted and was_edited:
        return Label.EDITED, None

    if rating >= 4:
        return Label.REJECTED, "high_rating_but_rejected"

    return Label.REJECTED, None


def _append(path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def save_feedback(
    lead: dict,
    system_prompt_id: str,
    generated_subject: str,
    generated_body: str,
    edited_subject: str,
    edited_body: str,
    rating: int,
    accepted: bool | None,
    tokens_input: int = 0,
    tokens_output: int = 0,
    latency_ms: int = 0,
) -> None:
    was_edited = (
        edited_subject.strip() != generated_subject.strip()
        or edited_body.strip() != generated_body.strip()
    )

    label, contradiction = _resolve_label(accepted, was_edited, rating)

    if contradiction:
        logger.warning(
            "Contradictory feedback for %s — label: %s, rating: %d, flag: %s",
            lead.get("customer_id"), label, rating, contradiction,
        )

    user_prompt_clean = build_user_prompt(lead)
    generated_completion = f"Subject: {generated_subject}\n\n{generated_body}"
    edited_completion = f"Subject: {edited_subject}\n\n{edited_body}"
    context_sources = list(set(lead.get("retrieved_sources", [])))

    base_meta = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "latency_ms": latency_ms,
        "timestamp": datetime.utcnow().isoformat(),
        "customer_id": lead.get("customer_id"),
        "was_edited": was_edited,
        "contradiction": contradiction,
    }

    # ── Full audit log — every record regardless of label ───────────────────
    full_record = {
        "system_prompt_id": system_prompt_id,
        "user_prompt": user_prompt_clean,
        "completion": edited_completion,
        "context_sources": context_sources,
        "label": label,
        "rating": rating,
        "is_preferred": accepted if accepted is not None else None,
        "meta": base_meta,
    }
    if was_edited:
        full_record["chosen"] = edited_completion
        full_record["rejected"] = generated_completion
    _append(FEEDBACK_LOG, full_record)

    # ── SFT log — accepted or edited, no contradictions ─────────────────────
    if label in (Label.ACCEPTED, Label.EDITED) and contradiction is None:
        sft_record = {
            "system_prompt_id": system_prompt_id,
            "prompt": user_prompt_clean,
            "completion": edited_completion,
            "label": label,
            "rating": rating,
            "meta": {
                "customer_id": base_meta["customer_id"],
                "was_edited": was_edited,
                "timestamp": base_meta["timestamp"],
            },
        }
        _append(SFT_LOG, sft_record)
        logger.info("SFT record saved for %s", lead.get("customer_id"))

    # ── DPO log — only when edited, no contradictions ────────────────────────
    if label == Label.EDITED and contradiction is None:
        dpo_record = {
            "system_prompt_id": system_prompt_id,
            "prompt": user_prompt_clean,
            "chosen": edited_completion,
            "rejected": generated_completion,
            "rating": rating,
            "meta": {
                "customer_id": base_meta["customer_id"],
                "timestamp": base_meta["timestamp"],
            },
        }
        _append(DPO_LOG, dpo_record)
        logger.info("DPO record saved for %s", lead.get("customer_id"))

    logger.info(
        "Feedback saved for %s — label: %s, rating: %d, edited: %s",
        lead.get("customer_id"), label, rating, was_edited,
    )


def load_feedback() -> list[dict]:
    if not FEEDBACK_LOG.exists():
        return []
    with open(FEEDBACK_LOG, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def feedback_stats() -> dict:
    records = load_feedback()
    if not records:
        return {"total": 0}

    ratings = [r["rating"] for r in records]
    edited = [r for r in records if r["meta"]["was_edited"]]
    contradictions = [r for r in records if r["meta"].get("contradiction")]

    sft_count = sum(1 for _ in open(SFT_LOG) if _.strip()) if SFT_LOG.exists() else 0
    dpo_count = sum(1 for _ in open(DPO_LOG) if _.strip()) if DPO_LOG.exists() else 0

    return {
        "total": len(records),
        "avg_rating": round(sum(ratings) / len(ratings), 2),
        "edited_count": len(edited),
        "edit_rate": round(len(edited) / len(records), 2),
        "contradiction_count": len(contradictions),
        "sft_pairs": sft_count,
        "dpo_pairs": dpo_count,
    }