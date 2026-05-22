import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path

from src.rag.prompts import build_user_prompt

logger = logging.getLogger(__name__)


class Label(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EDITED = "edited"
    NEUTRAL = "neutral"


def resolve_label(
    accepted: bool | None, was_edited: bool, rating: int
) -> tuple[str, str | None]:
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


def append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _build_lead_dict(
    customer_id: str,
    customer_name: str,
    booking_origin: str,
    haul_type: str,
    num_passengers: int,
    wants_extra_baggage: bool,
    wants_preferred_seat: bool,
    wants_in_flight_meals: bool,
    retrieved_sources: list[str],
) -> dict:
    return {
        "customer_id": customer_id,
        "customer_name": customer_name,
        "booking_origin": booking_origin,
        "haul_type": haul_type,
        "num_passengers": num_passengers,
        "wants_extra_baggage": wants_extra_baggage,
        "wants_preferred_seat": wants_preferred_seat,
        "wants_in_flight_meals": wants_in_flight_meals,
        "retrieved_sources": retrieved_sources,
    }


def _build_meta(
    tokens_input: int,
    tokens_output: int,
    latency_ms: int,
    customer_id: str,
    was_edited: bool,
    contradiction: str | None,
) -> dict:
    return {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "latency_ms": latency_ms,
        "timestamp": datetime.utcnow().isoformat(),
        "customer_id": customer_id,
        "was_edited": was_edited,
        "contradiction": contradiction,
    }


def _save_feedback_record(feedback_log: Path, record: dict) -> None:
    append(feedback_log, record)


def _save_sft_record(sft_log: Path, record: dict) -> None:
    append(sft_log, record)
    logger.info("SFT record saved for %s", record["meta"]["customer_id"])


def _save_dpo_record(dpo_log: Path, record: dict) -> None:
    append(dpo_log, record)
    logger.info("DPO record saved for %s", record["meta"]["customer_id"])


def save_feedback(
    feedback_log: Path,
    sft_log: Path,
    dpo_log: Path,
    customer_id: str,
    customer_name: str,
    email: str,
    route: str,
    booking_origin: str,
    haul_type: str,
    num_passengers: int,
    wants_extra_baggage: bool,
    wants_preferred_seat: bool,
    wants_in_flight_meals: bool,
    retrieved_sources: list[str],
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
    label, contradiction = resolve_label(accepted, was_edited, rating)
    if contradiction:
        logger.warning(
            "Contradictory feedback for %s — label: %s, rating: %d, flag: %s",
            customer_id, label, rating, contradiction,
        )
    lead = _build_lead_dict(
        customer_id, customer_name, booking_origin, haul_type,
        num_passengers, wants_extra_baggage, wants_preferred_seat,
        wants_in_flight_meals, retrieved_sources,
    )
    user_prompt_clean = build_user_prompt(lead)
    generated_completion = f"Subject: {generated_subject}\n\n{generated_body}"
    edited_completion = f"Subject: {edited_subject}\n\n{edited_body}"
    context_sources = list(set(retrieved_sources))
    base_meta = _build_meta(
        tokens_input, tokens_output, latency_ms,
        customer_id, was_edited, contradiction,
    )

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
    _save_feedback_record(feedback_log, full_record)

    if contradiction is None and label in (Label.ACCEPTED, Label.EDITED):
        _save_sft_record(sft_log, {
            "system_prompt_id": system_prompt_id,
            "prompt": user_prompt_clean,
            "completion": edited_completion,
            "label": label,
            "rating": rating,
            "meta": {
                "customer_id": customer_id,
                "was_edited": was_edited,
                "timestamp": base_meta["timestamp"],
            },
        })

    if contradiction is None and label == Label.EDITED:
        _save_dpo_record(dpo_log, {
            "system_prompt_id": system_prompt_id,
            "prompt": user_prompt_clean,
            "chosen": edited_completion,
            "rejected": generated_completion,
            "rating": rating,
            "meta": {
                "customer_id": customer_id,
                "timestamp": base_meta["timestamp"],
            },
        })

    logger.info(
        "Feedback saved for %s — label: %s, rating: %d, edited: %s",
        customer_id, label, rating, was_edited,
    )


def load_feedback(feedback_log: Path) -> list[dict]:
    if not feedback_log.exists():
        return []
    with open(feedback_log, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def feedback_stats(feedback_log: Path, sft_log: Path, dpo_log: Path) -> dict:
    records = load_feedback(feedback_log)
    if not records:
        return {"total": 0}
    ratings = [r["rating"] for r in records]
    edited = [r for r in records if r["meta"]["was_edited"]]
    contradictions = [r for r in records if r["meta"].get("contradiction")]
    sft_count = (
        sum(1 for _ in open(sft_log) if _.strip())
        if sft_log.exists() else 0
    )
    dpo_count = (
        sum(1 for _ in open(dpo_log) if _.strip())
        if dpo_log.exists() else 0
    )
    return {
        "total": len(records),
        "avg_rating": round(sum(ratings) / len(ratings), 2),
        "edited_count": len(edited),
        "edit_rate": round(len(edited) / len(records), 2),
        "contradiction_count": len(contradictions),
        "sft_pairs": sft_count,
        "dpo_pairs": dpo_count,
    }
