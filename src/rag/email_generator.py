import logging
import time

from groq import Groq

from src.config import ACTIVE_SYSTEM_PROMPT_ID, settings
from src.rag.prompt_builder import build_user_prompt, get_system_prompt
from src.rag.retriever import retrieve

logger = logging.getLogger(__name__)

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_TOKENS = 600
api_key = settings.GROQ_API_KEY.get_secret_value()

client = Groq(api_key=api_key)


def _build_query(lead: dict) -> str:
    parts = [f"{lead['haul_type']} flight policy"]
    if lead.get("wants_extra_baggage"):
        parts.append("baggage allowance policy")
    if lead.get("wants_preferred_seat"):
        parts.append("seat selection policy")
    if lead.get("wants_in_flight_meals"):
        parts.append("meal service policy")
    return " ".join(parts)


def _format_chunks(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )


def _parse_email(raw: str) -> tuple[str, str]:
    lines = raw.splitlines()
    subject = ""
    body_lines = []
    subject_line_idx = -1

    for i, line in enumerate(lines):
        cleaned = line.strip().lstrip("*#").strip()
        if cleaned.lower().startswith("subject:"):
            subject = cleaned.split(":", 1)[-1].strip().strip("*").strip()
            subject_line_idx = i
            break

    if subject_line_idx >= 0:
        start = subject_line_idx + 1
        while start < len(lines) and lines[start].strip() == "":
            start += 1
        body_lines = lines[start:]
    else:
        stripped = raw.strip()
        first_sentence_end = next(
            (i for i, c in enumerate(stripped) if c in ".!?"), len(stripped)
        )
        subject = stripped[:first_sentence_end + 1].strip()
        body_lines = stripped[first_sentence_end + 1:].strip().splitlines()

    body = "\n".join(body_lines).strip()

    if not body:
        logger.warning("Email body parsing failed. Raw output: %s", raw)
        body = raw.strip()

    return subject or "A journey worth taking — British Airways", body


def generate_email(lead: dict) -> dict:
    query = _build_query(lead)
    chunks = retrieve(query)
    policy_context = _format_chunks(chunks)

    system_prompt = get_system_prompt()
    user_prompt = build_user_prompt(lead, policy_context=policy_context)

    t0 = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.6,
    )
    latency_ms = round((time.time() - t0) * 1000)

    raw = response.choices[0].message.content.strip()
    subject, body = _parse_email(raw)

    return {
        "subject": subject,
        "body": body,
        "retrieved_sources": [c["source"] for c in chunks],
        "system_prompt_id": ACTIVE_SYSTEM_PROMPT_ID,
        "tokens_input": response.usage.prompt_tokens,
        "tokens_output": response.usage.completion_tokens,
        "latency_ms": latency_ms,
    }