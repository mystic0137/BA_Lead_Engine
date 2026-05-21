#main.py
import logging
import time
from contextlib import asynccontextmanager
from typing import List
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from app.state import ml
from app.schemas import (
    RoworientedInput, ColumnorientedInput,
    PredictionRoworiented, PredictionColumnoriented,
    RAGGenerateRequest, RAGGenerateResponse,
    RAGFeedbackRequest, RAGFeedbackResponse
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml.initialize()
    logger.info("Model loaded")
    yield
    ml.clear()


app = FastAPI(
    title="British Airways Lead Priority API",
    description="Real-time booking propensity and lead valuation",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict/single", response_model=PredictionRoworiented)
async def predict_single(data: RoworientedInput):
    try:
        result = await run_in_threadpool(
            ml.engine.run_row_oriented,
            [data.model_dump()]
        )
        return result
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference engine error")


@app.post("/predict/row_oriented", response_model=PredictionRoworiented)
async def predict_row_oriented(records: List[RoworientedInput]):
    try:
        return ml.engine.run_row_oriented([r.model_dump() for r in records])
    except Exception:
        logger.exception("Row oriented inference failed at API")
        raise HTTPException(status_code=500, detail="Inference engine error")

@app.post("/predict/column_oriented_bench", response_model=PredictionColumnoriented)
async def predict_column_oriented(records: ColumnorientedInput):
    try:
        return ml.engine.run_column_oriented(records.model_dump())
    except Exception:
        logger.exception("Row oriented inference failed at API")
        raise HTTPException(status_code=500, detail="Inference engine error")
    
@app.post("/predict/column_oriented", response_model=PredictionColumnoriented)
async def predict_column_oriented(file: UploadFile = File(...)):
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=500, detail="Invalid file type. Only CSV file allowed.")
    
    try:
        records = ml.engine.csv_to_column_oriented(file.file)

        return ml.engine.run_column_oriented(records.model_dump())
    except ValidationError as val_error:
        logger.warning(f"CSV data format validation failed: {val_error.json()}")
        raise HTTPException(status_code=422, detail=val_error.errors())
    except Exception:
        logger.exception("Column oriented inference failed at API")
        raise HTTPException(status_code=500, detail="Inference engine error")
    
@app.get("/health")
def health_check():
    loaded = ml.is_ready()
    return {
        "status": "healthy" if loaded else "unhealthy",
        "model_loaded": loaded,
        "api_version": "0.1.0",
    }


@app.post("/api/v1/rag/generate", response_model=RAGGenerateResponse)
async def rag_generate(request: RAGGenerateRequest):
    try:
        if not ml.is_rag_ready():
            raise HTTPException(status_code=503, detail="RAG components not initialized")
        
        result = await run_in_threadpool(_generate_email_internal, request.model_dump())
        return result
    except Exception as e:
        logger.exception("RAG email generation failed")
        raise HTTPException(status_code=500, detail=f"Email generation error: {str(e)}")


@app.post("/api/v1/rag/feedback", response_model=RAGFeedbackResponse)
async def rag_feedback(request: RAGFeedbackRequest):
    try:
        await run_in_threadpool(_save_feedback_internal, request.model_dump())
        return RAGFeedbackResponse(
            status="success",
            message="Feedback saved successfully"
        )
    except Exception as e:
        logger.exception("RAG feedback save failed")
        raise HTTPException(status_code=500, detail=f"Feedback save error: {str(e)}")


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


def _retrieve(query: str, top_k: int = 3) -> list[dict]:
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

    return chunks


def _build_user_prompt(lead: dict, policy_context: str) -> str:
    from src.rag.mappings import (
        HAUL_HOOK_MAP, SEGMENT_MAP,
        get_amenity_hooks, get_party_description
    )
    
    first_name = lead["customer_name"].split()[0]
    origin_city = lead["booking_origin"]
    hook = HAUL_HOOK_MAP.get(lead["haul_type"], "")
    party_desc = get_party_description(lead["num_passengers"])
    segment_desc = SEGMENT_MAP.get(lead.get("segment", ""), "valued traveler")
    amenity_hooks = get_amenity_hooks(lead)

    amenity_narrative = (
        "They have expressed interest in: " + ", ".join(amenity_hooks) + "."
        if amenity_hooks
        else "They have not selected specific add-ons yet — focus on the journey experience."
    )

    return (
        f"Draft a bespoke outreach email for {first_name}, who is traveling from {origin_city}.\n\n"
        f"--- NARRATIVE CONTEXT ---\n"
        f"This traveler is {segment_desc} and is planning a {party_desc}. "
        f"The tone of this outreach should be: {hook}\n\n"
        f"--- FOCUS AREAS ---\n"
        f"{amenity_narrative}\n\n"
        f"--- VERIFIED POLICY CONTEXT ---\n"
        f"{policy_context}\n\n"
        f"--- FINAL INSTRUCTION ---\n"
        f"Address {first_name} by first name. Blend the context and focus areas into a single, "
        f"sophisticated story. Do not use headers or bullet points in the email body. "
    )


def _get_system_prompt() -> str:
    from src.config import ACTIVE_SYSTEM_PROMPT_ID, SYSTEM_PROMPTS
    return SYSTEM_PROMPTS[ACTIVE_SYSTEM_PROMPT_ID]


def _generate_email_internal(lead: dict) -> dict:
    query = _build_query(lead)
    chunks = _retrieve(query)
    policy_context = _format_chunks(chunks)

    system_prompt = _get_system_prompt()
    user_prompt = _build_user_prompt(lead, policy_context)

    t0 = time.time()
    response = ml.groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=600,
        temperature=0.6,
    )
    latency_ms = round((time.time() - t0) * 1000)

    raw = response.choices[0].message.content.strip()
    subject, body = _parse_email(raw)

    from src.config import ACTIVE_SYSTEM_PROMPT_ID
    
    return {
        "subject": subject,
        "body": body,
        "retrieved_sources": [c["source"] for c in chunks],
        "system_prompt_id": ACTIVE_SYSTEM_PROMPT_ID,
        "tokens_input": response.usage.prompt_tokens,
        "tokens_output": response.usage.completion_tokens,
        "latency_ms": latency_ms,
    }


def _save_feedback_internal(data: dict) -> None:
    import json
    from datetime import datetime
    from pathlib import Path
    from src.config import FINETUNING_DIR
    from src.rag.mappings import get_amenity_hooks, get_party_description
    
    FINETUNING_DIR.mkdir(parents=True, exist_ok=True)
    
    FEEDBACK_LOG = FINETUNING_DIR / "feedback_log.jsonl"
    SFT_LOG = FINETUNING_DIR / "sft_log.jsonl"
    DPO_LOG = FINETUNING_DIR / "dpo_log.jsonl"
    
    was_edited = (
        data["edited_subject"].strip() != data["generated_subject"].strip()
        or data["edited_body"].strip() != data["generated_body"].strip()
    )
    
    accepted = data.get("accepted")
    rating = data["rating"]
    
    if accepted is None:
        label = "neutral"
        contradiction = None
    elif accepted and not was_edited:
        if rating < 3:
            label = "accepted"
            contradiction = "low_rating_but_accepted"
        else:
            label = "accepted"
            contradiction = None
    elif accepted and was_edited:
        label = "edited"
        contradiction = None
    else:
        if rating >= 4:
            label = "rejected"
            contradiction = "high_rating_but_rejected"
        else:
            label = "rejected"
            contradiction = None
    
    if contradiction:
        logger.warning(
            "Contradictory feedback for %s — label: %s, rating: %d, flag: %s",
            data.get("customer_id"), label, rating, contradiction,
        )
    
    user_prompt_clean = _build_user_prompt(data, "[CONTEXT_PLACEHOLDER]")
    generated_completion = f"Subject: {data['generated_subject']}\n\n{data['generated_body']}"
    edited_completion = f"Subject: {data['edited_subject']}\n\n{data['edited_body']}"
    context_sources = list(set(data.get("retrieved_sources", [])))
    
    base_meta = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "tokens_input": data.get("tokens_input", 0),
        "tokens_output": data.get("tokens_output", 0),
        "latency_ms": data.get("latency_ms", 0),
        "timestamp": datetime.utcnow().isoformat(),
        "customer_id": data.get("customer_id"),
        "was_edited": was_edited,
        "contradiction": contradiction,
    }
    
    full_record = {
        "system_prompt_id": data["system_prompt_id"],
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
    
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(full_record) + "\n")
    
    if label in ("accepted", "edited") and contradiction is None:
        sft_record = {
            "system_prompt_id": data["system_prompt_id"],
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
        with open(SFT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(sft_record) + "\n")
        logger.info("SFT record saved for %s", data.get("customer_id"))
    
    if label == "edited" and contradiction is None:
        dpo_record = {
            "system_prompt_id": data["system_prompt_id"],
            "prompt": user_prompt_clean,
            "chosen": edited_completion,
            "rejected": generated_completion,
            "rating": rating,
            "meta": {
                "customer_id": base_meta["customer_id"],
                "timestamp": base_meta["timestamp"],
            },
        }
        with open(DPO_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(dpo_record) + "\n")
        logger.info("DPO record saved for %s", data.get("customer_id"))
    
    logger.info(
        "Feedback saved for %s — label: %s, rating: %d, edited: %s",
        data.get("customer_id"), label, rating, was_edited,
    )