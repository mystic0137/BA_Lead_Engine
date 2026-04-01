import json
import logging
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import pandas as pd
import onnxruntime as rt
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import BookingInput, PredictionResponse
from src.analytics.finance import BACostCalculator
from src.config import XGBOOST_CONFIG_PATH, XGBOOST_ONNX_PATH

logger = logging.getLogger(__name__)

ml: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(XGBOOST_CONFIG_PATH) as f:
        ml["config"] = json.load(f)
    ml["session"] = rt.InferenceSession(
        str(XGBOOST_ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )
    ml["calculator"] = BACostCalculator()
    logger.info("Model loaded from %s", XGBOOST_ONNX_PATH)
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


def _run_inference_batch(records: list[dict]) -> list[dict]:
    config = ml["config"]
    session = ml["session"]

    expected_features: list[str] = config["expected_features"]
    numeric_features: list[str] = config["numeric_features"]
    threshold: float = config["threshold"]

    df = pd.DataFrame(records)[expected_features]
    df[numeric_features] = df[numeric_features].astype(np.float32)

    onnx_inputs = {
        inp.name: df[[inp.name]].values
        for inp in session.get_inputs()
    }

    try:
        outputs = session.run(None, onnx_inputs)
    except Exception:
        logger.exception("ONNX session.run failed")
        raise

    raw_probs = outputs[1]
    probs = raw_probs[:, 1] if isinstance(raw_probs, np.ndarray) else np.array([p[1] for p in raw_probs])

    results = []
    for prob_val, record in zip(probs, records):
        prob_val = float(prob_val)
        valuation = ml["calculator"].calculate_lead_value(prob_val, record)
        results.append({
            "probability": round(prob_val, 4),
            "booking_prediction": int(prob_val >= threshold),
            "business_logic": {
                "segment": valuation["segment"],
                "category": valuation["category"],
                "recommended_action": valuation["recommended_action"],
                "value_tier": valuation["value_tier"],
                "expected_value_usd": valuation["expected_value"],
                "potential_revenue_usd": valuation["potential_revenue"],
                "marginal_profit_usd": valuation["marginal_profit"],
                "priority_score": valuation["priority_score"],
            },
            "meta": {
                "model_version": "xgboost_v1_onnx",
                "threshold_used": threshold,
            },
        })
    return results


@app.post("/predict", response_model=PredictionResponse)
async def predict(data: BookingInput):
    try:
        return _run_inference_batch([data.model_dump()])[0]
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference engine error")


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(records: List[BookingInput]):
    try:
        return _run_inference_batch([r.model_dump() for r in records])
    except Exception:
        logger.exception("Batch inference failed")
        raise HTTPException(status_code=500, detail="Inference engine error")


@app.get("/health")
def health_check():
    loaded = bool(ml.get("session"))
    return {
        "status": "healthy" if loaded else "unhealthy",
        "model_loaded": loaded,
        "api_version": "0.1.0",
    }