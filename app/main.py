#main.py
import logging
import os
from contextlib import asynccontextmanager
from typing import List
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from core.state import ml
from src.rag.manager import RAGManager
from src.inference.csv_utils import csv_to_column_oriented
from app.schemas import (
    RoworientedInput, ColumnorientedInput,
    PredictionRoworiented, PredictionColumnoriented,
    RAGGenerateRequest, RAGGenerateResponse,
    RAGFeedbackRequest, RAGFeedbackResponse
)


rag = RAGManager()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml.initialize()
    logger.info("Inference engine loaded")
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
    allow_origins=os.getenv("CORS_ORIGINS", "http://127.0.0.1:8501,http://localhost:8501").split(","),
    allow_methods=["GET", "POST", "OPTIONS"],
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
    if not records:
        raise HTTPException(status_code=400, detail="Empty batch not allowed")
    try:
        result = await run_in_threadpool(
            ml.engine.run_row_oriented,
            [r.model_dump() for r in records]
        )
        return result
    except Exception:
        logger.exception("Row oriented inference failed at API")
        raise HTTPException(status_code=500, detail="Inference engine error")

@app.post("/predict/column_oriented_bench", response_model=PredictionColumnoriented)
async def predict_column_oriented(records: ColumnorientedInput):
    try:
        result = await run_in_threadpool(
            ml.engine.run_column_oriented,
            records.model_dump()
        )
        return result
    except Exception:
        logger.exception("Column oriented inference failed at API")
        raise HTTPException(status_code=500, detail="Inference engine error")
    
@app.post("/predict/column_oriented", response_model=PredictionColumnoriented)
async def predict_column_oriented(file: UploadFile = File(...)):
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=422, detail="Invalid file type. Only CSV file allowed.")
    
    try:
        records = await run_in_threadpool(csv_to_column_oriented, file.file)

        result = await run_in_threadpool(
            ml.engine.run_column_oriented,
            records.model_dump()
        )
        return result
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
        await run_in_threadpool(ml.initialize_rag)
        if not ml.is_rag_ready():
            raise HTTPException(status_code=503, detail="RAG components failed to initialize")
        result = await run_in_threadpool(rag.generate_email, request.model_dump())
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("RAG email generation failed")
        raise HTTPException(status_code=500, detail=f"Email generation error: {str(e)}")


@app.post("/api/v1/rag/feedback", response_model=RAGFeedbackResponse)
async def rag_feedback(request: RAGFeedbackRequest):
    try:
        await run_in_threadpool(rag.save_feedback, **request.model_dump())
        return RAGFeedbackResponse(
            status="success",
            message="Feedback saved successfully"
        )
    except Exception as e:
        logger.exception("RAG feedback save failed")
        raise HTTPException(status_code=500, detail=f"Feedback save error: {str(e)}")