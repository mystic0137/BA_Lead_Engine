import logging
import onnxruntime as rt
import pandas as pd
import numpy as np
import json
from typing import BinaryIO

from app.schemas import RoworientedInput, ColumnorientedInput, PredictionRoworiented, PredictionColumnoriented
from src.config import XGBOOST_ONNX_PATH, XGBOOST_CONFIG_PATH
from src.analytics.finance import BACostCalculator


#Helper
def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return json.load(f)

logger = logging.getLogger(__name__)

class InferenceEngine:
    def __init__(self):
        
        self._session = rt.InferenceSession(
            str(XGBOOST_ONNX_PATH),
            providers=["CPUExecutionProvider"]
        )
        self._calculator = BACostCalculator()
        config = load_config(XGBOOST_CONFIG_PATH)
        self._threshold = config["threshold"]

        self._all_features = [inp.name for inp in self._session.get_inputs()]
        self._numeric_features = [inp.name for inp in self._session.get_inputs() if "string" not in inp.type]
    
    @staticmethod
    def csv_to_column_oriented(file_stream: BinaryIO) -> ColumnorientedInput:

        df = pd.read_csv(file_stream, encoding="latin1")

        records = df.to_dict(orient="list")

        return ColumnorientedInput(
            **records
        )
    
    def _run_core(self, onnx_inputs: dict) -> np.ndarray:

        try:
            output = self._session.run(None, onnx_inputs)
        except Exception:
            logger.exception("ONNX session.run failed")
            raise

        raw_probs = output[1]
        probs = np.ascontiguousarray(raw_probs[:,1]).reshape(-1, 1)

        return probs
    
    def run_row_oriented(self, records: RoworientedInput) -> PredictionRoworiented:
        all_features = self._all_features
        numeric_features = self._numeric_features
        session = self._session
        calculator = self._calculator

        df = pd.DataFrame(records)[all_features]
        df[numeric_features] = df[numeric_features].astype(np.float32)

        onnx_inputs = {
            inp.name: df[inp.name].to_numpy().reshape(-1, 1)
            for inp in session.get_inputs()
        }

        probs = self._run_core(onnx_inputs).reshape(-1,)

        results = []
        for prob, record in zip(probs, records):
            valuation = calculator.calculate_lead_value(prob, record)

            results.append({
                "probability": float(prob),
                "booking_prediction": int(prob > self._threshold),
                "business_logic": valuation
            })
        
        final = {
            "predictions": results,
            "meta": {
                "model_version": "xgboost_onnx_v1",
                "threshold_used": self._threshold
            }
        }
        return final

    def run_column_oriented(self, records: ColumnorientedInput) -> PredictionColumnoriented:
        session = self._session
        threshold = self._threshold
        calculator = self._calculator
        onnx_inputs = {}
        
        for inp in session.get_inputs():

            if "string" in inp.type:
                onnx_inputs[inp.name] = np.array(records[inp.name]).astype(str).reshape(-1, 1)
            else:
                onnx_inputs[inp.name] = np.array(records[inp.name]).astype(np.float32).reshape(-1, 1)
        
        probs = self._run_core(onnx_inputs)
        
        classes = (
            probs > threshold
        ).astype(np.int8)
        
        valuation = calculator.vectorized_calculate_lead_value(probs, onnx_inputs)
        
        results = {
            "probability": np.round(probs, 4).flatten().tolist(),
            "booking_prediction": classes.flatten().tolist(),
            "business_logic": {
                key: valuation[key].flatten().tolist()
                for key in valuation
            }
        }
        
        final = {
            "predictions": results,
            "meta": {
                "model_version": "xgboost_onnx_v1",
                "threshold_used": threshold
            }
        }
        
        return final