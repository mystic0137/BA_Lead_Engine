import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as rt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA,
    RF_CONFIG_PATH,
    RF_ONNX_PATH,
    XGBOOST_CONFIG_PATH,
    XGBOOST_ONNX_PATH,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_df = pd.read_csv(RAW_DATA, encoding="latin1").drop_duplicates()
_X = _df.drop("booking_complete", axis=1)
_y = _df["booking_complete"]
_, X_TEST, _, Y_TEST = train_test_split(
    _X, _y, test_size=0.2, random_state=42, stratify=_y
)


def verify_onnx_model(onnx_path: Path, config_path: Path) -> None:
    with open(config_path) as f:
        config = json.load(f)

    threshold: float = config["threshold"]
    expected_features: list[str] = config["expected_features"]
    numeric_features: list[str] = config["numeric_features"]

    X_test = X_TEST[expected_features].copy()
    X_test[numeric_features] = X_test[numeric_features].astype(np.float32)

    session = rt.InferenceSession(str(onnx_path))
    input_names = [inp.name for inp in session.get_inputs()]
    onnx_inputs = {name: X_test[[name]].values for name in input_names}
    onnx_outputs = session.run(None, onnx_inputs)

    raw = onnx_outputs[1]
    probs = raw[:, 1] if isinstance(raw, np.ndarray) else np.array([p[1] for p in raw])
    preds = (probs >= threshold).astype(int)

    logger.info("Verifying: %s | Threshold: %.4f", onnx_path.name, threshold)
    logger.info("ROC-AUC: %.4f", roc_auc_score(Y_TEST, probs))
    print("\nConfusion Matrix:")
    print(confusion_matrix(Y_TEST, preds))
    print("\nClassification Report:")
    print(classification_report(Y_TEST, peds))


if __name__ == "__main__":
    for onnx_path, config_path in [
        (RF_ONNX_PATH, RF_CONFIG_PATH),
        (XGBOOST_ONNX_PATH, XGBOOST_CONFIG_PATH),
    ]:
        verify_onnx_model(onnx_path, config_path)