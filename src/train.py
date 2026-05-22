import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from skl2onnx import to_onnx, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType, StringTensorType
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost as xgb_converter

from src.config import (
    DEFAULT_THRESHOLD,
    ONNX_OPSET,
    RAW_DATA,
    MODELS_DIR,
    XGBOOST_CONFIG_PATH,
    XGBOOST_ONNX_PATH,
    init_dirs,
)
from src.models import ModelConfig, XGBClassifier, get_xgb_model, validate_features
from src.preprocess import build_preprocessor
from src.data_check import verify_integrity, DataIntegrityError

logger = logging.getLogger(__name__)

os.makedirs(MODELS_DIR, exist_ok=True)

def _xgb_shape_calculator(operator) -> None:
    """
    Replaces onnxconverter_common.calculate_linear_classifier_output_shapes.
    That function isinstance-checks against its own FloatTensorType, which
    differs from skl2onnx's — causing RuntimeError in the conversion pipeline.
    """
    N = operator.inputs[0].type.shape[0]
    operator.outputs[0].type = Int64TensorType(shape=[N])
    operator.outputs[1].type = FloatTensorType(shape=[N, 2])


update_registered_converter(
    XGBClassifier,
    "XGBoostXGBClassifier",
    _xgb_shape_calculator,
    xgb_converter,
)


def export_to_onnx(pipeline: Pipeline, X_sample: pd.DataFrame, save_path: Path) -> None:
    initial_type = [
        (col, StringTensorType([None, 1]) if X_sample[col].dtype == object else FloatTensorType([None, 1]))
        for col in X_sample.columns
    ]
    try:
        onx = to_onnx(pipeline, initial_types=initial_type, target_opset=ONNX_OPSET)
        save_path.write_bytes(onx.SerializeToString())
    except Exception:
        logger.exception("ONNX conversion failed for %s", save_path)
        raise


def train(threshold: float = DEFAULT_THRESHOLD) -> None:
    init_dirs()

    try:
        verify_integrity()
    except DataIntegrityError as e:
        logger.error(str(e))
        raise

    logger.info("Data Integrity Verified. Loading Data...")

    df = pd.read_csv(RAW_DATA, encoding="latin1").drop_duplicates()
    X = df.drop("booking_complete", axis=1)
    y = df["booking_complete"]

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    X[numeric_cols] = X[numeric_cols].astype(np.float32)

    config = ModelConfig()
    validate_features(list(X.columns), config)

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    spw = (len(y_train) - y_train.sum()) / y_train.sum()
    X_sample = X_train.iloc[:1]

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("xgb_classifier", get_xgb_model(config).set_params(scale_pos_weight=spw)),
    ])

    logger.info("Training XGBoost (%d estimators)", config.n_estimators)
    pipeline.fit(X_train, y_train)
    export_to_onnx(pipeline, X_sample, XGBOOST_ONNX_PATH)

    full_config = config.model_dump()
    full_config["threshold"] = threshold
    full_config["model_type"] = "XGB"
    with open(XGBOOST_CONFIG_PATH, "w") as f:
        json.dump(full_config, f, indent=4)

    logger.info("Training complete. Models and configs saved.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()