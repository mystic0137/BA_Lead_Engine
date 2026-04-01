# src/models.py
import multiprocessing
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import ALL_FEATURE_COLS, NUMERIC_COLS, OHE_COLS, TARGET_ENCODE_COLS

__all__ = ["XGBClassifier", "RandomForestClassifier", "get_xgb_model", "get_rf_model"]

DEFAULT_CPUS = max(1, multiprocessing.cpu_count() - 1)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    n_estimators: int = Field(default=400, ge=1, le=2000)
    learning_rate: float = Field(default=0.01, gt=0, lt=1)
    max_depth: int = Field(default=6, ge=1, le=20)
    random_state: int = 42
    n_jobs: int = DEFAULT_CPUS

    categorical_features: List[str] = Field(
        default_factory=lambda: TARGET_ENCODE_COLS + OHE_COLS
    )
    numeric_features: List[str] = Field(default_factory=lambda: NUMERIC_COLS)
    expected_features: List[str] = Field(default_factory=lambda: ALL_FEATURE_COLS)

    @property
    def all_features(self) -> List[str]:
        return self.categorical_features + self.numeric_features


def validate_features(df_columns: List[str], config: ModelConfig) -> None:
    missing = set(config.all_features) - set(df_columns)
    if missing:
        raise ValueError(f"Input data missing mandatory columns: {missing}")


def get_rf_model(config: ModelConfig) -> RandomForestClassifier:
    hyperparams = config.model_dump(include={
        "n_estimators",
        "random_state",
        "n_jobs",
    })
    return RandomForestClassifier(class_weight="balanced", **hyperparams)


def get_xgb_model(config: ModelConfig) -> XGBClassifier:
    hyperparams = config.model_dump(include={
        "n_estimators",
        "learning_rate",
        "max_depth",
        "random_state",
        "n_jobs",
    })
    return XGBClassifier(eval_metric="auc", **hyperparams)