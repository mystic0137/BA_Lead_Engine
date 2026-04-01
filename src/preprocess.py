import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

from src.config import NUMERIC_COLS, OHE_COLS, TARGET_ENCODE_COLS


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "target_enc",
                TargetEncoder(smooth=5, random_state=42),
                TARGET_ENCODE_COLS,
            ),
            (
                "ohe",
                OneHotEncoder(
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
                OHE_COLS,
            ),
        ],
        remainder="passthrough",
    )