from pathlib import Path
from src.config import (
    EXPORT_FIELDS, ROOT_DIR, DATA_DIR, MODELS_DIR,
    RAW_DATA, CHROMA_DB_PATH, POLICIES_DIR, FINETUNING_DIR,
    TARGET_ENCODE_COLS, OHE_COLS, NUMERIC_COLS, ALL_FEATURE_COLS,
    XGBOOST_ONNX_PATH, XGBOOST_CONFIG_PATH, RF_ONNX_PATH, RF_CONFIG_PATH,
    DEFAULT_THRESHOLD, ONNX_OPSET, EMBEDDING_MODEL,
    SYSTEM_PROMPTS, ACTIVE_SYSTEM_PROMPT_ID, init_dirs,
)


def test_root_dir_is_parent():
    assert ROOT_DIR == Path(__file__).resolve().parent.parent


def test_data_dir():
    assert DATA_DIR == ROOT_DIR / "data"


def test_models_dir():
    assert MODELS_DIR == ROOT_DIR / "models"


def test_raw_data_path():
    assert RAW_DATA == DATA_DIR / "raw" / "customer_booking.csv"


def test_chroma_db_path():
    assert CHROMA_DB_PATH == DATA_DIR / "chroma_db"


def test_policies_dir():
    assert POLICIES_DIR == DATA_DIR / "policies"


def test_finetuning_dir():
    assert FINETUNING_DIR == DATA_DIR / "finetuning"


def test_export_fields():
    assert "customer_id" in EXPORT_FIELDS
    assert "customer_name" in EXPORT_FIELDS
    assert "email" in EXPORT_FIELDS
    assert "route" in EXPORT_FIELDS
    assert "haul_type" in EXPORT_FIELDS


def test_feature_column_lists():
    assert "route" in TARGET_ENCODE_COLS
    assert "booking_origin" in TARGET_ENCODE_COLS
    assert "sales_channel" in OHE_COLS
    assert "trip_type" in OHE_COLS
    assert "flight_day" in OHE_COLS
    assert "num_passengers" in NUMERIC_COLS
    assert "purchase_lead" in NUMERIC_COLS
    assert "flight_hour" in NUMERIC_COLS


def test_all_feature_cols_is_union():
    expected = TARGET_ENCODE_COLS + OHE_COLS + NUMERIC_COLS
    assert ALL_FEATURE_COLS == expected


def test_model_paths():
    assert XGBOOST_ONNX_PATH == MODELS_DIR / "xgboost.onnx"
    assert XGBOOST_CONFIG_PATH == MODELS_DIR / "xgboost_config.json"
    assert RF_ONNX_PATH == MODELS_DIR / "random_forest.onnx"
    assert RF_CONFIG_PATH == MODELS_DIR / "random_forest_config.json"


def test_default_threshold():
    assert DEFAULT_THRESHOLD == 0.3090


def test_onnx_opset():
    assert ONNX_OPSET == {"": 17, "ai.onnx.ml": 3}


def test_embedding_model_path():
    assert EMBEDDING_MODEL == ROOT_DIR / "hf_models/all-MiniLM-L6-v2"


def test_system_prompts():
    assert "ba_copywriter_v1" in SYSTEM_PROMPTS
    assert len(SYSTEM_PROMPTS["ba_copywriter_v1"]) > 100


def test_active_system_prompt_id():
    assert ACTIVE_SYSTEM_PROMPT_ID == "ba_copywriter_v1"


def test_init_dirs_creates_directories(tmp_path):
    import src.config as config
    original_root = config.ROOT_DIR
    config.ROOT_DIR = tmp_path
    config.DATA_DIR = tmp_path / "data"
    config.MODELS_DIR = tmp_path / "models"
    config.CHROMA_DB_PATH = tmp_path / "data" / "chroma_db"
    config.POLICIES_DIR = tmp_path / "data" / "policies"
    config.FINETUNING_DIR = tmp_path / "data" / "finetuning"
    try:
        init_dirs()
        assert (tmp_path / "data" / "raw").exists()
        assert (tmp_path / "models").exists()
        assert (tmp_path / "data" / "chroma_db").exists()
        assert (tmp_path / "data" / "policies").exists()
        assert (tmp_path / "data" / "finetuning").exists()
    finally:
        config.ROOT_DIR = original_root
        config.DATA_DIR = original_root / "data"
        config.MODELS_DIR = original_root / "models"
        config.CHROMA_DB_PATH = original_root / "data" / "chroma_db"
        config.POLICIES_DIR = original_root / "data" / "policies"
        config.FINETUNING_DIR = original_root / "data" / "finetuning"
