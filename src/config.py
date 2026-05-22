# src/config.py
from pathlib import Path
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

EXPORT_FIELDS = [
    "customer_id",
    "customer_name", 
    "email",
    "phone",
    "num_passengers",
    "route",
    "booking_origin",
    "haul_type",
    "wants_extra_baggage",
    "wants_preferred_seat",
    "wants_in_flight_meals",
]

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

EXPECTED_DATA_HASH = "f7108ec998528b72a0624a2d137521be99486e87ebe93a018b7fcaa6fdc3b4d2"


RAW_DATA = DATA_DIR / "raw" / "customer_booking.csv"
CHROMA_DB_PATH = DATA_DIR / "chroma_db"
POLICIES_DIR = DATA_DIR / "policies"
FINETUNING_DIR = DATA_DIR / "finetuning"

TARGET_ENCODE_COLS = ["route", "booking_origin"]
OHE_COLS = ["sales_channel", "trip_type", "flight_day"]
NUMERIC_COLS = [
    "num_passengers", "purchase_lead", "length_of_stay",
    "flight_duration", "wants_extra_baggage",
    "wants_preferred_seat", "wants_in_flight_meals",
    "flight_hour",
]
ALL_FEATURE_COLS = TARGET_ENCODE_COLS + OHE_COLS + NUMERIC_COLS

XGBOOST_ONNX_PATH = MODELS_DIR / "xgboost.onnx"
XGBOOST_CONFIG_PATH = MODELS_DIR / "xgboost_config.json"
RF_ONNX_PATH = MODELS_DIR / "random_forest.onnx"
RF_CONFIG_PATH = MODELS_DIR / "random_forest_config.json"

DEFAULT_THRESHOLD = 0.3090
ONNX_OPSET: dict[str, int] = {"": 17, "ai.onnx.ml": 3}

EMBEDDING_MODEL = ROOT_DIR / "hf_models/all-MiniLM-L6-v2"

SYSTEM_PROMPTS: dict[str, str] = {
    "ba_copywriter_v1": (
        "You are an elegant and observant brand copywriter for British Airways. "
        "Your voice is warm, sophisticated, and distinctly British—never pushy or corporate. "
        "You excel at weaving specific traveler needs into a seamless, inspiring narrative. "
        "\n\nSTRICT RULES:\n"
        "1. Only reference services (baggage, seating, meals) explicitly found in the provided Policy Context.\n"
        "2. Never use internal jargon (e.g., 'The VIP', 'Long Haul', 'Segment').\n"
        "3. Never mention prices, probabilities, or data that suggests the user is in a database.\n"
        "4. Maximum 180 words. Focus on the 'feeling' of the journey."
    )
}

ACTIVE_SYSTEM_PROMPT_ID = "ba_copywriter_v1"

class Settings(BaseSettings):

    GROQ_API_KEY: SecretStr

    DEBUG_MODE: bool = False

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding='utf-8',
        extra='ignore'
    )

settings=Settings()

def init_dirs() -> None:
    (DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    POLICIES_DIR.mkdir(parents=True, exist_ok=True)
    FINETUNING_DIR.mkdir(parents=True, exist_ok=True)