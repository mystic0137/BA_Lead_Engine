import io
import os
import random
import uuid
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

from config import EXPORT_FIELDS

FAKE_NAMES = [
    "James Thornton", "Priya Nair", "Oliver Bennett", "Aisha Rahman",
    "Lucas Ferreira", "Sophie Müller", "Chen Wei", "Fatima Al-Amin",
    "Carlos Mendez", "Emma Larsson", "Ravi Shankar", "Nina Petrova",
    "David Okafor", "Yuki Tanaka", "Maria Costa", "Ahmed Hassan",
]

PRIORITY_META = {
    3: ("The Persuadable", "Schedule Call", "#e74c3c"),
    2: ("The VIP",         "Send Email",    "#2ecc71"),
    1: ("The Window Shopper", "Drip Sequence", "#f39c12"),
    0: ("The Lost Cause",  "No Action",     "#95a5a6"),
}

SEGMENT_COLORS = {v[0]: v[2] for v in PRIORITY_META.values()}

REQUIRED_COLUMNS = [
    "num_passengers", "sales_channel", "trip_type", "purchase_lead",
    "length_of_stay", "flight_hour", "flight_day", "route",
    "booking_origin", "wants_extra_baggage", "wants_preferred_seat",
    "wants_in_flight_meals", "flight_duration",
]

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def generate_fake_contact(name: str) -> dict:
    email = name.lower().replace(" ", ".") + "@example.com"
    phone = f"+1{random.randint(200,999)}{random.randint(100,999)}{random.randint(1000,9999)}"
    return {"email": email, "phone": phone}


def check_backend() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json().get("model_loaded", False)
    except requests.exceptions.ConnectionError:
        return False


def _haul_type(duration: float) -> str:
    if duration < 3:
        return "Short Haul"
    elif duration <= 6:
        return "Medium Haul"
    return "Long Haul"


def score_csv(file) -> dict | None:
    try:
        r = requests.post(
            f"{API_URL}/predict/column_oriented",
            files={"file": (file.name, file.getvalue(), "text/csv")},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Scoring request failed: {e}")
        return None


def build_results(df: pd.DataFrame, response: dict) -> pd.DataFrame:
    pred = response["predictions"]
    biz = pred["business_logic"]
    rows = []
    for i in range(len(pred["probability"])):
        priority = int(biz["priority_score"][i])
        segment, action, _ = PRIORITY_META[priority]
        name = FAKE_NAMES[i % len(FAKE_NAMES)]
        contact = generate_fake_contact(name)
        row = df.iloc[i]
        rows.append({
            "customer_id": str(uuid.uuid4())[:8].upper(),
            "customer_name": name,
            "email": contact["email"],
            "phone": contact["phone"],
            "route": row["route"],
            "booking_origin": row["booking_origin"],
            "haul_type": _haul_type(row["flight_duration"]),
            "num_passengers": row["num_passengers"],
            "wants_extra_baggage": bool(row["wants_extra_baggage"]),
            "wants_preferred_seat": bool(row["wants_preferred_seat"]),
            "wants_in_flight_meals": bool(row["wants_in_flight_meals"]),
            "booking_probability": pred["probability"][i],
            "booking_prediction": pred["booking_prediction"][i],
            "priority_score": priority,
            "value_tier": biz["value_tier"][i],
            "expected_value_usd": biz["expected_value_usd"][i],
            "potential_revenue_usd": biz["potential_revenue_usd"][i],
            "marginal_profit_usd": biz["marginal_profit_usd"][i],
            "segment": segment,
            "recommended_action": action,
            "scored_at": datetime.utcnow().isoformat(),
        })
    return pd.DataFrame(rows).sort_values("priority_score", ascending=False)
