import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def valid_payload() -> dict:
    return {
        "num_passengers": 2,
        "sales_channel": "Internet",
        "trip_type": "RoundTrip",
        "purchase_lead": 13,
        "length_of_stay": 7,
        "flight_hour": 10,
        "flight_day": "Sat",
        "route": "AKLHND",
        "booking_origin": "Australia",
        "wants_extra_baggage": 1,
        "wants_preferred_seat": 0,
        "wants_in_flight_meals": 1,
        "flight_duration": 8.5,
    }


def test_prediction_success(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()

    assert 0.0 <= data["probability"] <= 1.0
    assert data["booking_prediction"] in [0, 1]
    assert "business_logic" in data
    assert data["business_logic"]["category"] in [
        "Category 0", "Category 1", "Category 2", "Category 3"
    ]
    assert data["business_logic"]["priority_score"] >= 0


def test_prediction_invalid_hour(client, valid_payload):
    bad_payload = {**valid_payload, "flight_hour": 25}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
    assert "flight_hour" in response.text


def test_prediction_missing_field(client, valid_payload):
    bad_payload = {k: v for k, v in valid_payload.items() if k != "route"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_prediction_does_not_crash_on_extreme_values(client, valid_payload):
    extreme_payload = {
        **valid_payload,
        "wants_extra_baggage": 1,
        "wants_preferred_seat": 1,
        "wants_in_flight_meals": 1,
        "flight_duration": 12.0,
    }
    response = client.post("/predict", json=extreme_payload)
    assert response.status_code == 200