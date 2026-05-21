import io
import json
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def valid_payload():
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


class TestHealthEndpoint:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "api_version" in data

    def test_health_returns_healthy(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"


class TestPredictSingle:
    def test_prediction_success(self, client, valid_payload):
        response = client.post("/predict/single", json=valid_payload)
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 1
        pred = data["predictions"][0]
        assert 0.0 <= pred["probability"] <= 1.0
        assert pred["booking_prediction"] in [0, 1]
        assert "business_logic" in pred
        assert pred["business_logic"]["value_tier"] in ["High", "Medium", "Low"]
        assert pred["business_logic"]["priority_score"] in [0, 1, 2, 3]
        assert "meta" in data

    def test_prediction_invalid_hour(self, client, valid_payload):
        bad_payload = {**valid_payload, "flight_hour": 25}
        response = client.post("/predict/single", json=bad_payload)
        assert response.status_code == 422

    def test_prediction_missing_field(self, client, valid_payload):
        bad_payload = {k: v for k, v in valid_payload.items() if k != "route"}
        response = client.post("/predict/single", json=bad_payload)
        assert response.status_code == 422

    def test_prediction_extreme_values(self, client, valid_payload):
        extreme = {
            **valid_payload,
            "wants_extra_baggage": 1,
            "wants_preferred_seat": 1,
            "wants_in_flight_meals": 1,
            "flight_duration": 12.0,
        }
        response = client.post("/predict/single", json=extreme)
        assert response.status_code == 200

    def test_prediction_edge_values(self, client, valid_payload):
        edge = {
            **valid_payload,
            "num_passengers": 1,
            "purchase_lead": 0,
            "length_of_stay": 0,
            "flight_hour": 0,
            "flight_duration": 0.1,
        }
        response = client.post("/predict/single", json=edge)
        assert response.status_code == 200

    def test_prediction_max_values(self, client, valid_payload):
        maxed = {
            **valid_payload,
            "num_passengers": 20,
            "purchase_lead": 1000,
            "length_of_stay": 1000,
            "flight_hour": 23,
            "flight_duration": 24.0,
        }
        response = client.post("/predict/single", json=maxed)
        assert response.status_code == 200


class TestPredictRowOriented:
    def test_batch_prediction(self, client, valid_payload):
        records = [valid_payload, {**valid_payload, "num_passengers": 1, "flight_hour": 15}]
        response = client.post("/predict/row_oriented", json=records)
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 2

    def test_empty_batch(self, client):
        response = client.post("/predict/row_oriented", json=[])
        assert response.status_code == 500

    def test_invalid_record_in_batch(self, client, valid_payload):
        records = [valid_payload, {**valid_payload, "flight_hour": 99}]
        response = client.post("/predict/row_oriented", json=records)
        assert response.status_code == 422


class TestPredictColumnOriented:
    def test_column_oriented_json(self, client):
        payload = {
            "num_passengers": [2, 1],
            "purchase_lead": [13, 5],
            "length_of_stay": [7, 3],
            "flight_hour": [10, 14],
            "flight_duration": [8.5, 5.0],
            "wants_extra_baggage": [1, 0],
            "wants_preferred_seat": [0, 1],
            "wants_in_flight_meals": [1, 0],
            "sales_channel": ["Internet", "Mobile"],
            "trip_type": ["RoundTrip", "OneWay"],
            "flight_day": ["Sat", "Mon"],
            "route": ["AKLHND", "LHRJFK"],
            "booking_origin": ["Australia", "UK"],
        }
        response = client.post("/predict/column_oriented_bench", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]["probability"]) == 2
        assert len(data["predictions"]["booking_prediction"]) == 2
        assert len(data["predictions"]["business_logic"]["priority_score"]) == 2

    def test_column_oriented_csv_upload(self, client):
        csv_content = (
            "num_passengers,purchase_lead,length_of_stay,flight_hour,flight_duration,"
            "wants_extra_baggage,wants_preferred_seat,wants_in_flight_meals,"
            "sales_channel,trip_type,flight_day,route,booking_origin\n"
            "2,13,7,10,8.5,1,0,1,Internet,RoundTrip,Sat,AKLHND,Australia\n"
        )
        response = client.post(
            "/predict/column_oriented",
            files={"file": ("test.csv", csv_content, "text/csv")},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]["probability"]) == 1

    def test_invalid_file_type(self, client):
        response = client.post(
            "/predict/column_oriented",
            files={"file": ("test.txt", "not csv", "text/plain")},
        )
        assert response.status_code == 500


class TestRAGGenerate:
    def test_rag_generate_success(self, client):
        payload = {
            "customer_id": "CUST001",
            "customer_name": "John Doe",
            "email": "john@example.com",
            "route": "AKLHND",
            "booking_origin": "Australia",
            "haul_type": "Long Haul",
            "num_passengers": 2,
            "wants_extra_baggage": True,
            "wants_preferred_seat": False,
            "wants_in_flight_meals": True,
        }
        response = client.post("/api/v1/rag/generate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "subject" in data
        assert "body" in data
        assert "retrieved_sources" in data
        assert "system_prompt_id" in data
        assert "tokens_input" in data
        assert "tokens_output" in data
        assert "latency_ms" in data
        assert len(data["body"]) > 0

    def test_rag_generate_missing_fields(self, client):
        response = client.post("/api/v1/rag/generate", json={})
        assert response.status_code == 422


class TestRAGFeedback:
    def _feedback_payload(self, **overrides):
        base = {
            "customer_id": "CUST001",
            "customer_name": "John Doe",
            "booking_origin": "Australia",
            "haul_type": "Long Haul",
            "num_passengers": 2,
            "wants_extra_baggage": True,
            "wants_preferred_seat": False,
            "wants_in_flight_meals": True,
            "retrieved_sources": ["policy.md"],
            "system_prompt_id": "ba_copywriter_v1",
            "generated_subject": "Original Subject",
            "generated_body": "Original body text here",
            "edited_subject": "Original Subject",
            "edited_body": "Original body text here",
            "rating": 5,
            "accepted": True,
        }
        base.update(overrides)
        return base

    def test_feedback_success(self, client):
        response = client.post("/api/v1/rag/feedback", json=self._feedback_payload())
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_feedback_invalid_rating(self, client):
        payload = self._feedback_payload(rating=6)
        response = client.post("/api/v1/rag/feedback", json=payload)
        assert response.status_code == 422

    def test_feedback_with_edit(self, client):
        payload = self._feedback_payload(
            customer_id="CUST002",
            generated_subject="Original",
            generated_body="Original body",
            edited_subject="Edited",
            edited_body="Edited body",
            rating=4,
        )
        response = client.post("/api/v1/rag/feedback", json=payload)
        assert response.status_code == 200
