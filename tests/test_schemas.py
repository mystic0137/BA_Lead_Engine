import pytest
from pydantic import ValidationError
from app.schemas import (
    RoworientedInput, ColumnorientedInput,
    BusinessLogicRoworiented, BusinessLogicColumnoriented,
    RAGGenerateRequest, RAGFeedbackRequest,
    PredictionRoworiented, PredictionColumnoriented, Meta,
)
from pydantic import ValidationError


class TestRoworientedInput:
    def test_valid_payload(self, valid_single_record):
        data = RoworientedInput(**valid_single_record)
        assert data.num_passengers == 2
        assert data.flight_day == "Sat"

    def test_invalid_flight_hour(self, valid_single_record):
        with pytest.raises(ValidationError):
            RoworientedInput(**{**valid_single_record, "flight_hour": 25})

    def test_invalid_num_passengers(self, valid_single_record):
        with pytest.raises(ValidationError):
            RoworientedInput(**{**valid_single_record, "num_passengers": 0})

    def test_invalid_flight_duration(self, valid_single_record):
        with pytest.raises(ValidationError):
            RoworientedInput(**{**valid_single_record, "flight_duration": -1})

    def test_invalid_literal_sales_channel(self, valid_single_record):
        with pytest.raises(ValidationError):
            RoworientedInput(**{**valid_single_record, "sales_channel": "Phone"})

    def test_invalid_literal_trip_type(self, valid_single_record):
        with pytest.raises(ValidationError):
            RoworientedInput(**{**valid_single_record, "trip_type": "Invalid"})

    def test_invalid_literal_flight_day(self, valid_single_record):
        with pytest.raises(ValidationError):
            RoworientedInput(**{**valid_single_record, "flight_day": "XXX"})

    def test_missing_required_field(self, valid_single_record):
        with pytest.raises(ValidationError):
            RoworientedInput(**{k: v for k, v in valid_single_record.items() if k != "route"})


class TestBusinessLogicRoworiented:
    def test_valid(self):
        bl = BusinessLogicRoworiented(
            priority_score=2, value_tier="High",
            expected_value_usd=500.0, potential_revenue_usd=1000.0,
            marginal_profit_usd=496.5,
        )
        assert bl.priority_score == 2

    def test_invalid_value_tier(self):
        with pytest.raises(ValidationError):
            BusinessLogicRoworiented(
                priority_score=2, value_tier="Invalid",
                expected_value_usd=500.0, potential_revenue_usd=1000.0,
                marginal_profit_usd=496.5,
            )


class TestColumnorientedInput:
    def test_valid(self):
        data = ColumnorientedInput(
            num_passengers=[1, 2], purchase_lead=[10, 20],
            length_of_stay=[3, 7], flight_hour=[10, 14],
            flight_duration=[5.0, 8.5],
            wants_extra_baggage=[0, 1], wants_preferred_seat=[1, 0],
            wants_in_flight_meals=[0, 1],
            sales_channel=["Internet", "Mobile"],
            trip_type=["RoundTrip", "OneWay"],
            flight_day=["Mon", "Tue"],
            route=["AKLHND", "LHRJFK"],
            booking_origin=["Australia", "UK"],
        )
        assert len(data.num_passengers) == 2

    def test_invalid_literal(self):
        with pytest.raises(ValidationError):
            ColumnorientedInput(
                num_passengers=[1], purchase_lead=[10],
                length_of_stay=[3], flight_hour=[10],
                flight_duration=[5.0],
                wants_extra_baggage=[0], wants_preferred_seat=[1],
                wants_in_flight_meals=[0],
                sales_channel=["Phone"],
                trip_type=["RoundTrip"],
                flight_day=["Mon"],
                route=["AKLHND"],
                booking_origin=["Australia"],
            )


class TestRAGGenerateRequest:
    def test_valid(self):
        req = RAGGenerateRequest(
            customer_id="CUST001", customer_name="John Doe",
            email="john@test.com", route="AKLHND",
            booking_origin="Australia", haul_type="Long Haul",
            num_passengers=2, wants_extra_baggage=True,
            wants_preferred_seat=False, wants_in_flight_meals=True,
        )
        assert req.customer_id == "CUST001"

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            RAGGenerateRequest(customer_id="CUST001")


class TestRAGFeedbackRequest:
    def test_valid(self):
        req = RAGFeedbackRequest(
            customer_id="CUST001", retrieved_sources=["policy.md"],
            system_prompt_id="ba_copywriter_v1",
            generated_subject="Sub", generated_body="Body",
            edited_subject="Sub", edited_body="Body",
            rating=4, accepted=True,
        )
        assert req.rating == 4

    def test_invalid_rating(self):
        with pytest.raises(ValidationError):
            RAGFeedbackRequest(
                customer_id="CUST001", retrieved_sources=["policy.md"],
                system_prompt_id="ba_copywriter_v1",
                generated_subject="Sub", generated_body="Body",
                edited_subject="Sub", edited_body="Body",
                rating=6, accepted=True,
            )

    def test_rating_below_one(self):
        with pytest.raises(ValidationError):
            RAGFeedbackRequest(
                customer_id="CUST001", retrieved_sources=["policy.md"],
                system_prompt_id="ba_copywriter_v1",
                generated_subject="Sub", generated_body="Body",
                edited_subject="Sub", edited_body="Body",
                rating=0, accepted=True,
            )


class TestPredictionModels:
    def test_prediction_row_oriented(self):
        bl = BusinessLogicRoworiented(
            priority_score=2, value_tier="High",
            expected_value_usd=500.0, potential_revenue_usd=1000.0,
            marginal_profit_usd=496.5,
        )
        pred = PredictionRoworiented(
            predictions=[{"probability": 0.85, "booking_prediction": 1, "business_logic": bl}],
            meta=Meta(model_version="v1", threshold_used=0.309),
        )
        assert len(pred.predictions) == 1

    def test_prediction_column_oriented(self):
        bl = BusinessLogicColumnoriented(
            priority_score=[2], value_tier=["High"],
            expected_value_usd=[500.0], potential_revenue_usd=[1000.0],
            marginal_profit_usd=[496.5],
        )
        pred = PredictionColumnoriented(
            predictions={
                "probability": [0.85], "booking_prediction": [1],
                "business_logic": bl,
            },
            meta=Meta(model_version="v1", threshold_used=0.309),
        )
        assert len(pred.predictions.probability) == 1
