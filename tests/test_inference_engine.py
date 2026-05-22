import io
import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from app.schemas import RoworientedInput, ColumnorientedInput
from src.inference.engine import InferenceEngine, load_config
from src.inference.csv_utils import csv_to_column_oriented


class TestLoadConfig:
    def test_loads_json_config(self, tmp_path):
        config_data = {"threshold": 0.5, "model_type": "XGB"}
        p = tmp_path / "config.json"
        p.write_text(json.dumps(config_data))
        result = load_config(str(p))
        assert result == config_data

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.json")


class TestInferenceEngineInit:
    def test_loads_config_and_session(self):
        engine = InferenceEngine()
        assert engine._threshold == 0.309
        assert len(engine._all_features) > 0
        assert len(engine._numeric_features) > 0
        assert engine._calculator is not None

    def test_numeric_features_filtered_correctly(self):
        engine = InferenceEngine()
        for feat in engine._numeric_features:
            assert feat not in {"route", "booking_origin", "sales_channel", "trip_type", "flight_day"}


class TestCsvToColumnOriented:
    def test_parses_csv(self):
        csv_content = (
            "num_passengers,purchase_lead,length_of_stay,flight_hour,flight_duration,"
            "wants_extra_baggage,wants_preferred_seat,wants_in_flight_meals,"
            "sales_channel,trip_type,flight_day,route,booking_origin\n"
            "2,10,7,14,8.5,1,0,1,Internet,RoundTrip,Sat,AKLHND,Australia\n"
            "1,5,3,10,5.0,0,1,0,Mobile,OneWay,Mon,LHRJFK,UK\n"
        )
        result = csv_to_column_oriented(io.BytesIO(csv_content.encode("latin1")))
        assert len(result.num_passengers) == 2
        assert result.sales_channel == ["Internet", "Mobile"]

    def test_empty_csv(self):
        csv_content = (
            "num_passengers,purchase_lead,length_of_stay,flight_hour,flight_duration,"
            "wants_extra_baggage,wants_preferred_seat,wants_in_flight_meals,"
            "sales_channel,trip_type,flight_day,route,booking_origin\n"
        )
        result = csv_to_column_oriented(io.BytesIO(csv_content.encode("latin1")))
        assert len(result.num_passengers) == 0


class TestRunCore:
    def test_returns_probabilities(self):
        engine = InferenceEngine()
        onnx_inputs = {
            "route": np.array([["AKLHND"]]),
            "booking_origin": np.array([["Australia"]]),
            "sales_channel": np.array([["Internet"]]),
            "trip_type": np.array([["RoundTrip"]]),
            "flight_day": np.array([["Sat"]]),
            "num_passengers": np.array([[2]], dtype=np.float32),
            "purchase_lead": np.array([[13]], dtype=np.float32),
            "length_of_stay": np.array([[7]], dtype=np.float32),
            "flight_hour": np.array([[10]], dtype=np.float32),
            "flight_duration": np.array([[8.5]], dtype=np.float32),
            "wants_extra_baggage": np.array([[1]], dtype=np.float32),
            "wants_preferred_seat": np.array([[0]], dtype=np.float32),
            "wants_in_flight_meals": np.array([[1]], dtype=np.float32),
        }
        probs = engine._run_core(onnx_inputs)
        assert probs.shape == (1, 1)
        assert 0.0 <= probs[0, 0] <= 1.0

    def test_multiple_records(self):
        engine = InferenceEngine()
        onnx_inputs = {
            "route": np.array([["AKLHND"], ["LHRJFK"]]),
            "booking_origin": np.array([["Australia"], ["UK"]]),
            "sales_channel": np.array([["Internet"], ["Mobile"]]),
            "trip_type": np.array([["RoundTrip"], ["OneWay"]]),
            "flight_day": np.array([["Sat"], ["Mon"]]),
            "num_passengers": np.array([[2], [1]], dtype=np.float32),
            "purchase_lead": np.array([[13], [5]], dtype=np.float32),
            "length_of_stay": np.array([[7], [3]], dtype=np.float32),
            "flight_hour": np.array([[10], [14]], dtype=np.float32),
            "flight_duration": np.array([[8.5], [5.0]], dtype=np.float32),
            "wants_extra_baggage": np.array([[1], [0]], dtype=np.float32),
            "wants_preferred_seat": np.array([[0], [1]], dtype=np.float32),
            "wants_in_flight_meals": np.array([[1], [0]], dtype=np.float32),
        }
        probs = engine._run_core(onnx_inputs)
        assert probs.shape == (2, 1)

    def test_onnx_failure_raises(self):
        engine = InferenceEngine()
        with patch.object(engine._session, "run", side_effect=RuntimeError("ONNX failed")):
            with pytest.raises(RuntimeError):
                engine._run_core({"dummy": np.array([[1.0]])})


class TestRunRowOriented:
    def test_single_record(self, valid_single_record):
        engine = InferenceEngine()
        record = RoworientedInput(**valid_single_record)
        result = engine.run_row_oriented([record.model_dump()])
        assert len(result["predictions"]) == 1
        pred = result["predictions"][0]
        assert 0.0 <= pred["probability"] <= 1.0
        assert pred["booking_prediction"] in (0, 1)
        assert "priority_score" in pred["business_logic"]
        assert "value_tier" in pred["business_logic"]
        assert result["meta"]["model_version"] == "xgboost_onnx_v1"
        assert result["meta"]["threshold_used"] == 0.309

    def test_multiple_records(self, valid_single_record):
        engine = InferenceEngine()
        r1 = RoworientedInput(**valid_single_record)
        r2 = RoworientedInput(**{**valid_single_record, "num_passengers": 1, "flight_hour": 15})
        result = engine.run_row_oriented([r1.model_dump(), r2.model_dump()])
        assert len(result["predictions"]) == 2

    def test_with_high_probability_lead(self, valid_single_record):
        engine = InferenceEngine()
        with patch.object(engine._session, "run", return_value=[
            np.zeros((1, 1)), np.array([[0.1, 0.85]], dtype=np.float32)
        ]):
            record = RoworientedInput(**valid_single_record)
            result = engine.run_row_oriented([record.model_dump()])
            assert result["predictions"][0]["probability"] == pytest.approx(0.85, rel=1e-3)

    def test_with_low_probability_lead(self, valid_single_record):
        engine = InferenceEngine()
        with patch.object(engine._session, "run", return_value=[
            np.zeros((1, 1)), np.array([[0.9, 0.1]], dtype=np.float32)
        ]):
            record = RoworientedInput(**valid_single_record)
            result = engine.run_row_oriented([record.model_dump()])
            assert result["predictions"][0]["probability"] == pytest.approx(0.1)


class TestRunColumnOriented:
    def test_single_record(self):
        engine = InferenceEngine()
        records = ColumnorientedInput(
            num_passengers=[2], purchase_lead=[13], length_of_stay=[7],
            flight_hour=[10], flight_duration=[8.5],
            wants_extra_baggage=[1], wants_preferred_seat=[0],
            wants_in_flight_meals=[1],
            sales_channel=["Internet"], trip_type=["RoundTrip"],
            flight_day=["Sat"], route=["AKLHND"], booking_origin=["Australia"],
        )
        result = engine.run_column_oriented(records.model_dump())
        assert len(result["predictions"]["probability"]) == 1
        assert result["predictions"]["booking_prediction"][0] in (0, 1)
        assert len(result["predictions"]["business_logic"]["priority_score"]) == 1

    def test_multiple_records(self):
        engine = InferenceEngine()
        records = ColumnorientedInput(
            num_passengers=[2, 1], purchase_lead=[13, 5],
            length_of_stay=[7, 3], flight_hour=[10, 14],
            flight_duration=[8.5, 5.0],
            wants_extra_baggage=[1, 0], wants_preferred_seat=[0, 1],
            wants_in_flight_meals=[1, 0],
            sales_channel=["Internet", "Mobile"],
            trip_type=["RoundTrip", "OneWay"],
            flight_day=["Sat", "Mon"],
            route=["AKLHND", "LHRJFK"],
            booking_origin=["Australia", "UK"],
        )
        result = engine.run_column_oriented(records.model_dump())
        assert len(result["predictions"]["probability"]) == 2
