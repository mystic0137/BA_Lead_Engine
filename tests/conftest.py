import os
import json
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


os.environ["GROQ_API_KEY"] = "gsk_test_key_for_testing"
os.environ["DEBUG_MODE"] = "true"


_EXPECTED_FEATURES = [
    "route", "booking_origin", "sales_channel", "trip_type", "flight_day",
    "num_passengers", "purchase_lead", "length_of_stay", "flight_duration",
    "wants_extra_baggage", "wants_preferred_seat", "wants_in_flight_meals", "flight_hour",
]

_STRING_FEATURES = {"route", "booking_origin", "sales_channel", "trip_type", "flight_day"}
_NUMERIC_FEATURES = [f for f in _EXPECTED_FEATURES if f not in _STRING_FEATURES]

_onnx_patcher = patch("onnxruntime.InferenceSession")
_mock_onnx_session = _onnx_patcher.start()
_mock_onnx_instance = MagicMock()
_mock_onnx_session.return_value = _mock_onnx_instance

_mock_inputs = []
for feat in _EXPECTED_FEATURES:
    inp = MagicMock()
    inp.name = feat
    if feat in _STRING_FEATURES:
        inp.type = "tensor(string)"
    else:
        inp.type = "tensor(float)"
    _mock_inputs.append(inp)

_mock_onnx_instance.get_inputs.return_value = _mock_inputs


def _mock_onnx_run(_, inputs_dict):
    sample = next(iter(inputs_dict.values()))
    n = len(sample) if hasattr(sample, '__len__') else 1
    probs = np.random.rand(n, 2).astype(np.float32)
    probs[:, 1] = 0.3 + np.random.rand(n) * 0.4
    return [np.zeros((n, 1)), probs]


_mock_onnx_instance.run.side_effect = _mock_onnx_run

_chroma_patcher = patch("chromadb.PersistentClient")
_mock_chroma = _chroma_patcher.start()
_mock_collection = MagicMock()
_mock_collection.query.return_value = {
    "documents": [["test doc"]],
    "metadatas": [[{"source": "test.md"}]],
    "distances": [[0.1]],
}
_mock_chroma.return_value.get_collection.return_value = _mock_collection

import numpy as np

_st_patcher = patch("sentence_transformers.SentenceTransformer")
_mock_st = _st_patcher.start()
_mock_model = MagicMock()
_mock_model.encode.return_value = np.array([0.1] * 384)
_mock_st.return_value = _mock_model

_groq_patcher = patch("groq.Groq")
_mock_groq_class = _groq_patcher.start()
_mock_groq_instance = MagicMock()
_mock_groq_class.return_value = _mock_groq_instance
_mock_choice = MagicMock()
_mock_choice.message.content = "Subject: Test Flight Offer\n\nEnjoy your trip!"
_mock_usage = MagicMock()
_mock_usage.prompt_tokens = 50
_mock_usage.completion_tokens = 30
_mock_response = MagicMock()
_mock_response.choices = [_mock_choice]
_mock_response.usage = _mock_usage
_mock_groq_instance.chat.completions.create.return_value = _mock_response


@pytest.fixture(autouse=True)
def _reset_mocks():
    _mock_onnx_instance.reset_mock()
    _mock_collection.reset_mock()
    _mock_model.reset_mock()
    _mock_groq_instance.reset_mock()
    yield


@pytest.fixture
def mock_onnx_run():
    return _mock_onnx_instance.run


@pytest.fixture
def mock_model_config_json(tmp_path):
    config = {
        "threshold": 0.309,
        "expected_features": [
            "route", "booking_origin", "sales_channel", "trip_type",
            "flight_day", "num_passengers", "purchase_lead", "length_of_stay",
            "flight_duration", "wants_extra_baggage", "wants_preferred_seat",
            "wants_in_flight_meals", "flight_hour"
        ],
        "numeric_features": [
            "num_passengers", "purchase_lead", "length_of_stay",
            "flight_duration", "wants_extra_baggage", "wants_preferred_seat",
            "wants_in_flight_meals", "flight_hour"
        ],
    }
    p = tmp_path / "model_config.json"
    p.write_text(json.dumps(config))
    return str(p)


@pytest.fixture
def valid_lead():
    return {
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
        "segment": "The VIP",
    }


@pytest.fixture
def valid_single_record():
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
