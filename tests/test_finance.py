import json
import numpy as np
from pathlib import Path
from src.analytics.finance import (
    _base_flight_value,
    _calculate_potential_revenue,
    _lead_value_tier,
    BACostCalculator,
)


class TestBaseFlightValue:
    def test_short_haul(self):
        assert _base_flight_value(2.9) == 150.0
        assert _base_flight_value(0) == 150.0

    def test_medium_haul(self):
        assert _base_flight_value(3.0) == 350.0
        assert _base_flight_value(6.0) == 350.0

    def test_long_haul(self):
        assert _base_flight_value(6.1) == 550.0
        assert _base_flight_value(24.0) == 550.0


class TestBaseFlightValueVectorized:
    def test_short_haul(self):
        result = _base_flight_value(np.array([2.9, 0.0]))
        assert np.all(result == 150.0)

    def test_medium_haul(self):
        result = _base_flight_value(np.array([3.0, 6.0]))
        assert np.all(result == 350.0)

    def test_long_haul(self):
        result = _base_flight_value(np.array([6.1, 24.0]))
        assert np.all(result == 550.0)

    def test_mixed(self):
        result = _base_flight_value(np.array([1.0, 4.0, 10.0]))
        assert np.all(result == [150.0, 350.0, 550.0])


class TestCalculatePotentialRevenue:
    def test_basic(self):
        data = {
            "flight_duration": 8.0,
            "num_passengers": 1,
            "wants_extra_baggage": 0,
            "wants_preferred_seat": 0,
            "wants_in_flight_meals": 0,
        }
        rev = _calculate_potential_revenue(data)
        assert rev == 550.0

    def test_with_amenities(self):
        data = {
            "flight_duration": 8.0,
            "num_passengers": 2,
            "wants_extra_baggage": 1,
            "wants_preferred_seat": 1,
            "wants_in_flight_meals": 1,
        }
        rev = _calculate_potential_revenue(data)
        expected = (550.0 + 50.0 + 40.0 + 20.0) * 2
        assert rev == expected

    def test_revenue_cap(self):
        data = {
            "flight_duration": 8.0,
            "num_passengers": 20,
            "wants_extra_baggage": 1,
            "wants_preferred_seat": 1,
            "wants_in_flight_meals": 1,
        }
        rev = _calculate_potential_revenue(data)
        assert rev == 10000.0

    def test_defaults_for_missing_fields(self):
        data = {}
        rev = _calculate_potential_revenue(data)
        assert rev == 150.0


class TestCalculatePotentialRevenueVectorized:
    def test_basic(self):
        data = {
            "flight_duration": np.array([8.0]),
            "num_passengers": np.array([1]),
            "wants_extra_baggage": np.array([0]),
            "wants_preferred_seat": np.array([0]),
            "wants_in_flight_meals": np.array([0]),
        }
        rev = _calculate_potential_revenue(data)
        assert rev[0] == 550.0

    def test_multi_row(self):
        data = {
            "flight_duration": np.array([2.0, 8.0]),
            "num_passengers": np.array([1, 2]),
            "wants_extra_baggage": np.array([0, 1]),
            "wants_preferred_seat": np.array([0, 1]),
            "wants_in_flight_meals": np.array([0, 1]),
        }
        rev = _calculate_potential_revenue(data)
        assert rev[0] == 150.0
        expected = (550.0 + 50.0 + 40.0 + 20.0) * 2
        assert rev[1] == expected


class TestLeadValueTier:
    def test_high_value(self):
        assert _lead_value_tier(2500.0) == 4
        assert _lead_value_tier(5000.0) == 4

    def test_medium_value(self):
        assert _lead_value_tier(800.0) == 3
        assert _lead_value_tier(2499.99) == 3

    def test_low_value(self):
        assert _lead_value_tier(0.0) == 2
        assert _lead_value_tier(799.99) == 2


class TestLeadValueTierVectorized:
    def test_mixed(self):
        rev = np.array([500.0, 1000.0, 3000.0])
        tiers = _lead_value_tier(rev)
        assert np.all(tiers == [2, 3, 4])


class TestBACostCalculator:
    def test_default_initialization(self):
        calc = BACostCalculator()
        assert calc.priority_cost_map["email_cost"] == 3.50
        assert calc.priority_cost_map["drip_sequence_cost"] == 12.00
        assert calc.priority_cost_map["call_cost"] == 35.00
        assert calc.value_tier_map == {2: "Low", 3: "Medium", 4: "High"}

    def test_custom_costs(self):
        calc = BACostCalculator(email_cost=5.0, drip_sequence_cost=15.0, call_cost=40.0)
        assert calc.priority_cost_map["email_cost"] == 5.0

    def test_calculate_lead_value_vip(self):
        calc = BACostCalculator()
        data = {
            "flight_duration": 8.0,
            "num_passengers": 5,
            "wants_extra_baggage": 1,
            "wants_preferred_seat": 1,
            "wants_in_flight_meals": 1,
        }
        result = calc.calculate_lead_value(0.90, data)
        assert result["priority_score"] == 2
        assert result["value_tier"] == "High"
        assert result["potential_revenue_usd"] > 0
        assert result["expected_value_usd"] > 0

    def test_calculate_lead_value_persuadable(self):
        calc = BACostCalculator()
        data = {
            "flight_duration": 8.0,
            "num_passengers": 5,
            "wants_extra_baggage": 1,
            "wants_preferred_seat": 1,
            "wants_in_flight_meals": 1,
        }
        result = calc.calculate_lead_value(0.50, data)
        assert result["priority_score"] == 3
        assert result["value_tier"] == "High"

    def test_calculate_lead_value_window_shopper(self):
        calc = BACostCalculator()
        data = {
            "flight_duration": 2.0,
            "num_passengers": 1,
            "wants_extra_baggage": 0,
            "wants_preferred_seat": 0,
            "wants_in_flight_meals": 0,
        }
        result = calc.calculate_lead_value(0.90, data)
        assert result["priority_score"] == 1
        assert result["value_tier"] == "Low"

    def test_calculate_lead_value_lost_cause(self):
        calc = BACostCalculator()
        data = {
            "flight_duration": 2.0,
            "num_passengers": 1,
            "wants_extra_baggage": 0,
            "wants_preferred_seat": 0,
            "wants_in_flight_meals": 0,
        }
        result = calc.calculate_lead_value(0.20, data)
        assert result["priority_score"] == 0
        assert result["value_tier"] == "Low"

    def test_vectorized_calculate_lead_value(self):
        calc = BACostCalculator()
        data = {
            "flight_duration": np.array([8.0, 8.0, 2.0, 2.0]),
            "num_passengers": np.array([5, 5, 1, 1]),
            "wants_extra_baggage": np.array([1, 1, 0, 0]),
            "wants_preferred_seat": np.array([1, 1, 0, 0]),
            "wants_in_flight_meals": np.array([1, 1, 0, 0]),
        }
        probs = np.array([0.90, 0.50, 0.90, 0.20])
        result = calc.calculate_lead_value(probs, data)
        assert len(result["priority_score"]) == 4
        assert result["priority_score"][0] == 2
        assert result["priority_score"][3] == 0
        assert all(t in ("High", "Medium", "Low") for t in result["value_tier"])

    def test_priority_queue_operations(self):
        calc = BACostCalculator()
        assert calc.get_queue_size() == 0
        assert calc.get_next_best_lead() is None
        assert calc.peek_top_lead() is None

        calc.add_to_priority_queue("CUST001", 500.0, {"name": "Alice"})
        calc.add_to_priority_queue("CUST002", 1000.0, {"name": "Bob"})
        calc.add_to_priority_queue("CUST003", 750.0, {"name": "Charlie"})
        assert calc.get_queue_size() == 3

        best = calc.peek_top_lead()
        assert best[0] == 1000.0
        assert best[1] == "CUST002"

        best = calc.get_next_best_lead()
        assert best[0] == 1000.0
        assert best[1] == "CUST002"
        assert calc.get_queue_size() == 2

        second = calc.get_next_best_lead()
        assert second[0] == 750.0

    def test_export_to_jsonl(self, tmp_path):
        calc = BACostCalculator()
        calc.add_to_priority_queue("CUST001", 500.0, {"name": "Alice"})
        calc.add_to_priority_queue("CUST002", 1000.0, {"name": "Bob"})

        file_path = tmp_path / "queue.jsonl"
        calc.export_all_to_jsonl(file_path, clear_queue=True)
        assert file_path.exists()

        lines = file_path.read_text().strip().split("\n")
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert "customer_id" in record
        assert "ev" in record
        assert "payload" in record
        assert calc.get_queue_size() == 0

    def test_export_without_clear(self, tmp_path):
        calc = BACostCalculator()
        calc.add_to_priority_queue("CUST001", 500.0, {"name": "Alice"})

        file_path = tmp_path / "queue_keep.jsonl"
        calc.export_all_to_jsonl(file_path, clear_queue=False)
        assert calc.get_queue_size() == 1
