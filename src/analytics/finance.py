import json
import threading
import heapq
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


LOW_VALUE_MAX = 800.0
MEDIUM_VALUE_MAX = 2500.0

HIGH_PROB = 0.75
LOW_PROB = 0.30

HIGH_TIER = 4

REVENUE_CAP = 10000


def _base_flight_value(duration: np.ndarray | float) -> np.ndarray | float:
    return np.select(
        [
            duration < 3,
            duration <= 6,
        ],
        [
            150.0,
            350.0,
        ],
        default=550.0
    )


def _calculate_potential_revenue(input_data: dict) -> np.ndarray | float:
    duration = np.asarray(input_data.get("flight_duration", 0))
    num_passengers = np.asarray(input_data.get("num_passengers", 1))
    extra_baggage = np.asarray(input_data.get("wants_extra_baggage", 0))
    preferred_seat = np.asarray(input_data.get("wants_preferred_seat", 0))
    flight_meals = np.asarray(input_data.get("wants_in_flight_meals", 0))

    base = _base_flight_value(duration)

    amenities = (
        extra_baggage * 50.0
        + preferred_seat * 40.0
        + flight_meals * 20.0
    )
    return np.minimum((base + amenities) * num_passengers, REVENUE_CAP)


def _lead_value_tier(potential_revenue: np.ndarray | float) -> np.ndarray | float:
    return np.select(
        [
            potential_revenue >= MEDIUM_VALUE_MAX,
            potential_revenue >= LOW_VALUE_MAX
        ],
        [
            HIGH_TIER,
            HIGH_TIER - 1,
        ],
        default=HIGH_TIER - 2
    )


def _to_native(val):
    if isinstance(val, np.ndarray):
        if val.ndim == 0:
            return val.item()
        return np.ravel(val).tolist()
    if isinstance(val, np.generic):
        return val.item()
    return val


class BACostCalculator:
    def __init__(
        self,
        email_cost: float = 3.50,
        drip_sequence_cost: float = 12.00,
        call_cost: float = 35.00,
    ):
        self.priority_cost_map = {
            "email_cost": email_cost,
            "drip_sequence_cost": drip_sequence_cost,
            "call_cost": call_cost
        }
        self.value_tier_map = {
            2: "Low",
            3: "Medium",
            4: "High",
        }
        self.np_cost_map = np.array([
            0.00,
            drip_sequence_cost,
            email_cost,
            call_cost
        ])

        self._lead_queue: List[Tuple] = []
        self._lock = threading.Lock()

    def calculate_lead_value(self, prob: np.ndarray | float, input_data: dict) -> dict:
        prob_arr = np.asarray(prob)

        potential_revenue = _calculate_potential_revenue(input_data)
        expected_value = prob_arr * potential_revenue
        value_tier = _lead_value_tier(potential_revenue)
        is_high_prob = prob_arr >= HIGH_PROB
        is_low_prob = prob_arr < LOW_PROB
        value_tier_map = self.value_tier_map

        highest_lead_value = (
            (~is_high_prob)
            & (~is_low_prob)
            & (value_tier == HIGH_TIER)
        )
        medium_lead_value = (
            (is_high_prob)
            & (value_tier == HIGH_TIER)
        )
        low_lead_value = (
            (is_high_prob)
            & (value_tier < HIGH_TIER)
        )

        priority_score = np.select(
            [
                highest_lead_value,
                medium_lead_value,
                low_lead_value,
            ],
            [
                3,
                2,
                1,
            ],
            default=0
        )
        cost = self.np_cost_map[priority_score]

        return {
            "priority_score": _to_native(priority_score),
            "value_tier": _to_native(np.vectorize(value_tier_map.get)(value_tier)),
            "potential_revenue_usd": _to_native(np.round(potential_revenue, 2)),
            "expected_value_usd": _to_native(np.round(expected_value, 2)),
            "marginal_profit_usd": _to_native(np.round(expected_value - cost, 2)),
        }

    def add_to_priority_queue(self, customer_id: str, ev: float, metadata: Dict) -> None:
        with self._lock:
            heapq.heappush(self._lead_queue, (-ev, customer_id, metadata))

    def get_next_best_lead(self) -> Optional[Tuple[float, str, Dict]]:
        with self._lock:
            if not self._lead_queue:
                return None
            neg_ev, customer_id, data = heapq.heappop(self._lead_queue)
        return (-neg_ev, customer_id, data)

    def peek_top_lead(self) -> Optional[Tuple[float, str, Dict]]:
        with self._lock:
            if not self._lead_queue:
                return None
            neg_ev, cust_id, data = self._lead_queue[0]
        return (-neg_ev, cust_id, data)

    def get_queue_size(self) -> int:
        with self._lock:
            return len(self._lead_queue)

    def export_all_to_jsonl(self, file_path: Path, clear_queue: bool = True) -> None:
        with self._lock:
            if clear_queue:
                leads_to_save = []
                while self._lead_queue:
                    leads_to_save.append(heapq.heappop(self._lead_queue))
            else:
                leads_to_save = heapq.nsmallest(len(self._lead_queue), self._lead_queue)

        with open(file_path, "w") as f:
            for neg_ev, cust_id, data in leads_to_save:
                f.write(json.dumps({
                    "customer_id": cust_id,
                    "ev": -neg_ev,
                    "payload": data,
                }) + "\n")
