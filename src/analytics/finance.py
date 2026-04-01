import json
import threading
import heapq
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Lead value thresholds
LOW_VALUE_MAX = 500.0
MEDIUM_VALUE_MAX = 1500.0

# Probability thresholds
HIGH_PROB = 0.75
LOW_PROB = 0.30

REVENUE_CAP = 10000


def _base_flight_value(duration: float) -> float:
    if duration < 3:
        return 150.0
    elif duration <= 6:
        return 400.0
    else:
        return 800.0


def _calculate_potential_revenue(input_data: Dict) -> float:
    duration = input_data.get("flight_duration", 0)
    num_passengers = input_data.get("num_passengers", 1)

    base = _base_flight_value(duration)
    amenities = (
        input_data.get("wants_extra_baggage", 0) * 50.0
        + input_data.get("wants_preferred_seat", 0) * 40.0
        + input_data.get("wants_in_flight_meals", 0) * 20.0
    )
    return min((base + amenities) * num_passengers, REVENUE_CAP)


def _lead_value_tier(potential_revenue: float) -> str:
    if potential_revenue >= MEDIUM_VALUE_MAX:
        return "High"
    elif potential_revenue >= LOW_VALUE_MAX:
        return "Medium"
    else:
        return "Low"


class BACostCalculator:
    def __init__(
        self,
        nudge_email_cost: float = 3.50,
        drip_sequence_cost: float = 12,
        call_cost: float = 35.00,
    ):
        self.email_cost = nudge_email_cost
        self.drip_cost = drip_sequence_cost
        self.call_cost = call_cost

        self._lead_queue: List[Tuple] = []
        self._lock = threading.Lock()

    def calculate_lead_value(self, prob: float, input_data: Dict) -> Dict:
        potential_revenue = _calculate_potential_revenue(input_data)
        expected_value = prob * potential_revenue
        value_tier = _lead_value_tier(potential_revenue)
        is_high_prob = prob >= HIGH_PROB
        is_low_prob = prob < LOW_PROB

        # Matrix from framework:
        # High prob + High value  → The VIP (Sure Thing):
        #                           Don't discount, nudge to close
        # Low/Med prob + High value → The Persuadable (Target):
        #                             Highest priority, small incentive
        # High prob + Low value   → The Window Shopper:
        #                           Let them book naturally
        # Low prob + Low value    → The Lost Cause:
        #                           Ignore

        if is_high_prob and value_tier == "High":
            segment = "The VIP"
            category = "Category 1"
            action = "Automated Nudge — No Discount (High Margin, Near Certain)"
            cost = self.email_cost
            priority_score = 3

        elif not is_high_prob and not is_low_prob and value_tier == "High":
            segment = "The Persuadable"
            category = "Category 0"
            action = "Priority Human Call — Offer Incentive (Highest ROI Target)"
            cost = self.call_cost
            priority_score = 4

        elif is_high_prob and value_tier in ("Low", "Medium"):
            segment = "The Window Shopper"
            category = "Category 2"
            action = "Email Drip Sequence — Let Them Book Naturally"
            cost = self.drip_cost
            priority_score = 1

        else:
            segment = "The Lost Cause"
            category = "Category 3"
            action = "Suppression — No Action"
            cost = 0.0
            priority_score = 0

        return {
            "segment": segment,
            "category": category,
            "recommended_action": action,
            "value_tier": value_tier,
            "potential_revenue": round(potential_revenue, 2),
            "expected_value": round(expected_value, 2),
            "marginal_profit": round(expected_value - cost, 2),
            "priority_score": priority_score,
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