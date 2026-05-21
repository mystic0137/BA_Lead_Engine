import pytest
from src.rag.prompt_builder import build_user_prompt, get_system_prompt
from src.config import SYSTEM_PROMPTS, ACTIVE_SYSTEM_PROMPT_ID


class TestBuildUserPrompt:
    def test_basic_prompt(self, valid_lead):
        prompt = build_user_prompt(valid_lead)
        assert "John" in prompt
        assert "Australia" in prompt
        assert "savour" in prompt
        assert "our most valued traveler" in prompt
        assert "freedom to pack" in prompt
        assert "dining experience" in prompt
        assert "Address John by first name" in prompt

    def test_with_policy_context(self, valid_lead):
        policy = "All passengers are entitled to one carry-on bag."
        prompt = build_user_prompt(valid_lead, policy_context=policy)
        assert policy in prompt

    def test_no_amenities(self):
        lead = {
            "customer_name": "Jane Smith",
            "booking_origin": "London",
            "haul_type": "Short Haul",
            "num_passengers": 1,
            "wants_extra_baggage": False,
            "wants_preferred_seat": False,
            "wants_in_flight_meals": False,
            "segment": "The Window Shopper",
        }
        prompt = build_user_prompt(lead)
        assert "Jane" in prompt
        assert "London" in prompt
        assert "not selected specific add-ons" in prompt

    def test_large_group(self):
        lead = {
            "customer_name": "Bob Johnson",
            "booking_origin": "New York",
            "haul_type": "Long Haul",
            "num_passengers": 5,
            "wants_extra_baggage": True,
            "wants_preferred_seat": False,
            "wants_in_flight_meals": False,
            "segment": "The Persuadable",
        }
        prompt = build_user_prompt(lead)
        assert "Bob" in prompt
        assert "family" in prompt

    def test_unknown_segment(self):
        lead = {
            "customer_name": "Alice Wonder",
            "booking_origin": "Paris",
            "haul_type": "Long Haul",
            "num_passengers": 2,
            "wants_extra_baggage": False,
            "wants_preferred_seat": False,
            "wants_in_flight_meals": False,
            "segment": "Unknown",
        }
        prompt = build_user_prompt(lead)
        assert "valued traveler" in prompt


class TestGetSystemPrompt:
    def test_returns_ba_copywriter_prompt(self):
        prompt = get_system_prompt()
        assert "British Airways" in prompt
        assert "warm, sophisticated" in prompt
        assert "STRICT RULES" in prompt

    def test_specific_prompt_id(self):
        prompt = get_system_prompt("ba_copywriter_v1")
        assert prompt == SYSTEM_PROMPTS["ba_copywriter_v1"]

    def test_invalid_prompt_id(self):
        with pytest.raises(KeyError):
            get_system_prompt("nonexistent")
