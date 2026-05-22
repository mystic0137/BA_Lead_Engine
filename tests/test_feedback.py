import json
import pytest
from src.rag.manager import RAGManager
from src.rag.feedback import Label, resolve_label


rm = RAGManager()


class TestResolveLabel:
    def test_accepted_no_edit_good_rating(self):
        label, contra = resolve_label(True, False, 5)
        assert label == Label.ACCEPTED
        assert contra is None

    def test_accepted_no_edit_low_rating(self):
        label, contra = resolve_label(True, False, 2)
        assert label == Label.ACCEPTED
        assert contra == "low_rating_but_accepted"

    def test_accepted_with_edit(self):
        label, contra = resolve_label(True, True, 4)
        assert label == Label.EDITED
        assert contra is None

    def test_rejected_high_rating(self):
        label, contra = resolve_label(False, False, 5)
        assert label == Label.REJECTED
        assert contra == "high_rating_but_rejected"

    def test_rejected_low_rating(self):
        label, contra = resolve_label(False, False, 2)
        assert label == Label.REJECTED
        assert contra is None

    def test_neutral(self):
        label, contra = resolve_label(None, False, 3)
        assert label == Label.NEUTRAL
        assert contra is None


class TestSaveFeedback:
    @pytest.fixture
    def lead(self):
        return {
            "customer_id": "CUST001",
            "customer_name": "John Doe",
            "booking_origin": "Australia",
            "haul_type": "Long Haul",
            "num_passengers": 2,
            "wants_extra_baggage": True,
            "wants_preferred_seat": False,
            "wants_in_flight_meals": True,
            "retrieved_sources": ["policy1.md", "policy2.md"],
        }

    @pytest.fixture
    def manager(self, tmp_path):
        rm.feedback_log = tmp_path / "feedback_log.jsonl"
        rm.sft_log = tmp_path / "sft_log.jsonl"
        rm.dpo_log = tmp_path / "dpo_log.jsonl"

    def test_save_accepted_feedback(self, lead, manager):
        rm.save_feedback(
            customer_id=lead["customer_id"],
            customer_name=lead["customer_name"],
            email="",
            route="",
            booking_origin=lead["booking_origin"],
            haul_type=lead["haul_type"],
            num_passengers=lead["num_passengers"],
            wants_extra_baggage=lead["wants_extra_baggage"],
            wants_preferred_seat=lead["wants_preferred_seat"],
            wants_in_flight_meals=lead["wants_in_flight_meals"],
            retrieved_sources=lead["retrieved_sources"],
            system_prompt_id="ba_copywriter_v1",
            generated_subject="Original Subject",
            generated_body="Original body text",
            edited_subject="Original Subject",
            edited_body="Original body text",
            rating=5,
            accepted=True,
        )

        assert rm.feedback_log.exists()
        assert rm.sft_log.exists()

        feedback_record = json.loads(rm.feedback_log.read_text())
        assert feedback_record["label"] == "accepted"
        assert feedback_record["rating"] == 5
        assert feedback_record["meta"]["customer_id"] == "CUST001"

    def test_save_rejected_feedback(self, lead, manager):
        rm.save_feedback(
            customer_id=lead["customer_id"],
            customer_name=lead["customer_name"],
            email="",
            route="",
            booking_origin=lead["booking_origin"],
            haul_type=lead["haul_type"],
            num_passengers=lead["num_passengers"],
            wants_extra_baggage=lead["wants_extra_baggage"],
            wants_preferred_seat=lead["wants_preferred_seat"],
            wants_in_flight_meals=lead["wants_in_flight_meals"],
            retrieved_sources=lead["retrieved_sources"],
            system_prompt_id="ba_copywriter_v1",
            generated_subject="Original Subject",
            generated_body="Original body text",
            edited_subject="Original Subject",
            edited_body="Original body text",
            rating=2,
            accepted=False,
        )

        assert rm.feedback_log.exists()
        feedback_record = json.loads(rm.feedback_log.read_text())
        assert feedback_record["label"] == "rejected"

    def test_save_edited_feedback_creates_dpo(self, lead, manager):
        rm.save_feedback(
            customer_id=lead["customer_id"],
            customer_name=lead["customer_name"],
            email="",
            route="",
            booking_origin=lead["booking_origin"],
            haul_type=lead["haul_type"],
            num_passengers=lead["num_passengers"],
            wants_extra_baggage=lead["wants_extra_baggage"],
            wants_preferred_seat=lead["wants_preferred_seat"],
            wants_in_flight_meals=lead["wants_in_flight_meals"],
            retrieved_sources=lead["retrieved_sources"],
            system_prompt_id="ba_copywriter_v1",
            generated_subject="Original Subject",
            generated_body="Original body text",
            edited_subject="Edited Subject",
            edited_body="Edited body text",
            rating=4,
            accepted=True,
        )

        assert rm.feedback_log.exists()
        assert rm.sft_log.exists()
        assert rm.dpo_log.exists()

        dpo_record = json.loads(rm.dpo_log.read_text())
        assert "chosen" in dpo_record
        assert "rejected" in dpo_record
        assert "Edited" in dpo_record["chosen"]
        assert "Original" in dpo_record["rejected"]

    def test_save_neutral_feedback(self, lead, manager):
        rm.save_feedback(
            customer_id=lead["customer_id"],
            customer_name=lead["customer_name"],
            email="",
            route="",
            booking_origin=lead["booking_origin"],
            haul_type=lead["haul_type"],
            num_passengers=lead["num_passengers"],
            wants_extra_baggage=lead["wants_extra_baggage"],
            wants_preferred_seat=lead["wants_preferred_seat"],
            wants_in_flight_meals=lead["wants_in_flight_meals"],
            retrieved_sources=lead["retrieved_sources"],
            system_prompt_id="ba_copywriter_v1",
            generated_subject="Subject",
            generated_body="Body",
            edited_subject="Subject",
            edited_body="Body",
            rating=3,
            accepted=None,
        )

        assert rm.feedback_log.exists()
        feedback_record = json.loads(rm.feedback_log.read_text())
        assert feedback_record["label"] == "neutral"
        assert not rm.sft_log.exists()
        assert not rm.dpo_log.exists()


class TestLoadFeedback:
    def test_no_feedback_file(self, tmp_path):
        rm.feedback_log = tmp_path / "nonexistent.jsonl"
        assert rm.load_feedback() == []

    def test_load_feedback_records(self, tmp_path):
        feedback_file = tmp_path / "feedback_log.jsonl"
        records = [
            {"label": "accepted", "rating": 5, "meta": {"customer_id": "C1", "was_edited": False}},
            {"label": "rejected", "rating": 2, "meta": {"customer_id": "C2", "was_edited": False}},
        ]
        feedback_file.write_text("\n".join(json.dumps(r) for r in records))

        rm.feedback_log = feedback_file
        loaded = rm.load_feedback()
        assert len(loaded) == 2


class TestFeedbackStats:
    def test_no_feedback(self, tmp_path):
        rm.feedback_log = tmp_path / "empty.jsonl"
        stats = rm.feedback_stats()
        assert stats == {"total": 0}

    def test_feedback_stats(self, tmp_path):
        feedback_file = tmp_path / "feedback_log.jsonl"
        sft_file = tmp_path / "sft_log.jsonl"
        dpo_file = tmp_path / "dpo_log.jsonl"

        sft_file.write_text("{}\n{}\n")
        dpo_file.write_text("{}\n")

        records = [
            {
                "label": "accepted", "rating": 5,
                "meta": {"customer_id": "C1", "was_edited": False, "contradiction": None},
            },
            {
                "label": "edited", "rating": 4,
                "meta": {"customer_id": "C2", "was_edited": True, "contradiction": None},
            },
            {
                "label": "rejected", "rating": 2,
                "meta": {"customer_id": "C3", "was_edited": False, "contradiction": None},
            },
        ]
        feedback_file.write_text("\n".join(json.dumps(r) for r in records))

        rm.feedback_log = feedback_file
        rm.sft_log = sft_file
        rm.dpo_log = dpo_file

        stats = rm.feedback_stats()
        assert stats["total"] == 3
        assert stats["avg_rating"] == round((5 + 4 + 2) / 3, 2)
        assert stats["edited_count"] == 1
        assert stats["edit_rate"] == round(1 / 3, 2)
        assert stats["sft_pairs"] == 2
        assert stats["dpo_pairs"] == 1
