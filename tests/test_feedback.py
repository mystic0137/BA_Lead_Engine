import json
import pytest
from pathlib import Path
from unittest.mock import patch
from src.rag.feedback import (
    _resolve_label, save_feedback, load_feedback, feedback_stats, Label,
    FEEDBACK_LOG, SFT_LOG, DPO_LOG,
)


class TestResolveLabel:
    def test_accepted_no_edit_good_rating(self):
        label, contra = _resolve_label(True, False, 5)
        assert label == Label.ACCEPTED
        assert contra is None

    def test_accepted_no_edit_low_rating(self):
        label, contra = _resolve_label(True, False, 2)
        assert label == Label.ACCEPTED
        assert contra == "low_rating_but_accepted"

    def test_accepted_with_edit(self):
        label, contra = _resolve_label(True, True, 4)
        assert label == Label.EDITED
        assert contra is None

    def test_rejected_high_rating(self):
        label, contra = _resolve_label(False, False, 5)
        assert label == Label.REJECTED
        assert contra == "high_rating_but_rejected"

    def test_rejected_low_rating(self):
        label, contra = _resolve_label(False, False, 2)
        assert label == Label.REJECTED
        assert contra is None

    def test_neutral(self):
        label, contra = _resolve_label(None, False, 3)
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
            "segment": "The VIP",
            "retrieved_sources": ["policy1.md", "policy2.md"],
        }

    def test_save_accepted_feedback(self, lead, tmp_path):
        feedback_dir = tmp_path / "finetuning"
        feedback_dir.mkdir()

        with patch("src.rag.feedback.FINETUNING_DIR", feedback_dir), \
             patch("src.rag.feedback.FEEDBACK_LOG", feedback_dir / "feedback_log.jsonl"), \
             patch("src.rag.feedback.SFT_LOG", feedback_dir / "sft_log.jsonl"), \
             patch("src.rag.feedback.DPO_LOG", feedback_dir / "dpo_log.jsonl"):

            save_feedback(
                lead=lead,
                system_prompt_id="ba_copywriter_v1",
                generated_subject="Original Subject",
                generated_body="Original body text",
                edited_subject="Original Subject",
                edited_body="Original body text",
                rating=5,
                accepted=True,
            )

            assert (feedback_dir / "feedback_log.jsonl").exists()
            assert (feedback_dir / "sft_log.jsonl").exists()

            feedback_record = json.loads(
                (feedback_dir / "feedback_log.jsonl").read_text()
            )
            assert feedback_record["label"] == "accepted"
            assert feedback_record["rating"] == 5
            assert feedback_record["meta"]["customer_id"] == "CUST001"

    def test_save_rejected_feedback(self, lead, tmp_path):
        feedback_dir = tmp_path / "finetuning"
        feedback_dir.mkdir()

        with patch("src.rag.feedback.FINETUNING_DIR", feedback_dir), \
             patch("src.rag.feedback.FEEDBACK_LOG", feedback_dir / "feedback_log.jsonl"), \
             patch("src.rag.feedback.SFT_LOG", feedback_dir / "sft_log.jsonl"), \
             patch("src.rag.feedback.DPO_LOG", feedback_dir / "dpo_log.jsonl"):

            save_feedback(
                lead=lead,
                system_prompt_id="ba_copywriter_v1",
                generated_subject="Original Subject",
                generated_body="Original body text",
                edited_subject="Original Subject",
                edited_body="Original body text",
                rating=2,
                accepted=False,
            )

            assert (feedback_dir / "feedback_log.jsonl").exists()
            feedback_record = json.loads(
                (feedback_dir / "feedback_log.jsonl").read_text()
            )
            assert feedback_record["label"] == "rejected"

    def test_save_edited_feedback_creates_dpo(self, lead, tmp_path):
        feedback_dir = tmp_path / "finetuning"
        feedback_dir.mkdir()

        with patch("src.rag.feedback.FINETUNING_DIR", feedback_dir), \
             patch("src.rag.feedback.FEEDBACK_LOG", feedback_dir / "feedback_log.jsonl"), \
             patch("src.rag.feedback.SFT_LOG", feedback_dir / "sft_log.jsonl"), \
             patch("src.rag.feedback.DPO_LOG", feedback_dir / "dpo_log.jsonl"):

            save_feedback(
                lead=lead,
                system_prompt_id="ba_copywriter_v1",
                generated_subject="Original Subject",
                generated_body="Original body text",
                edited_subject="Edited Subject",
                edited_body="Edited body text",
                rating=4,
                accepted=True,
            )

            assert (feedback_dir / "feedback_log.jsonl").exists()
            assert (feedback_dir / "sft_log.jsonl").exists()
            assert (feedback_dir / "dpo_log.jsonl").exists()

            dpo_record = json.loads(
                (feedback_dir / "dpo_log.jsonl").read_text()
            )
            assert "chosen" in dpo_record
            assert "rejected" in dpo_record
            assert "Edited" in dpo_record["chosen"]
            assert "Original" in dpo_record["rejected"]

    def test_save_neutral_feedback(self, lead, tmp_path):
        feedback_dir = tmp_path / "finetuning"
        feedback_dir.mkdir()

        with patch("src.rag.feedback.FINETUNING_DIR", feedback_dir), \
             patch("src.rag.feedback.FEEDBACK_LOG", feedback_dir / "feedback_log.jsonl"), \
             patch("src.rag.feedback.SFT_LOG", feedback_dir / "sft_log.jsonl"), \
             patch("src.rag.feedback.DPO_LOG", feedback_dir / "dpo_log.jsonl"):

            save_feedback(
                lead=lead,
                system_prompt_id="ba_copywriter_v1",
                generated_subject="Subject",
                generated_body="Body",
                edited_subject="Subject",
                edited_body="Body",
                rating=3,
                accepted=None,
            )

            assert (feedback_dir / "feedback_log.jsonl").exists()
            feedback_record = json.loads(
                (feedback_dir / "feedback_log.jsonl").read_text()
            )
            assert feedback_record["label"] == "neutral"
            assert not (feedback_dir / "sft_log.jsonl").exists()
            assert not (feedback_dir / "dpo_log.jsonl").exists()


class TestLoadFeedback:
    def test_no_feedback_file(self, tmp_path):
        with patch("src.rag.feedback.FEEDBACK_LOG", tmp_path / "nonexistent.jsonl"):
            assert load_feedback() == []

    def test_load_feedback_records(self, tmp_path):
        feedback_file = tmp_path / "feedback_log.jsonl"
        records = [
            {"label": "accepted", "rating": 5, "meta": {"customer_id": "C1", "was_edited": False}},
            {"label": "rejected", "rating": 2, "meta": {"customer_id": "C2", "was_edited": False}},
        ]
        feedback_file.write_text("\n".join(json.dumps(r) for r in records))

        with patch("src.rag.feedback.FEEDBACK_LOG", feedback_file):
            loaded = load_feedback()
            assert len(loaded) == 2


class TestFeedbackStats:
    def test_no_feedback(self, tmp_path):
        with patch("src.rag.feedback.FEEDBACK_LOG", tmp_path / "empty.jsonl"):
            stats = feedback_stats()
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

        with patch("src.rag.feedback.FEEDBACK_LOG", feedback_file), \
             patch("src.rag.feedback.SFT_LOG", sft_file), \
             patch("src.rag.feedback.DPO_LOG", dpo_file):
            stats = feedback_stats()
            assert stats["total"] == 3
            assert stats["avg_rating"] == round((5 + 4 + 2) / 3, 2)
            assert stats["edited_count"] == 1
            assert stats["edit_rate"] == round(1 / 3, 2)
            assert stats["sft_pairs"] == 2
            assert stats["dpo_pairs"] == 1
