import pytest
from src.rag.email_generator import _build_query, _format_chunks, _parse_email


class TestBuildQuery:
    def test_basic_query(self):
        lead = {"haul_type": "Long Haul", "wants_extra_baggage": False,
                "wants_preferred_seat": False, "wants_in_flight_meals": False}
        query = _build_query(lead)
        assert query == "Long Haul flight policy"

    def test_with_baggage(self):
        lead = {"haul_type": "Short Haul", "wants_extra_baggage": True,
                "wants_preferred_seat": False, "wants_in_flight_meals": False}
        query = _build_query(lead)
        assert "baggage" in query

    def test_with_all_amenities(self):
        lead = {"haul_type": "Long Haul", "wants_extra_baggage": True,
                "wants_preferred_seat": True, "wants_in_flight_meals": True}
        query = _build_query(lead)
        assert "baggage" in query
        assert "seat" in query
        assert "meal" in query


class TestFormatChunks:
    def test_single_chunk(self):
        chunks = [{"source": "policy.md", "text": "Baggage allowance is 23kg."}]
        result = _format_chunks(chunks)
        assert "[Source: policy.md]" in result
        assert "Baggage allowance" in result

    def test_multiple_chunks(self):
        chunks = [
            {"source": "a.md", "text": "Text A"},
            {"source": "b.md", "text": "Text B"},
        ]
        result = _format_chunks(chunks)
        assert "---" in result
        assert "Text A" in result
        assert "Text B" in result

    def test_empty_chunks(self):
        result = _format_chunks([])
        assert result == ""


class TestParseEmail:
    def test_standard_email(self):
        raw = "Subject: Exclusive Offer\n\nDear John, enjoy your trip!\nBest, BA"
        subject, body = _parse_email(raw)
        assert subject == "Exclusive Offer"
        assert "Dear John" in body

    def test_subject_with_asterisks(self):
        raw = "**Subject:** Amazing Deal\n\nBody text here"
        subject, body = _parse_email(raw)
        assert subject == "Amazing Deal"
        assert "Body text" in body

    def test_subject_with_hash(self):
        raw = "# Subject: Test\n\nContent"
        subject, body = _parse_email(raw)
        assert subject == "Test"

    def test_no_subject_label(self):
        raw = "Enjoy this amazing flight offer! We hope you have a great time."
        subject, body = _parse_email(raw)
        assert "Enjoy" in subject
        assert "great time" in body

    def test_empty_body_falls_back(self):
        raw = "Subject: Only Subject"
        subject, body = _parse_email(raw)
        assert subject == "Only Subject"
        assert body == raw.strip()

    def test_empty_string(self):
        subject, body = _parse_email("")
        assert subject == "A journey worth taking — British Airways"

    def test_multiline_body(self):
        raw = "Subject: Welcome\n\nLine 1\nLine 2\n\nLine 3"
        subject, body = _parse_email(raw)
        assert subject == "Welcome"
        assert "Line 1" in body
        assert "Line 2" in body
        assert "Line 3" in body
