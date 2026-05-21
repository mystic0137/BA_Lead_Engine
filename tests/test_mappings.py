from src.rag.mappings import (
    SEGMENT_MAP, HAUL_HOOK_MAP, PARTY_SIZE_MAP, AMENITY_DESCRIPTIONS,
    get_party_description, get_amenity_hooks,
)


class TestConstants:
    def test_segment_map_keys(self):
        assert set(SEGMENT_MAP.keys()) == {
            "The VIP", "The Persuadable", "The Window Shopper", "The Lost Cause"
        }

    def test_haul_hook_map_keys(self):
        assert set(HAUL_HOOK_MAP.keys()) == {"Short Haul", "Long Haul"}

    def test_party_size_map_keys(self):
        assert 1 in PARTY_SIZE_MAP
        assert 2 in PARTY_SIZE_MAP
        assert "3+" in PARTY_SIZE_MAP

    def test_amenity_descriptions_keys(self):
        assert set(AMENITY_DESCRIPTIONS.keys()) == {
            "wants_extra_baggage", "wants_preferred_seat", "wants_in_flight_meals"
        }


class TestGetPartyDescription:
    def test_solo(self):
        assert "solo" in get_party_description(1)

    def test_couple(self):
        assert "couple" in get_party_description(2)

    def test_family(self):
        assert "family" in get_party_description(3)
        assert "family" in get_party_description(5)

    def test_unknown_count(self):
        result = get_party_description(0)
        assert "traveler" in result


class TestGetAmenityHooks:
    def test_all_amenities(self):
        lead = {
            "wants_extra_baggage": True,
            "wants_preferred_seat": True,
            "wants_in_flight_meals": True,
        }
        hooks = get_amenity_hooks(lead)
        assert len(hooks) == 3

    def test_no_amenities(self):
        lead = {}
        hooks = get_amenity_hooks(lead)
        assert hooks == []

    def test_partial_amenities(self):
        lead = {"wants_extra_baggage": True, "wants_preferred_seat": False}
        hooks = get_amenity_hooks(lead)
        assert len(hooks) == 1
        assert "freedom to pack" in hooks[0]
