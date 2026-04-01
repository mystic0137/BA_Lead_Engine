SEGMENT_MAP = {
    "The VIP": "our most valued traveler who is ready to book a premium experience",
    "The Persuadable": "someone who has shown strong interest and just needs the right nudge",
    "The Window Shopper": "an explorer who loves discovering new destinations",
    "The Lost Cause": "a traveler we hope to inspire for a future journey",
}

HAUL_HOOK_MAP = {
    "Short Haul": "A quick escape is just around the corner — the perfect weekend you'll actually remember.",
    "Long Haul": "Some journeys deserve to be savoured. Settle in, relax, and let us take care of every mile.",
}

PARTY_SIZE_MAP = {
    1: "a solo explorer seeking their next adventure",
    2: "a couple planning a memorable retreat",
    "3+": "a family ready to share an unforgettable journey together",
}

AMENITY_DESCRIPTIONS = {
    "wants_extra_baggage": "the freedom to pack everything they need without compromise",
    "wants_preferred_seat": "the comfort of their perfect seat from the moment they board",
    "wants_in_flight_meals": "a curated dining experience at 35,000 feet",
}


def get_party_description(count: int) -> str:
    if count >= 3:
        return PARTY_SIZE_MAP["3+"]
    return PARTY_SIZE_MAP.get(count, "a valued traveler")


def get_amenity_hooks(lead: dict) -> list[str]:
    return [
        desc for key, desc in AMENITY_DESCRIPTIONS.items()
        if lead.get(key)
    ]
