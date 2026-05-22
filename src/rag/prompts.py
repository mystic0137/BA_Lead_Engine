import logging

from src.config import ACTIVE_SYSTEM_PROMPT_ID, SYSTEM_PROMPTS

logger = logging.getLogger(__name__)

SEGMENT_MAP = {
    "The VIP": "our most valued traveler who is ready to book a premium experience",
    "The Persuadable": "someone who has shown strong interest and just needs the right nudge",
    "The Window Shopper": "an explorer who loves discovering new destinations",
    "The Lost Cause": "a traveler we hope to inspire for a future journey",
}

HAUL_HOOK_MAP = {
    "Short Haul": "A quick escape is just around the corner — the perfect weekend you'll actually remember.",
    "Medium Haul": "Far enough to feel like a real getaway, close enough to make it spontaneous — this is the sweet spot of travel.",
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

MAX_TOKENS = 600


def get_party_description(count: int) -> str:
    if count >= 3:
        return PARTY_SIZE_MAP["3+"]
    return PARTY_SIZE_MAP.get(count, "a valued traveler")


def get_amenity_hooks(lead: dict) -> list[str]:
    return [
        desc for key, desc in AMENITY_DESCRIPTIONS.items()
        if lead.get(key)
    ]


def build_user_prompt(lead: dict, policy_context: str = "") -> str:
    first_name = lead["customer_name"].split()[0]
    origin_city = lead["booking_origin"]
    hook = HAUL_HOOK_MAP.get(lead["haul_type"], "")
    party_desc = get_party_description(lead["num_passengers"])
    segment_desc = SEGMENT_MAP.get(lead.get("segment", ""), "valued traveler")
    amenity_hooks_list = get_amenity_hooks(lead)
    amenity_narrative = (
        "They have expressed interest in: " + ", ".join(amenity_hooks_list) + "."
        if amenity_hooks_list
        else "They have not selected specific add-ons yet — focus on the journey experience."
    )
    return (
        f"Draft a bespoke outreach email for {first_name}, who is traveling from {origin_city}.\n\n"
        f"--- NARRATIVE CONTEXT ---\n"
        f"This traveler is {segment_desc} and is planning a {party_desc}. "
        f"The tone of this outreach should be: {hook}\n\n"
        f"--- FOCUS AREAS ---\n"
        f"{amenity_narrative}\n\n"
        f"--- VERIFIED POLICY CONTEXT ---\n"
        f"{policy_context}\n\n"
        f"--- FINAL INSTRUCTION ---\n"
        f"Address {first_name} by first name. Blend the context and focus areas into a single, "
        f"sophisticated story. Do not use headers or bullet points in the email body. "
    )


def get_system_prompt() -> str:
    return SYSTEM_PROMPTS[ACTIVE_SYSTEM_PROMPT_ID]


def build_query(lead: dict) -> str:
    parts = [f"{lead['haul_type']} flight policy"]
    if lead.get("wants_extra_baggage"):
        parts.append("baggage allowance policy")
    if lead.get("wants_preferred_seat"):
        parts.append("seat selection policy")
    if lead.get("wants_in_flight_meals"):
        parts.append("meal service policy")
    return " ".join(parts)


def format_chunks(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )


def parse_email(raw: str) -> tuple[str, str]:
    lines = raw.splitlines()
    subject = ""
    body_lines = []
    subject_line_idx = -1
    for i, line in enumerate(lines):
        cleaned = line.strip().lstrip("*#").strip()
        if cleaned.lower().startswith("subject:"):
            subject = cleaned.split(":", 1)[-1].strip().strip("*").strip()
            subject_line_idx = i
            break
    if subject_line_idx >= 0:
        start = subject_line_idx + 1
        while start < len(lines) and lines[start].strip() == "":
            start += 1
        body_lines = lines[start:]
    else:
        stripped = raw.strip()
        first_sentence_end = next(
            (i for i, c in enumerate(stripped) if c in ".!?"), len(stripped)
        )
        subject = stripped[:first_sentence_end + 1].strip()
        body_lines = stripped[first_sentence_end + 1:].strip().splitlines()
    body = "\n".join(body_lines).strip()
    if not body:
        logger.warning("Email body parsing failed. Raw output: %s", raw)
        body = raw.strip()
    return subject or "A journey worth taking — British Airways", body
