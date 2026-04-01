from src.config import ACTIVE_SYSTEM_PROMPT_ID, SYSTEM_PROMPTS
from src.rag.mappings import (
    HAUL_HOOK_MAP,
    SEGMENT_MAP,
    get_amenity_hooks,
    get_party_description,
)


def build_user_prompt(lead: dict, policy_context: str = "[CONTEXT_PLACEHOLDER]") -> str:
    first_name = lead["customer_name"].split()[0]
    origin_city = lead["booking_origin"]
    hook = HAUL_HOOK_MAP.get(lead["haul_type"], "")
    party_desc = get_party_description(lead["num_passengers"])
    segment_desc = SEGMENT_MAP.get(lead["segment"], "valued traveler")
    amenity_hooks = get_amenity_hooks(lead)

    amenity_narrative = (
        "They have expressed interest in: " + ", ".join(amenity_hooks) + "."
        if amenity_hooks
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


def get_system_prompt(prompt_id: str = ACTIVE_SYSTEM_PROMPT_ID) -> str:
    return SYSTEM_PROMPTS[prompt_id]