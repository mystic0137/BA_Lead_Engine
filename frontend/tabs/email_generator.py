import urllib.parse
import logging

import requests
import streamlit as st

from tabs._shared import API_URL

logger = logging.getLogger(__name__)


def render():
    st.subheader("AI Email Generator")
    st.caption(
        "Generates BA policy-grounded outreach emails for high-value VIP leads. "
        "Powered by Llama-4-Scout via Groq + ChromaDB RAG."
    )

    if "results_df" not in st.session_state:
        st.info("Score leads in the Batch Scoring tab first.")
        return

    results_df = st.session_state["results_df"]

    nudge_leads = results_df[
        (results_df["priority_score"] == 2) &
        (results_df["value_tier"] == "High")
    ]

    if len(nudge_leads) == 0:
        st.warning("No high-value VIP leads found in current batch.")
    else:
        sample = nudge_leads.sample(
            n=min(5, len(nudge_leads)),
            random_state=42,
        ).reset_index(drop=True)

        st.write(f"**Selected {len(sample)} high-value VIP leads for outreach:**")
        st.dataframe(
            sample[[
                "customer_name", "route", "booking_origin",
                "haul_type", "num_passengers", "segment",
                "wants_extra_baggage", "wants_preferred_seat",
                "wants_in_flight_meals",
            ]],
            width='stretch',
        )

        st.divider()

        if st.button("Generate Emails", type="primary"):
            st.session_state["generated_emails"] = []
            progress = st.progress(0, text="Generating emails...")

            for i, (_, lead) in enumerate(sample.iterrows()):
                with st.spinner(f"Generating email for {lead['customer_name']}..."):
                    try:
                        lead_dict = lead.to_dict()
                        payload = {
                            "customer_id": lead_dict["customer_id"],
                            "customer_name": lead_dict["customer_name"],
                            "email": lead_dict["email"],
                            "route": lead_dict["route"],
                            "booking_origin": lead_dict["booking_origin"],
                            "haul_type": lead_dict["haul_type"],
                            "num_passengers": int(lead_dict["num_passengers"]),
                            "wants_extra_baggage": bool(lead_dict["wants_extra_baggage"]),
                            "wants_preferred_seat": bool(lead_dict["wants_preferred_seat"]),
                            "wants_in_flight_meals": bool(lead_dict["wants_in_flight_meals"]),
                        }
                        r = requests.post(
                            f"{API_URL}/api/v1/rag/generate",
                            json=payload,
                            timeout=60,
                        )
                        r.raise_for_status()
                        result = r.json()
                        st.session_state["generated_emails"].append({
                            "lead": lead_dict,
                            "email": result,
                        })
                    except Exception as e:
                        logger.exception(e)
                        st.error(f"Failed for {lead['customer_name']}: {e}")
                        continue

                progress.progress(
                    (i + 1) / len(sample),
                    text=f"Generated {i + 1}/{len(sample)}",
                )

            progress.empty()
            st.success(f"Generated {len(st.session_state['generated_emails'])} emails.")

    if "generated_emails" in st.session_state and st.session_state["generated_emails"]:
        st.divider()
        st.subheader("Generated Emails")

        for item in st.session_state["generated_emails"]:
            lead = item["lead"]
            email = item["email"]
            cid = lead["customer_id"]

            with st.expander(
                f"📧 {lead['customer_name']} — {lead['route']} ({lead['haul_type']})",
                expanded=True,
            ):
                col_meta, col_sources = st.columns([3, 1])
                with col_meta:
                    st.markdown(f"**To:** {lead['email']}")
                with col_sources:
                    st.caption("Policy sources used:")
                    for src in set(email["retrieved_sources"]):
                        st.caption(f"• {src}")

                st.divider()

                edited_subject = st.text_input(
                    "Subject", value=email["subject"], key=f"subject_{cid}"
                )
                edited_body = st.text_area(
                    "Email Body", value=email["body"], height=220, key=f"body_{cid}"
                )

                rating = st.feedback("stars", key=f"rating_{cid}")
                rating_value = (rating + 1) if rating is not None else 3

                choice_key = f"choice_{cid}"
                if choice_key not in st.session_state:
                    st.session_state[choice_key] = None

                col_accept, col_reject, col_save, col_draft, col_download = st.columns(
                    [1, 1, 1, 1, 2]
                )

                with col_accept:
                    if st.button("👍 Accept", key=f"acc_{cid}"):
                        st.session_state[choice_key] = "accepted"
                        st.rerun()

                with col_reject:
                    if st.button("👎 Reject", key=f"rej_{cid}"):
                        st.session_state[choice_key] = "rejected"
                        st.rerun()

                current_choice = st.session_state[choice_key]
                if current_choice == "accepted":
                    st.success("Current Selection: Accepted ✓")
                elif current_choice == "rejected":
                    st.error("Current Selection: Rejected ✗")

                with col_save:
                    if st.button("💾 Save Feedback", key=f"save_{cid}"):
                        choice = st.session_state.get(choice_key)
                        accepted = choice == "accepted" if choice is not None else None

                        feedback_payload = {
                            "customer_id": lead.get("customer_id", cid),
                            "customer_name": lead.get("customer_name"),
                            "email": lead.get("email"),
                            "route": lead.get("route"),
                            "booking_origin": lead.get("booking_origin"),
                            "haul_type": lead.get("haul_type"),
                            "num_passengers": lead.get("num_passengers"),
                            "wants_extra_baggage": lead.get("wants_extra_baggage"),
                            "wants_preferred_seat": lead.get("wants_preferred_seat"),
                            "wants_in_flight_meals": lead.get("wants_in_flight_meals"),
                            "retrieved_sources": email["retrieved_sources"],
                            "system_prompt_id": email.get("system_prompt_id", "ba_copywriter_v1"),
                            "generated_subject": email["subject"],
                            "generated_body": email["body"],
                            "edited_subject": edited_subject,
                            "edited_body": edited_body,
                            "rating": rating_value,
                            "accepted": accepted,
                            "tokens_input": email.get("tokens_input", 0),
                            "tokens_output": email.get("tokens_output", 0),
                            "latency_ms": email.get("latency_ms", 0),
                        }

                        try:
                            r = requests.post(
                                f"{API_URL}/api/v1/rag/feedback",
                                json=feedback_payload,
                                timeout=10,
                            )
                            r.raise_for_status()
                            st.session_state[choice_key] = None
                            st.success("Saved to feedback log.")
                        except Exception as e:
                            st.error(f"Failed to save feedback: {e}")

                with col_draft:
                    mailto = (
                        f"mailto:{urllib.parse.quote(lead['email'])}"
                        f"?subject={urllib.parse.quote(edited_subject)}"
                        f"&body={urllib.parse.quote(edited_body)}"
                    )
                    st.link_button("📨 Open in Outlook", url=mailto)

                with col_download:
                    st.download_button(
                        label="⬇️ Download Email (.txt)",
                        data=f"Subject: {edited_subject}\n\n{edited_body}",
                        file_name=f"email_{cid}.txt",
                        mime="text/plain",
                        key=f"dl_{cid}",
                    )

                st.caption(
                    "⚠️ Review before sending. Verify all claims against BA policy."
                )
