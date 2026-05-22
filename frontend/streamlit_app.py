import json
import uuid
import random
import urllib.parse
import io
import os
from datetime import datetime

import numpy as np
import logging
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix, roc_curve, auc

from config import EXPORT_FIELDS

st.set_page_config(page_title="BA Lead Prioritization Engine", layout="wide")
st.title("British Airways — Lead Prioritization Engine")

logger = logging.getLogger(__name__)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

REQUIRED_COLUMNS = [
    "num_passengers", "sales_channel", "trip_type", "purchase_lead",
    "length_of_stay", "flight_hour", "flight_day", "route",
    "booking_origin", "wants_extra_baggage", "wants_preferred_seat",
    "wants_in_flight_meals", "flight_duration",
]

FAKE_NAMES = [
    "James Thornton", "Priya Nair", "Oliver Bennett", "Aisha Rahman",
    "Lucas Ferreira", "Sophie Müller", "Chen Wei", "Fatima Al-Amin",
    "Carlos Mendez", "Emma Larsson", "Ravi Shankar", "Nina Petrova",
    "David Okafor", "Yuki Tanaka", "Maria Costa", "Ahmed Hassan",
]

PRIORITY_META = {
    3: ("The Persuadable", "Schedule Call", "#e74c3c"),
    2: ("The VIP",         "Send Email",    "#2ecc71"),
    1: ("The Window Shopper", "Drip Sequence", "#f39c12"),
    0: ("The Lost Cause",  "No Action",     "#95a5a6"),
}

SEGMENT_COLORS = {v[0]: v[2] for v in PRIORITY_META.values()}


def generate_fake_contact(name: str) -> dict:
    email = name.lower().replace(" ", ".") + "@example.com"
    phone = f"+1{random.randint(200,999)}{random.randint(100,999)}{random.randint(1000,9999)}"
    return {"email": email, "phone": phone}


def check_backend() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json().get("model_loaded", False)
    except requests.exceptions.ConnectionError:
        return False


def _haul_type(duration: float) -> str:
    if duration < 3:
        return "Short Haul"
    elif duration <= 6:
        return "Medium Haul"
    return "Long Haul"


def score_csv(file) -> dict | None:
    try:
        r = requests.post(
            f"{API_URL}/predict/column_oriented",
            files={"file": (file.name, file.getvalue(), "text/csv")},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Scoring request failed: {e}")
        return None


def build_results(df: pd.DataFrame, response: dict) -> pd.DataFrame:
    pred = response["predictions"]
    biz  = pred["business_logic"]

    rows = []
    for i in range(len(pred["probability"])):
        priority = int(biz["priority_score"][i])
        segment, action, _ = PRIORITY_META[priority]
        name    = FAKE_NAMES[i % len(FAKE_NAMES)]
        contact = generate_fake_contact(name)
        row     = df.iloc[i]
        rows.append({
            "customer_id":          str(uuid.uuid4())[:8].upper(),
            "customer_name":        name,
            "email":                contact["email"],
            "phone":                contact["phone"],
            "route":                row["route"],
            "booking_origin":       row["booking_origin"],
            "haul_type":            _haul_type(row["flight_duration"]),
            "num_passengers":       row["num_passengers"],
            "wants_extra_baggage":  bool(row["wants_extra_baggage"]),
            "wants_preferred_seat": bool(row["wants_preferred_seat"]),
            "wants_in_flight_meals":bool(row["wants_in_flight_meals"]),
            "booking_probability":  pred["probability"][i],
            "booking_prediction":   pred["booking_prediction"][i],
            "priority_score":       priority,
            "value_tier":           biz["value_tier"][i],
            "expected_value_usd":   biz["expected_value_usd"][i],
            "potential_revenue_usd":biz["potential_revenue_usd"][i],
            "marginal_profit_usd":  biz["marginal_profit_usd"][i],
            "segment":              segment,
            "recommended_action":   action,
            "scored_at":            datetime.utcnow().isoformat(),
        })
    return pd.DataFrame(rows).sort_values("priority_score", ascending=False)


# ── Backend check ─────────────────────────────────────────────────────────────
if not check_backend():
    st.error("FastAPI backend offline or model not loaded. Run `make serve` first.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "Batch Scoring", "Lead Queue", "Model Analysis", "Email Generator"
])

# ── TAB 1: Batch Scoring ──────────────────────────────────────────────────────
with tab1:
    st.subheader("Upload Customer CSV")
    st.caption(f"Required columns: `{', '.join(REQUIRED_COLUMNS)}`")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded, encoding="latin1")
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        st.write(f"**{len(df)} records loaded.** Preview:")
        st.dataframe(df.head(5), width='stretch')

        sample_pct = st.slider(
            "Score a random sample (%)", 10, 100, 100, step=10,
            help="Score a subset for large files"
        )
        if sample_pct < 100:
            df = df.sample(frac=sample_pct / 100, random_state=42).reset_index(drop=True)
            st.info(f"Scoring {len(df)} records ({sample_pct}% sample)")

        if st.button("Score All Leads", type="primary"):
            csv_buffer = io.BytesIO(df.to_csv(index=False).encode())
            csv_buffer.name = uploaded.name

            with st.spinner("Scoring leads..."):
                response = score_csv(csv_buffer)

            if response is None:
                st.stop()

            results_df = build_results(df, response)
            st.session_state["results_df"] = results_df
            st.session_state["score_meta"] = response["meta"]
            st.success(f"Scored {len(results_df)} leads successfully.")

        if "results_df" in st.session_state:
            results_df = st.session_state["results_df"]
            meta       = st.session_state["score_meta"]

            st.divider()
            st.subheader("Segment Summary")

            seg_counts = results_df["segment"].value_counts()
            c0, c1, c2, c3 = st.columns(4)
            c0.metric("🔴 The Persuadable",  seg_counts.get("The Persuadable", 0))
            c1.metric("🟢 The VIP",          seg_counts.get("The VIP", 0))
            c2.metric("🟡 Window Shopper",   seg_counts.get("The Window Shopper", 0))
            c3.metric("⚫ Lost Cause",        seg_counts.get("The Lost Cause", 0))

            actionable     = results_df[results_df["priority_score"] >= 2]
            marketing_spend = (
                actionable["expected_value_usd"] - actionable["marginal_profit_usd"]
            ).sum()

            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Actionable Leads", f"{len(actionable):,}",
                      help="Persuadables + VIPs worth contacting")
            e2.metric("Expected Revenue (Actionable)",
                      f"${actionable['expected_value_usd'].sum():,.0f}",
                      help="Probability-weighted revenue from leads worth contacting")
            e3.metric("Avg Probability (Persuadable)",
                      f"{results_df[results_df['priority_score'] == 3]['booking_probability'].mean():.1%}")
            e4.metric("Total Marketing Spend", f"${marketing_spend:,.0f}",
                      help="Cost to action all Persuadable + VIP leads")

            st.caption(
                f"Model: `{meta['model_version']}` — "
                f"Threshold: `{meta['threshold_used']}`"
            )

            fig_pie = px.pie(
                results_df,
                names="segment",
                color="segment",
                color_discrete_map=SEGMENT_COLORS,
                title="Lead Segment Distribution",
                hole=0.4,
            )
            st.plotly_chart(fig_pie, width='stretch')

            st.divider()
            st.subheader("All Scored Leads")
            st.dataframe(
                results_df[[
                    "customer_name", "route", "segment", "value_tier",
                    "booking_probability", "recommended_action",
                    "potential_revenue_usd", "marginal_profit_usd", "priority_score",
                ]],
                width='stretch',
            )

# ── TAB 2: Lead Queue ─────────────────────────────────────────────────────────
with tab2:
    st.subheader("Priority Lead Queue")

    if "results_df" not in st.session_state:
        st.info("Score leads in the Batch Scoring tab first.")
    else:
        results_df = st.session_state["results_df"]

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            seg_filter = st.multiselect(
                "Filter by Segment",
                options=sorted(results_df["segment"].unique().tolist()),
                default=["The Persuadable", "The VIP"],
            )
        with col_f2:
            tier_filter = st.multiselect(
                "Filter by Value Tier",
                options=["High", "Medium", "Low"],
                default=["High", "Medium"],
            )
        with col_f3:
            action_filter = st.multiselect(
                "Filter by Recommended Action",
                options=sorted(results_df["recommended_action"].unique().tolist()),
                default=sorted(results_df["recommended_action"].unique().tolist()),
            )

        queue_df = results_df[
            results_df["segment"].isin(seg_filter) &
            results_df["value_tier"].isin(tier_filter) &
            results_df["recommended_action"].isin(action_filter)
        ].sort_values("priority_score", ascending=False)

        st.write(f"**{len(queue_df)} leads in queue**")

        rev_by_seg = (
            queue_df.groupby("segment")["potential_revenue_usd"]
            .sum()
            .reset_index()
            .sort_values("potential_revenue_usd", ascending=False)
        )
        fig_bar = px.bar(
            rev_by_seg,
            x="segment",
            y="potential_revenue_usd",
            color="segment",
            color_discrete_map=SEGMENT_COLORS,
            title="Pipeline Value by Segment (USD)",
            labels={
                "potential_revenue_usd": "Total Revenue (USD)",
                "segment": "Segment",
            },
        )
        st.plotly_chart(fig_bar, width='stretch')

        st.dataframe(
            queue_df[[
                "customer_name", "email", "phone", "route",
                "booking_origin", "segment", "value_tier",
                "booking_probability", "recommended_action",
                "potential_revenue_usd", "marginal_profit_usd",
            ]],
            width='stretch',
        )

        st.divider()
        col_json, col_csv = st.columns(2)
        with col_json:
            export_records = queue_df[
                [f for f in EXPORT_FIELDS if f in queue_df.columns]
            ].to_dict(orient="records")
            st.download_button(
                label="Download Priority Queue (JSONL)",
                data="\n".join(json.dumps(r) for r in export_records),
                file_name=f"ba_lead_queue_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.jsonl",
                mime="application/jsonl",
            )
        with col_csv:
            st.download_button(
                label="Download Full Results (CSV)",
                data=results_df.to_csv(index=False),
                file_name=f"ba_leads_full_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )

# ── TAB 3: Model Analysis ─────────────────────────────────────────────────────
with tab3:
    st.subheader("Model Performance Analysis")
    st.caption(
        "Upload CSV with a `booking_complete` ground truth column to evaluate model performance."
    )

    eval_file = st.file_uploader("Upload CSV with ground truth", type=["csv"], key="eval")

    if eval_file:
        eval_df = pd.read_csv(eval_file, encoding="latin1")

        if "booking_complete" not in eval_df.columns:
            st.error("CSV must contain a `booking_complete` column (0/1).")
            st.stop()

        missing_eval = [c for c in REQUIRED_COLUMNS if c not in eval_df.columns]
        if missing_eval:
            st.error(f"Missing columns: {missing_eval}")
            st.stop()

        threshold = st.slider(
            "Classification Threshold", 0.0, 1.0, 0.309, step=0.01,
            help="Adjust to see precision/recall tradeoff in real time"
        )

        y_true = eval_df["booking_complete"].values

        if st.button("Run Evaluation", type="primary"):
            eval_file.seek(0)
            with st.spinner("Evaluating..."):
                response = score_csv(eval_file)

            if response is None:
                st.stop()

            probs = np.array(response["predictions"]["probability"])
            st.session_state["eval_probs"]  = probs
            st.session_state["eval_y_true"] = y_true

        if "eval_probs" in st.session_state:
            probs_arr   = st.session_state["eval_probs"]
            y_true_arr  = st.session_state["eval_y_true"]
            preds       = (probs_arr >= threshold).astype(int)

            cm                  = confusion_matrix(y_true_arr, preds)
            tn, fp, fn, tp      = cm.ravel()
            precision           = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall              = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1                  = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0
            )
            fpr_arr, tpr_arr, _ = roc_curve(y_true_arr, probs_arr)
            roc_auc             = auc(fpr_arr, tpr_arr)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("ROC-AUC",   f"{roc_auc:.4f}")
            m2.metric("Precision", f"{precision:.4f}")
            m3.metric("Recall",    f"{recall:.4f}")
            m4.metric("F1",        f"{f1:.4f}")

            col_cm, col_roc = st.columns(2)

            with col_cm:
                fig_cm = go.Figure(go.Heatmap(
                    z=cm,
                    x=["Pred: No Booking", "Pred: Booking"],
                    y=["Actual: No Booking", "Actual: Booking"],
                    text=cm,
                    texttemplate="%{text}",
                    colorscale="Blues",
                    showscale=False,
                ))
                fig_cm.update_layout(title=f"Confusion Matrix (threshold={threshold:.3f})")
                st.plotly_chart(fig_cm, width='stretch')

            with col_roc:
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr_arr, y=tpr_arr,
                    mode="lines",
                    name=f"ROC (AUC={roc_auc:.3f})",
                    line=dict(color="#2ecc71", width=2),
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode="lines",
                    name="Random",
                    line=dict(color="#95a5a6", dash="dash"),
                ))
                fig_roc.update_layout(
                    title="ROC Curve",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                )
                st.plotly_chart(fig_roc, width='stretch')

            fig_dist = go.Figure()
            for label, color, name in [
                (0, "#95a5a6", "No Booking"),
                (1, "#e74c3c", "Booking"),
            ]:
                fig_dist.add_trace(go.Histogram(
                    x=probs_arr[y_true_arr == label],
                    name=f"Actual: {name}",
                    opacity=0.6,
                    marker_color=color,
                    nbinsx=40,
                ))
            fig_dist.add_vline(
                x=threshold,
                line_dash="dash",
                line_color="white",
                annotation_text=f"Threshold {threshold:.3f}",
            )
            fig_dist.update_layout(
                title="Predicted Probability Distribution",
                xaxis_title="Booking Probability",
                yaxis_title="Count",
                barmode="overlay",
            )
            st.plotly_chart(fig_dist, width='stretch')

# ── TAB 4: Email Generator ────────────────────────────────────────────────────
with tab4:
    st.subheader("AI Email Generator")
    st.caption(
        "Generates BA policy-grounded outreach emails for high-value VIP leads. "
        "Powered by Llama-4-Scout via Groq + ChromaDB RAG."
    )

    if "results_df" not in st.session_state:
        st.info("Score leads in the Batch Scoring tab first.")
    else:
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
                                "lead":  lead_dict,
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
                lead  = item["lead"]
                email = item["email"]
                cid   = lead["customer_id"]

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

                    rating       = st.feedback("stars", key=f"rating_{cid}")
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
                            choice   = st.session_state.get(choice_key)
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