import json
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px

from config import EXPORT_FIELDS
from tabs._shared import SEGMENT_COLORS


def render():
    st.subheader("Priority Lead Queue")

    if "results_df" not in st.session_state:
        st.info("Score leads in the Batch Scoring tab first.")
        return

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
