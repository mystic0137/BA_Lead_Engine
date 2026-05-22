import io

import pandas as pd
import streamlit as st
import plotly.express as px

from tabs._shared import (
    REQUIRED_COLUMNS, SEGMENT_COLORS, score_csv, build_results,
)


def render():
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
            meta = st.session_state["score_meta"]

            st.divider()
            st.subheader("Segment Summary")

            seg_counts = results_df["segment"].value_counts()
            c0, c1, c2, c3 = st.columns(4)
            c0.metric("🔴 The Persuadable", seg_counts.get("The Persuadable", 0))
            c1.metric("🟢 The VIP", seg_counts.get("The VIP", 0))
            c2.metric("🟡 Window Shopper", seg_counts.get("The Window Shopper", 0))
            c3.metric("⚫ Lost Cause", seg_counts.get("The Lost Cause", 0))

            actionable = results_df[results_df["priority_score"] >= 2]
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
