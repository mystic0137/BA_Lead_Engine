import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix, roc_curve, auc

from tabs._shared import REQUIRED_COLUMNS, score_csv


def render():
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
            st.session_state["eval_probs"] = probs
            st.session_state["eval_y_true"] = y_true

        if "eval_probs" in st.session_state:
            probs_arr = st.session_state["eval_probs"]
            y_true_arr = st.session_state["eval_y_true"]
            preds = (probs_arr >= threshold).astype(int)

            cm = confusion_matrix(y_true_arr, preds)
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0
            )
            fpr_arr, tpr_arr, _ = roc_curve(y_true_arr, probs_arr)
            roc_auc = auc(fpr_arr, tpr_arr)

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
