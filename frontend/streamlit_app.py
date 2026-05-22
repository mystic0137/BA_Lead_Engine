import logging

import streamlit as st

from tabs._shared import check_backend
from tabs import batch_scoring, lead_queue, model_analysis, email_generator

st.set_page_config(page_title="BA Lead Prioritization Engine", layout="wide")
st.title("British Airways — Lead Prioritization Engine")

logger = logging.getLogger(__name__)

if not check_backend():
    st.error("FastAPI backend offline or model not loaded. Run `make serve` first.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "Batch Scoring", "Lead Queue", "Model Analysis", "Email Generator"
])

with tab1:
    batch_scoring.render()

with tab2:
    lead_queue.render()

with tab3:
    model_analysis.render()

with tab4:
    email_generator.render()
