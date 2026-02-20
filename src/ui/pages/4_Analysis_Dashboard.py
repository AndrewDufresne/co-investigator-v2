"""Page 4: Analysis Dashboard — visualize risk analysis and typology results."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.session import init_session_state
from src.ui.components.risk_charts import (
    render_crime_type_chart,
    render_risk_indicators,
    render_typology_results,
)

init_session_state()

st.title("📊 Analysis Dashboard")

if not st.session_state.get("crime_types") and not st.session_state.get("risk_indicators"):
    st.warning("⚠️ No analysis data available. Please run the pipeline first.")
    st.stop()

# ── Summary metrics ──
col1, col2, col3, col4 = st.columns(4)

crime_types = st.session_state.get("crime_types", [])
risk_indicators = st.session_state.get("risk_indicators", [])
typology_results = st.session_state.get("typology_results", {})
compliance_score = st.session_state.get("compliance_score")

with col1:
    st.metric("Crime Types", len(crime_types))
with col2:
    st.metric("Risk Indicators", len(risk_indicators))
with col3:
    aggregate = typology_results.get("_aggregate", {})
    st.metric("Typology Findings", aggregate.get("total_findings", 0))
with col4:
    if compliance_score is not None:
        st.metric("Compliance Score", f"{compliance_score:.1%}")

st.divider()

# ── Crime types chart ──
render_crime_type_chart(crime_types)

st.divider()

# ── Risk indicators ──
render_risk_indicators(risk_indicators)

st.divider()

# ── Typology results ──
render_typology_results(typology_results)

# ── External intel ──
external = st.session_state.get("external_intel", [])
if external:
    st.divider()
    st.subheader("🌐 External Intelligence")
    for item in external:
        with st.container():
            st.write(f"**Source:** {item.get('source', 'N/A')} | **Entity:** {item.get('entity', 'N/A')}")
            st.write(f"**Finding:** {item.get('finding', 'N/A')}")
            st.caption(f"Relevance: {item.get('relevance', 'N/A')}")
            st.divider()
