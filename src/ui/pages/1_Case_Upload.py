"""Page 1: Case Upload — upload and preview JSON case files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.session import init_session_state, reset_case_state
from src.ui.components.case_viewer import render_case_overview
from src.agents.ingestion import data_ingestion_agent
from src.config import get_settings

init_session_state()

st.title("📄 Case Upload")

# ── Upload section ──
tab_upload, tab_sample = st.tabs(["Upload File", "Load Sample"])

with tab_upload:
    uploaded = st.file_uploader(
        "Upload a JSON case file",
        type=["json"],
        help="Upload a structured AML case file in JSON format",
    )
    if uploaded:
        try:
            raw_data = json.loads(uploaded.read().decode("utf-8"))
            reset_case_state()
            st.session_state.case_data = raw_data
            st.session_state.case_file_name = uploaded.name
            st.success(f"✅ Loaded case file: {uploaded.name}")
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")

with tab_sample:
    settings = get_settings()
    samples_dir = settings.samples_dir
    sample_files = sorted(samples_dir.glob("*.json")) if samples_dir.exists() else []

    if sample_files:
        selected = st.selectbox(
            "Select a sample case",
            options=sample_files,
            format_func=lambda p: p.stem.replace("_", " ").title() + " (Sample data is fictional; do not associate it with the real world)",
        )
        if st.button("Load Sample", type="primary"):
            raw_data = json.loads(selected.read_text(encoding="utf-8"))
            reset_case_state()
            st.session_state.case_data = raw_data
            st.session_state.case_file_name = selected.name
            st.success(f"✅ Loaded sample : {selected.name} (Sample data is fictional; do not associate it with the real world)")
    else:
        st.info(f"No sample files found in `{samples_dir}`. Add JSON files to that directory.")

# ── Preview section ──
if st.session_state.get("case_data"):
    st.divider()
    st.subheader("📋 Case Preview")

    # Run quick ingestion for preview
    if not st.session_state.get("structured_data"):
        with st.spinner("Parsing case data..."):
            result = data_ingestion_agent({"raw_data": st.session_state.case_data})
            st.session_state.structured_data = result["structured_data"]

    render_case_overview(st.session_state.structured_data)

    # Raw JSON toggle
    with st.expander("🔧 Raw JSON"):
        st.json(st.session_state.case_data)

    st.divider()
    st.info("✅ Case loaded. Navigate to **SAR Generate** page to run the analysis pipeline.")
