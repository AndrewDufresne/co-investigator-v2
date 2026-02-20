"""Streamlit session state management for the Co-Investigator app."""

from __future__ import annotations

import uuid
from typing import Any

import streamlit as st


def init_session_state() -> None:
    """Initialize all session state variables with defaults."""
    defaults: dict[str, Any] = {
        # App state
        "app_initialized": False,
        "current_page": "home",
        # Case data
        "case_data": None,
        "case_id": None,
        "case_file_name": None,
        # Graph execution
        "graph_app": None,
        "thread_id": None,
        "execution_status": "idle",  # idle | running | paused | completed | error
        "execution_log": [],
        # Results
        "structured_data": None,
        "masked_data": None,
        "risk_indicators": None,
        "crime_types": None,
        "typology_results": None,
        "external_intel": None,
        "narrative_draft": None,
        "narrative_intro": None,
        "chain_of_thought": None,
        "compliance_result": None,
        "compliance_score": None,
        "final_narrative": None,
        # HITL
        "human_feedback": None,
        "iteration_count": 0,
        # History
        "completed_cases": [],
    }

    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def reset_case_state() -> None:
    """Reset case-specific state for a new analysis."""
    st.session_state.case_data = None
    st.session_state.case_id = None
    st.session_state.case_file_name = None
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.execution_status = "idle"
    st.session_state.execution_log = []
    st.session_state.structured_data = None
    st.session_state.masked_data = None
    st.session_state.risk_indicators = None
    st.session_state.crime_types = None
    st.session_state.typology_results = None
    st.session_state.external_intel = None
    st.session_state.narrative_draft = None
    st.session_state.narrative_intro = None
    st.session_state.chain_of_thought = None
    st.session_state.compliance_result = None
    st.session_state.compliance_score = None
    st.session_state.final_narrative = None
    st.session_state.human_feedback = None
    st.session_state.iteration_count = 0


def update_from_graph_state(state: dict[str, Any]) -> None:
    """Update session state from a LangGraph state snapshot."""
    field_mappings = [
        "structured_data", "masked_data", "risk_indicators", "crime_types",
        "typology_results", "external_intel", "narrative_draft", "narrative_intro",
        "chain_of_thought", "compliance_result", "compliance_score",
        "final_narrative", "human_feedback", "iteration_count",
    ]
    for field in field_mappings:
        if field in state:
            st.session_state[field] = state[field]

    # Update case_id from raw_data or structured_data
    if "structured_data" in state and state["structured_data"]:
        st.session_state.case_id = state["structured_data"].get("case_id")

    # Update status based on state
    status = state.get("status")
    if status:
        st.session_state.execution_status = status
