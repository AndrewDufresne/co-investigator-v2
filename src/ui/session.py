"""Streamlit session state management for the Argus app."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
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
        "mask_mapping": None,
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
        # History & tracing
        "run_history": [],       # list[PipelineRunRecord dicts]
        "current_run": None,     # current PipelineRunRecord dict (while running)
        "completed_cases": [],   # legacy — kept for backward compat
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
    st.session_state.mask_mapping = None
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
    st.session_state.current_run = None


def update_from_graph_state(state: dict[str, Any]) -> None:
    """Update session state from a LangGraph state snapshot."""
    field_mappings = [
        "structured_data", "masked_data", "mask_mapping", "risk_indicators", "crime_types",
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


# ── Pipeline Run Record helpers ──


def create_run(mode: str = "full") -> dict[str, Any]:
    """Create a new PipelineRunRecord and set it as current_run.

    Returns:
        The newly created run record dict.
    """
    run: dict[str, Any] = {
        "run_id": str(uuid.uuid4()),
        "case_id": st.session_state.get("case_id") or "unknown",
        "mode": mode,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "status": "running",
        "crime_types": [],
        "compliance_score": None,
        "compliance_status": None,
        "narrative_preview": "",
        "final_narrative": "",
        "iteration_count": 0,
        "agents_trace": [],
    }
    st.session_state.current_run = run
    return run


def add_trace_entry(
    node_name: str,
    started_at: str,
    finished_at: str,
    duration_ms: int,
    input_snapshot: dict[str, Any],
    output_delta: dict[str, Any],
    has_llm_call: bool = False,
    llm_stream_text: str | None = None,
) -> None:
    """Append an AgentTraceEntry to the current run."""
    run = st.session_state.get("current_run")
    if run is None:
        return
    entry: dict[str, Any] = {
        "node_name": node_name,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_ms": duration_ms,
        "input_snapshot": input_snapshot,
        "output_delta": output_delta,
        "has_llm_call": has_llm_call,
        "llm_stream_text": llm_stream_text,
    }
    run["agents_trace"].append(entry)


def finish_run(
    status: str = "completed",
) -> None:
    """Finalize the current run and append it to run_history."""
    run = st.session_state.get("current_run")
    if run is None:
        return

    run["finished_at"] = datetime.now(timezone.utc).isoformat()
    run["status"] = status

    # Populate summary fields from session state
    run["case_id"] = st.session_state.get("case_id") or run.get("case_id", "unknown")
    run["crime_types"] = st.session_state.get("crime_types") or []
    run["compliance_score"] = st.session_state.get("compliance_score")

    cr = st.session_state.get("compliance_result")
    run["compliance_status"] = cr.get("status") if isinstance(cr, dict) else None

    run["iteration_count"] = st.session_state.get("iteration_count", 0)

    narrative = st.session_state.get("final_narrative") or st.session_state.get("narrative_draft") or ""
    run["final_narrative"] = narrative
    run["narrative_preview"] = narrative[:200]

    # Append to history list
    history: list[dict[str, Any]] = st.session_state.get("run_history", [])
    history.append(run)
    st.session_state.run_history = history
    st.session_state.current_run = None


def update_run_status(status: str) -> None:
    """Update the status of the most recent run in run_history (e.g. approved/rejected)."""
    history: list[dict[str, Any]] = st.session_state.get("run_history", [])
    if history:
        history[-1]["status"] = status
        # Also refresh narrative in case it was finalized after approve
        narrative = st.session_state.get("final_narrative") or history[-1].get("final_narrative", "")
        history[-1]["final_narrative"] = narrative
        history[-1]["narrative_preview"] = narrative[:200]


# ── PII display helpers (UI-only, never persisted back to graph) ──


def unmask_for_display(text: str) -> str:
    """Replace PII placeholders with real values for UI display.

    Uses the mask_mapping (reverse map) stored in session state.
    This is purely cosmetic — session state data remains masked so that
    any value written back to the LangGraph state stays safe.
    """
    if not text:
        return text
    reverse_map: dict[str, str] = st.session_state.get("mask_mapping") or {}
    if not reverse_map:
        return text
    result = text
    for placeholder, original in reverse_map.items():
        result = result.replace(placeholder, original)
    return result


def remask_text(text: str) -> str:
    """Re-apply PII masking to user-edited text before sending back to LangGraph.

    Builds the forward map (original → placeholder) from the stored reverse map
    and replaces real values with their placeholders.
    """
    if not text:
        return text
    reverse_map: dict[str, str] = st.session_state.get("mask_mapping") or {}
    if not reverse_map:
        return text
    result = text
    # reverse_map is placeholder→original; we need original→placeholder
    for placeholder, original in reverse_map.items():
        result = result.replace(original, placeholder)
    return result
