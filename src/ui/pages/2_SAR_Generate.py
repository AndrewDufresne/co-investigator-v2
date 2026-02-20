"""Page 2: SAR Generate — run the full agent pipeline on uploaded case data."""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.session import init_session_state, update_from_graph_state
from src.ui.components.progress_tracker import render_progress_tracker
from src.ui.components.risk_charts import render_crime_type_chart, render_risk_indicators
from src.graph.sar_graph import build_sar_graph
from src.config import get_settings

init_session_state()

st.title("🔍 SAR Generation Pipeline")

# ── Check prerequisites ──
if not st.session_state.get("case_data"):
    st.warning("⚠️ No case data loaded. Please go to **Case Upload** first.")
    st.stop()

# ── Pipeline control ──
col1, col2, col3 = st.columns(3)

with col1:
    run_pipeline = st.button(
        "▶️ Run Pipeline",
        type="primary",
        disabled=st.session_state.execution_status == "running",
        width='stretch',
    )

with col2:
    resume = st.button(
        "⏩ Resume (Approve & Continue)",
        disabled=st.session_state.execution_status not in ("paused", "review"),
        width='stretch',
    )

with col3:
    st.write(f"**Status:** {st.session_state.execution_status}")

st.divider()

# ── Run pipeline ──
if run_pipeline:
    st.session_state.execution_status = "running"
    st.session_state.thread_id = str(uuid.uuid4())

    # Build graph
    with st.spinner("Building agent graph..."):
        app = build_sar_graph(interrupt_before=["unmask"])
        st.session_state.graph_app = app

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    initial_state = {
        "raw_data": st.session_state.case_data,
        "iteration_count": 0,
        "max_iterations": get_settings().max_iterations,
    }

    # Stream execution
    progress = st.empty()
    status_container = st.container()
    completed_nodes = []

    with status_container:
        with st.status("Running SAR pipeline...", expanded=True) as status_widget:
            try:
                for event in app.stream(initial_state, config, stream_mode="updates"):
                    for node_name, node_output in event.items():
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        st.write(f"✅ `{timestamp}` — **{node_name}** completed")

                        completed_nodes.append(node_name)
                        st.session_state.execution_log.append({
                            "timestamp": timestamp,
                            "node": node_name,
                            "message": "Completed",
                            "level": "info",
                        })

                        # Update session state from outputs
                        if isinstance(node_output, dict):
                            update_from_graph_state(node_output)

                # Check if interrupted (paused at unmask)
                snapshot = app.get_state(config)
                if snapshot.next:
                    st.session_state.execution_status = "review"
                    status_widget.update(label="⏸️ Pipeline paused — awaiting human review", state="complete")
                else:
                    st.session_state.execution_status = "completed"
                    status_widget.update(label="✅ Pipeline completed", state="complete")

            except Exception as e:
                st.session_state.execution_status = "error"
                status_widget.update(label=f"❌ Error: {e}", state="error")
                st.error(f"Pipeline error: {e}")

    st.rerun()

# ── Resume pipeline (after HITL review) ──
if resume and st.session_state.get("graph_app"):
    app = st.session_state.graph_app
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    st.session_state.execution_status = "running"

    with st.status("Resuming pipeline...", expanded=True) as status_widget:
        try:
            for event in app.stream(None, config, stream_mode="updates"):
                for node_name, node_output in event.items():
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    st.write(f"✅ `{timestamp}` — **{node_name}** completed")
                    if isinstance(node_output, dict):
                        update_from_graph_state(node_output)

            st.session_state.execution_status = "completed"
            status_widget.update(label="✅ Pipeline completed", state="complete")
        except Exception as e:
            st.session_state.execution_status = "error"
            status_widget.update(label=f"❌ Error: {e}", state="error")

    st.rerun()

# ── Display current results ──
col_left, col_right = st.columns([1, 1])

with col_left:
    render_crime_type_chart(st.session_state.get("crime_types"))

with col_right:
    render_risk_indicators(st.session_state.get("risk_indicators"))

# Show narrative preview if available
if st.session_state.get("narrative_draft"):
    st.divider()
    st.subheader("📝 Narrative Draft Preview")
    st.markdown(st.session_state.narrative_draft[:2000])
    if len(st.session_state.narrative_draft) > 2000:
        st.caption("... (navigate to Narrative Review for full text)")

# Show compliance score
if st.session_state.get("compliance_result"):
    st.divider()
    result = st.session_state.compliance_result
    score = result.get("overall_score", 0)
    status_text = result.get("status", "UNKNOWN")
    color = "🟢" if status_text == "PASS" else "🔴"
    st.metric("Compliance Result", f"{color} {status_text} ({score:.1%})")
