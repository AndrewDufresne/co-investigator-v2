"""Page 2: SAR Generate — run the full agent pipeline on uploaded case data.

Features:
  - Real-time LLM token streaming for agents that call DeepSeek.
  - Per-agent I/O trace capture (input snapshot + output delta).
  - Pipeline run history recording (session-level).
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.session import (
    init_session_state,
    update_from_graph_state,
    create_run,
    add_trace_entry,
    finish_run,
    unmask_for_display,
)
from src.ui.components.progress_tracker import render_progress_tracker
from src.ui.components.risk_charts import render_crime_type_chart, render_risk_indicators
from src.ui.components.agent_trace_viewer import render_agent_trace
from src.graph.sar_graph import build_sar_graph
from src.core.models import AGENT_IO_FIELDS, LLM_AGENT_NODES
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


# ── Helper: extract input snapshot for a node from current session state ──

def _snapshot_input(node_name: str) -> dict:
    """Return a dict of the session-state fields this node reads as input."""
    fields = AGENT_IO_FIELDS.get(node_name, {}).get("input", [])
    snap: dict = {}
    for f in fields:
        val = st.session_state.get(f)
        if val is not None:
            # Truncate very large values for display
            snap[f] = _truncate(val)
    return snap


def _snapshot_output(node_name: str, node_output: dict) -> dict:
    """Return only the output fields this node is expected to write."""
    fields = AGENT_IO_FIELDS.get(node_name, {}).get("output", [])
    out: dict = {}
    for f in fields:
        if f in node_output:
            out[f] = _truncate(node_output[f])
    # Fallback: if no mapping, just use the raw output keys
    if not out:
        for k, v in node_output.items():
            out[k] = _truncate(v)
    return out


def _truncate(value, max_str: int = 500, max_list: int = 10):
    """Truncate large values for trace display."""
    if isinstance(value, str) and len(value) > max_str:
        return value[:max_str] + f"... ({len(value)} chars total)"
    if isinstance(value, list) and len(value) > max_list:
        return value[:max_list] + [f"... ({len(value)} items total)"]
    if isinstance(value, dict) and len(str(value)) > max_str:
        preview = str(value)[:max_str]
        return {"__truncated__": preview + "..."}
    return value


def _run_stream(app, initial_state, config, *, is_resume: bool = False):
    """Execute the LangGraph stream with dual mode (updates + messages).

    Handles:
      - Per-node progress updates (updates events)
      - Real-time LLM token display (messages events)
      - I/O trace capture per node
    """
    completed_nodes: list[str] = []
    # Track the current node for LLM streaming
    current_llm_node: str | None = None
    llm_text_buffer: str = ""
    llm_placeholder = None
    node_start_time: float | None = None
    input_snap: dict = {}

    label = "Resuming pipeline..." if is_resume else "Running SAR pipeline..."

    with st.status(label, expanded=True) as status_widget:
        try:
            stream_input = None if is_resume else initial_state

            for event in app.stream(
                stream_input,
                config,
                stream_mode=["updates", "messages"],
            ):
                mode, payload = event

                if mode == "messages":
                    # payload is a tuple (AIMessageChunk, metadata_dict)
                    chunk, metadata = payload
                    node = metadata.get("langgraph_node", "")

                    # Only render for LLM nodes
                    if node not in LLM_AGENT_NODES:
                        continue

                    # Start a new streaming block for this node
                    if node != current_llm_node:
                        # Flush previous buffer
                        current_llm_node = node
                        llm_text_buffer = ""
                        st.write(f"🤖 **{node}** — LLM streaming:")
                        llm_placeholder = st.empty()

                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if token:
                        llm_text_buffer += token
                        if llm_placeholder is not None:
                            llm_placeholder.markdown(f"```\n{llm_text_buffer}\n```")

                elif mode == "updates":
                    for node_name, node_output in payload.items():
                        # Skip LangGraph internal/system events (not real agent nodes)
                        if node_name.startswith("__") or node_name == "interrupt":
                            continue

                        # Calculate duration
                        t_now = time.time()
                        duration_ms = int((t_now - node_start_time) * 1000) if node_start_time else 0

                        timestamp = datetime.now().strftime("%H:%M:%S")
                        st.write(f"✅ `{timestamp}` — **{node_name}** completed")

                        completed_nodes.append(node_name)
                        st.session_state.execution_log.append({
                            "timestamp": timestamp,
                            "node": node_name,
                            "message": "Completed",
                            "level": "info",
                        })

                        # Capture I/O trace
                        in_snap = _snapshot_input(node_name)
                        out_snap = _snapshot_output(node_name, node_output) if isinstance(node_output, dict) else {}

                        has_llm = node_name in LLM_AGENT_NODES
                        llm_text = llm_text_buffer if (has_llm and llm_text_buffer) else None

                        add_trace_entry(
                            node_name=node_name,
                            started_at=datetime.now(timezone.utc).isoformat(),
                            finished_at=datetime.now(timezone.utc).isoformat(),
                            duration_ms=duration_ms,
                            input_snapshot=in_snap,
                            output_delta=out_snap,
                            has_llm_call=has_llm,
                            llm_stream_text=llm_text,
                        )

                        # Reset LLM buffer after node completion
                        if node_name == current_llm_node:
                            llm_text_buffer = ""
                            current_llm_node = None
                            llm_placeholder = None

                        # Update session state from outputs
                        if isinstance(node_output, dict):
                            update_from_graph_state(node_output)

                        # Start timing next node
                        node_start_time = time.time()

            # Check if interrupted (paused at unmask)
            snapshot = app.get_state(config)
            if snapshot.next:
                st.session_state.execution_status = "review"
                finish_run(status="review")
                status_widget.update(label="⏸️ Pipeline paused — awaiting human review", state="complete")
            else:
                st.session_state.execution_status = "completed"
                finish_run(status="completed")
                status_widget.update(label="✅ Pipeline completed", state="complete")

        except Exception as e:
            st.session_state.execution_status = "error"
            finish_run(status="error")
            status_widget.update(label=f"❌ Error: {e}", state="error")
            st.error(f"Pipeline error: {e}")


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

    # Create run history record
    create_run(mode="full")

    _run_stream(app, initial_state, config, is_resume=False)
    st.rerun()

# ── Resume pipeline (after HITL review) ──
if resume and st.session_state.get("graph_app"):
    app = st.session_state.graph_app
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    st.session_state.execution_status = "running"

    # Create a new run record for the resume leg
    create_run(mode="full-resume")

    _run_stream(app, None, config, is_resume=True)
    st.rerun()

# ── Display agent trace (if available) ──
current_run = st.session_state.get("current_run")
last_run_trace = None

# Prefer the current run; fall back to the most recent history entry
if current_run and current_run.get("agents_trace"):
    last_run_trace = current_run["agents_trace"]
else:
    history = st.session_state.get("run_history", [])
    if history and history[-1].get("agents_trace"):
        last_run_trace = history[-1]["agents_trace"]

if last_run_trace:
    st.divider()
    render_agent_trace(last_run_trace)

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
    _preview_text = unmask_for_display(st.session_state.narrative_draft)
    st.markdown(_preview_text[:2000])
    if len(_preview_text) > 2000:
        st.caption("... (navigate to Narrative Review for full text)")

# Show compliance score
if st.session_state.get("compliance_result"):
    st.divider()
    result = st.session_state.compliance_result
    score = result.get("overall_score", 0)
    status_text = result.get("status", "UNKNOWN")
    color = "🟢" if status_text == "PASS" else "🔴"
    st.metric("Compliance Result", f"{color} {status_text} ({score:.1%})")
