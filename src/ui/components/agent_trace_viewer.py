"""Agent trace viewer component — renders per-agent I/O and LLM stream data."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st


def render_agent_trace(
    trace: list[dict[str, Any]],
    *,
    title: str = "📊 Pipeline Agent Trace",
    expanded_default: bool = False,
) -> None:
    """Render a list of AgentTraceEntry dicts as expandable sections.

    Args:
        trace: List of trace entry dicts from a pipeline run.
        title: Section title.
        expanded_default: Whether each agent expander starts open.
    """
    if not trace:
        st.info("No agent trace data available.")
        return

    st.subheader(title)

    for i, entry in enumerate(trace):
        node = entry.get("node_name", f"node-{i}")
        duration = entry.get("duration_ms", 0)
        has_llm = entry.get("has_llm_call", False)

        # Build label
        icon = "🚀" if has_llm else "⚙️"
        duration_str = f"{duration / 1000:.1f}s" if duration >= 1000 else f"{duration}ms"
        label = f"{icon} **{node}** ({duration_str})"
        if has_llm:
            label += "  •  LLM"

        with st.expander(label, expanded=expanded_default):
            col_in, col_out = st.columns(2)

            # ── Input snapshot ──
            with col_in:
                st.markdown("**📥 Input**")
                in_snap = entry.get("input_snapshot", {})
                if in_snap:
                    _render_dict(in_snap)
                else:
                    st.caption("(no tracked input)")

            # ── Output delta ──
            with col_out:
                st.markdown("**📤 Output**")
                out_snap = entry.get("output_delta", {})
                if out_snap:
                    _render_dict(out_snap)
                else:
                    st.caption("(no tracked output)")

            # ── LLM stream text ──
            if has_llm:
                llm_text = entry.get("llm_stream_text")
                if llm_text:
                    st.divider()
                    st.markdown("**💬 LLM Response**")
                    st.code(llm_text, language=None)

            # ── Timing ──
            st.caption(f"Started: {entry.get('started_at', '?')}  |  Finished: {entry.get('finished_at', '?')}")


def render_run_summary(run: dict[str, Any]) -> None:
    """Render a compact summary card for a single pipeline run."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Status", run.get("status", "—").upper())
    with col2:
        crime_count = len(run.get("crime_types", []))
        st.metric("Crime Types", crime_count)
    with col3:
        score = run.get("compliance_score")
        st.metric("Compliance", f"{score:.1%}" if score is not None else "—")
    with col4:
        st.metric("Iterations", run.get("iteration_count", 0))

    # Agent count
    trace = run.get("agents_trace", [])
    llm_count = sum(1 for e in trace if e.get("has_llm_call"))
    total_dur = sum(e.get("duration_ms", 0) for e in trace)
    dur_str = f"{total_dur / 1000:.1f}s" if total_dur >= 1000 else f"{total_dur}ms"
    st.caption(f"Agents: {len(trace)} ({llm_count} with LLM)  |  Total duration: {dur_str}")


# ── Private helpers ──

def _render_dict(d: dict[str, Any]) -> None:
    """Render a dict as key: value pairs, using json for complex values."""
    for key, val in d.items():
        if isinstance(val, str):
            # Truncate long strings in display
            display = val if len(val) <= 300 else val[:300] + "..."
            st.text_area(key, value=display, height=80, disabled=True, key=f"_trace_{key}_{id(val)}")
        elif isinstance(val, (list, dict)):
            try:
                formatted = json.dumps(val, indent=2, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                formatted = str(val)
            # Cap at reasonable length
            if len(formatted) > 2000:
                formatted = formatted[:2000] + "\n... (truncated)"
            st.code(formatted, language="json")
        else:
            st.write(f"**{key}:** {val}")
