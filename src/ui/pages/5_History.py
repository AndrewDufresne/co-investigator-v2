"""Page 5: History — view pipeline run history with agent traces."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.session import init_session_state, unmask_for_display
from src.ui.components.agent_trace_viewer import render_agent_trace, render_run_summary

init_session_state()

st.title("📋 Pipeline Run History")

history: list[dict] = st.session_state.get("run_history", [])

if not history:
    st.info("No pipeline runs yet in this session. Generate a SAR report to see it here.")
    st.stop()

st.caption(f"{len(history)} run(s) in current session")

for idx, run in enumerate(reversed(history)):
    run_num = len(history) - idx
    case_id = run.get("case_id", "Unknown")
    started = run.get("started_at", "N/A")
    status = run.get("status", "unknown")

    # Status badge
    status_icons = {
        "completed": "✅",
        "approved": "🟢",
        "rejected": "🔴",
        "review": "⏸️",
        "running": "🔄",
        "error": "❌",
    }
    icon = status_icons.get(status, "❓")

    with st.expander(
        f"{icon} Run #{run_num} — {case_id} — {status.upper()} — {started[:19]}",
        expanded=(idx == 0),
    ):
        # Summary metrics
        render_run_summary(run)

        st.divider()

        # Tabs: Narrative / Trace / Raw JSON
        tab_narrative, tab_trace, tab_json = st.tabs(["📄 Narrative", "📊 Agent Trace", "🗂️ Raw Data"])

        with tab_narrative:
            _raw_narr = run.get("final_narrative", "")
            narrative = unmask_for_display(_raw_narr) if _raw_narr else ""
            if narrative:
                st.markdown(narrative[:3000])
                if len(narrative) > 3000:
                    with st.expander("Show full narrative"):
                        st.markdown(narrative)
            else:
                st.caption("No narrative available for this run.")

            # Crime types
            crimes = run.get("crime_types", [])
            if crimes:
                st.subheader("Detected Crime Types")
                for ct in crimes:
                    ctype = ct.get("type", "?")
                    conf = ct.get("confidence", 0)
                    st.write(f"- **{ctype}**: {conf:.0%}")

        with tab_trace:
            trace = run.get("agents_trace", [])
            if trace:
                render_agent_trace(trace, title="", expanded_default=False)
            else:
                st.caption("No agent trace data for this run.")

        with tab_json:
            # Download full run record as JSON
            run_json = json.dumps(run, indent=2, ensure_ascii=False, default=str)
            st.download_button(
                "📥 Download Run Data (JSON)",
                data=run_json,
                file_name=f"run_{run.get('run_id', 'unknown')}.json",
                mime="application/json",
                key=f"dl_run_{idx}",
            )
            st.code(run_json[:5000], language="json")
            if len(run_json) > 5000:
                st.caption(f"... ({len(run_json)} chars total — download for full data)")
