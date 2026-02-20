"""Page 3: Narrative Review — human-in-the-loop narrative editing and feedback."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.session import init_session_state, update_from_graph_state
from src.ui.components.narrative_editor import (
    render_narrative_editor,
    render_compliance_result,
    render_feedback_form,
)

init_session_state()

st.title("✏️ Narrative Review")

if not st.session_state.get("narrative_draft"):
    st.warning("⚠️ No narrative generated yet. Please run the pipeline on **SAR Generate** page first.")
    st.stop()

# ── Display compliance result ──
render_compliance_result(
    st.session_state.get("compliance_result"),
    st.session_state.get("compliance_score"),
)

st.divider()

# ── Chain of Thought ──
cot = st.session_state.get("chain_of_thought")
if cot:
    with st.expander("🧠 Chain-of-Thought Reasoning", expanded=False):
        for i, step in enumerate(cot, 1):
            st.write(f"{i}. {step}")

# ── Narrative display/edit ──
is_final = st.session_state.get("final_narrative") is not None
narrative = st.session_state.get("final_narrative") or st.session_state.get("narrative_draft")

st.subheader("📄 SAR Narrative" + (" (Final)" if is_final else " (Draft)"))

if st.session_state.get("narrative_intro"):
    st.markdown(f"**Introduction:** {st.session_state.narrative_intro}")
    st.divider()

edited_narrative = st.text_area(
    "Narrative text (editable):",
    value=narrative,
    height=500,
    key="narrative_review_editor",
)

st.divider()

# ── Feedback / Approve section ──
iteration = st.session_state.get("iteration_count", 0)
max_iter = st.session_state.get("max_iterations", 3) if "max_iterations" in st.session_state else 3

st.caption(f"Iteration: {iteration} / {max_iter}")

if not is_final:
    feedback_result = render_feedback_form()

    if feedback_result == "__APPROVE__":
        # Approve: resume graph to unmask
        st.session_state.final_narrative = edited_narrative
        st.session_state.execution_status = "completed"

        # If graph is paused, resume it
        app = st.session_state.get("graph_app")
        if app and st.session_state.get("thread_id"):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            try:
                for event in app.stream(None, config, stream_mode="updates"):
                    for node_name, node_output in event.items():
                        if isinstance(node_output, dict):
                            update_from_graph_state(node_output)
            except Exception:
                pass  # May already be at END

        st.success("✅ Narrative approved and finalized!")
        st.rerun()

    elif feedback_result and feedback_result != "__APPROVE__":
        # Submit feedback for regeneration
        st.session_state.human_feedback = feedback_result

        app = st.session_state.get("graph_app")
        if app and st.session_state.get("thread_id"):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            # Update graph state with feedback
            app.update_state(
                config,
                {
                    "human_feedback": feedback_result,
                    "narrative_draft": edited_narrative,
                },
            )

            st.info("🔄 Regenerating narrative with your feedback...")

            with st.status("Regenerating..."):
                try:
                    for event in app.stream(None, config, stream_mode="updates"):
                        for node_name, node_output in event.items():
                            st.write(f"✅ **{node_name}** completed")
                            if isinstance(node_output, dict):
                                update_from_graph_state(node_output)
                except Exception as e:
                    st.error(f"Error during regeneration: {e}")

            st.rerun()
        else:
            st.warning("Graph not available. Please re-run the pipeline from SAR Generate page.")
else:
    # Final narrative
    st.success("✅ This narrative has been approved and finalized.")
    narrative_text = narrative if isinstance(narrative, str) else ""

    # Export options
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download Narrative (TXT)",
            data=narrative_text,
            file_name=f"SAR_{st.session_state.get('case_id', 'unknown')}.txt",
            mime="text/plain",
        )
    with col2:
        import json
        export_data = {
            "case_id": st.session_state.get("case_id"),
            "narrative": narrative_text,
            "compliance_score": st.session_state.get("compliance_score"),
            "crime_types": st.session_state.get("crime_types"),
            "risk_indicators": st.session_state.get("risk_indicators"),
            "iterations": iteration,
        }
        st.download_button(
            "📥 Download Full Report (JSON)",
            data=json.dumps(export_data, indent=2, ensure_ascii=False),
            file_name=f"SAR_Report_{st.session_state.get('case_id', 'unknown')}.json",
            mime="application/json",
        )
