"""Narrative editor component — displays and allows editing of SAR narratives."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_narrative_editor(
    narrative: str | None,
    intro: str | None = None,
    chain_of_thought: list[str] | None = None,
    editable: bool = True,
) -> str | None:
    """Render the SAR narrative with optional editing capabilities.

    Returns the (possibly edited) narrative text, or None if no narrative exists.
    """
    if not narrative:
        st.info("No narrative generated yet.")
        return None

    # Chain of Thought reasoning
    if chain_of_thought:
        with st.expander("🧠 Chain-of-Thought Reasoning", expanded=False):
            for i, step in enumerate(chain_of_thought, 1):
                st.write(f"{i}. {step}")

    # Introduction section
    if intro:
        st.subheader("📝 Narrative Introduction")
        st.markdown(intro)
        st.divider()

    # Main narrative
    st.subheader("📄 SAR Narrative")

    if editable:
        edited = st.text_area(
            "Edit narrative below:",
            value=narrative,
            height=400,
            key="narrative_editor",
        )
        return edited
    else:
        st.markdown(narrative)
        return narrative


def render_compliance_result(result: dict[str, Any] | None, score: float | None = None) -> None:
    """Render compliance validation results."""
    if not result:
        st.info("Compliance validation not yet performed.")
        return

    st.subheader("✅ Compliance Validation")

    # Overall score
    status = result.get("status", "UNKNOWN")
    overall = result.get("overall_score", score or 0.0)

    col1, col2 = st.columns(2)
    with col1:
        color = "🟢" if status == "PASS" else "🔴"
        st.metric("Status", f"{color} {status}")
    with col2:
        st.metric("Overall Score", f"{overall:.1%}")

    # Progress bar
    st.progress(min(overall, 1.0))

    # Individual checks
    checks = result.get("checks", [])
    if checks:
        with st.expander("Detailed Checks", expanded=False):
            for check in checks:
                passed = check.get("passed", False)
                icon = "✅" if passed else "❌"
                st.write(f"{icon} **{check.get('dimension', 'Unknown')}** — {check.get('details', '')}")
                st.progress(min(check.get("score", 0), 1.0))

    # Improvement suggestions
    suggestions = result.get("improvement_suggestions", [])
    if suggestions:
        with st.expander("📋 Improvement Suggestions", expanded=True):
            for i, s in enumerate(suggestions, 1):
                st.write(f"{i}. {s}")


def render_feedback_form() -> str | None:
    """Render the human feedback form for HITL review.

    Returns the feedback text if submitted, None otherwise.
    """
    st.subheader("💬 Investigator Feedback")
    st.caption("Provide feedback to refine the narrative. The system will incorporate your input and regenerate.")

    feedback = st.text_area(
        "Your feedback:",
        placeholder="e.g., 'Add more detail about the wire transfers on March 15' or 'The conclusion needs stronger language about the structuring pattern'",
        height=150,
        key="feedback_input",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📤 Submit Feedback & Regenerate", type="primary", width='stretch'):
            if feedback.strip():
                return feedback.strip()
            st.warning("Please enter feedback before submitting.")
    with col2:
        if st.button("✅ Approve Narrative", width='stretch'):
            return "__APPROVE__"

    return None
