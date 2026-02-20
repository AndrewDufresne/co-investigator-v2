"""Page 5: History — view completed case history."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.session import init_session_state

init_session_state()

st.title("📋 Case History")

completed = st.session_state.get("completed_cases", [])

if not completed:
    st.info("No completed cases yet. Generate a SAR report to see it here.")
    st.stop()

for i, case in enumerate(reversed(completed)):
    with st.expander(
        f"📄 {case.get('case_id', 'Unknown')} — {case.get('timestamp', 'N/A')}",
        expanded=(i == 0),
    ):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Crime Types", len(case.get("crime_types", [])))
        with col2:
            st.metric("Compliance Score", f"{case.get('compliance_score', 0):.1%}")
        with col3:
            st.metric("Iterations", case.get("iterations", 0))

        if case.get("narrative"):
            st.subheader("Narrative")
            st.markdown(case["narrative"][:500])
            if len(case.get("narrative", "")) > 500:
                with st.expander("Full narrative"):
                    st.markdown(case["narrative"])

        st.download_button(
            "📥 Download Report",
            data=json.dumps(case, indent=2, ensure_ascii=False),
            file_name=f"SAR_{case.get('case_id', 'unknown')}.json",
            mime="application/json",
            key=f"download_{i}",
        )
