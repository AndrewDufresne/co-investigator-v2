"""Argus AI — Streamlit multi-page app entry point.

Run with: streamlit run src/app.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.session import init_session_state

# ── Configure logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

# ── Page config ──
st.set_page_config(
    page_title="Argus AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialize session state ──
init_session_state()

# ── Multi-page navigation (st.navigation API) ──
pages = [
    st.Page("ui/pages/0_Home.py", title="Home", icon="🏠", default=True),
    st.Page("ui/pages/1_Case_Upload.py", title="Case Upload", icon="📄"),
    st.Page("ui/pages/2_SAR_Generate.py", title="SAR Generate", icon="🔍"),
    st.Page("ui/pages/3_Narrative_Review.py", title="Narrative Review", icon="✏️"),
    st.Page("ui/pages/4_Analysis_Dashboard.py", title="Analysis Dashboard", icon="📊"),
    st.Page("ui/pages/5_History.py", title="History", icon="📋"),
]

pg = st.navigation(pages)

# ── Sidebar ──
with st.sidebar:
    st.header("📊 Status")
    status = st.session_state.get("execution_status", "idle")
    status_icons = {
        "idle": "⬜ Idle",
        "running": "🔄 Running",
        "paused": "⏸️ Paused (Human Review)",
        "review": "⏸️ Paused (Human Review)",
        "completed": "✅ Completed",
        "error": "❌ Error",
    }
    st.write(f"**Pipeline:** {status_icons.get(status, status)}")

    if st.session_state.get("case_id"):
        st.write(f"**Case:** {st.session_state.case_id}")
    if st.session_state.get("compliance_score") is not None:
        st.write(f"**Compliance Score:** {st.session_state.compliance_score:.1%}")

    iteration = st.session_state.get("iteration_count", 0)
    if iteration > 0:
        st.write(f"**Iterations:** {iteration}")

# ── Run selected page ──
pg.run()
