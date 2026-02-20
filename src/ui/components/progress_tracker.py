"""Progress tracker component — real-time display of graph execution progress."""

from __future__ import annotations

from typing import Any

import streamlit as st

# Node display names and order
NODE_LABELS = {
    "ingest": ("📥 Data Ingestion", "Parsing and structuring case data"),
    "mask": ("🔒 Privacy Guard", "Masking PII before analysis"),
    "crime_detect": ("🔍 Crime Detection", "Identifying crime typologies"),
    "plan": ("📋 Planning", "Creating analysis execution plan"),
    "typology": ("🧪 Typology Analysis", "Running specialized pattern detection"),
    "external_intel": ("🌐 External Intel", "Gathering external intelligence"),
    "narrative": ("✏️ Narrative Generation", "Drafting SAR narrative"),
    "compliance": ("✅ Compliance Check", "Validating regulatory compliance"),
    "feedback": ("💬 Feedback Loop", "Integrating feedback for revision"),
    "unmask": ("🔓 Privacy Restore", "Restoring PII in final narrative"),
}

NODE_ORDER = [
    "ingest", "mask", "crime_detect", "plan", "typology",
    "external_intel", "narrative", "compliance", "feedback", "unmask",
]


def render_progress_tracker(
    completed_nodes: list[str],
    current_node: str | None = None,
    status: str = "idle",
) -> None:
    """Render a visual progress tracker showing pipeline stage."""
    st.subheader("🔄 Pipeline Progress")

    for node_id in NODE_ORDER:
        label, description = NODE_LABELS.get(node_id, (node_id, ""))

        if node_id in completed_nodes:
            st.success(f"✅ {label}")
        elif node_id == current_node:
            st.info(f"⏳ {label} — *{description}*")
        else:
            st.write(f"⬜ {label}")


def render_execution_log(log: list[dict[str, Any]]) -> None:
    """Render execution log entries."""
    if not log:
        return

    with st.expander("📜 Execution Log", expanded=False):
        for entry in reversed(log):
            ts = entry.get("timestamp", "")
            node = entry.get("node", "")
            msg = entry.get("message", "")
            level = entry.get("level", "info")

            if level == "error":
                st.error(f"[{ts}] **{node}**: {msg}")
            elif level == "warning":
                st.warning(f"[{ts}] **{node}**: {msg}")
            else:
                st.write(f"`{ts}` **{node}**: {msg}")
