"""Risk charts component — Plotly visualizations for risk analysis."""

from __future__ import annotations

from typing import Any

import streamlit as st

try:
    import plotly.graph_objects as go
    import plotly.express as px

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def render_crime_type_chart(crime_types: list[dict[str, Any]] | None) -> None:
    """Render crime type detection results as a horizontal bar chart."""
    if not crime_types:
        st.info("No crime types detected yet.")
        return

    st.subheader("🎯 Detected Crime Types")

    if not HAS_PLOTLY:
        # Fallback to text display
        for ct in crime_types:
            pct = ct.get("confidence", 0) * 100
            st.write(f"**{ct.get('type', 'Unknown')}**: {pct:.0f}% confidence")
            st.progress(ct.get("confidence", 0))
        return

    types = [ct.get("type", "unknown") for ct in crime_types]
    scores = [ct.get("confidence", 0) for ct in crime_types]

    fig = go.Figure(go.Bar(
        x=scores,
        y=types,
        orientation="h",
        marker_color=["#e74c3c" if s > 0.5 else "#f39c12" if s > 0.3 else "#3498db" for s in scores],
        text=[f"{s:.0%}" for s in scores],
        textposition="auto",
    ))
    fig.update_layout(
        title="Crime Type Confidence Scores",
        xaxis_title="Confidence",
        xaxis=dict(range=[0, 1]),
        height=max(200, len(types) * 60),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, width='stretch')


def render_risk_indicators(indicators: list[dict[str, Any]] | None) -> None:
    """Render risk indicators as categorized cards."""
    if not indicators:
        st.info("No risk indicators found yet.")
        return

    st.subheader("⚠️ Risk Indicators")

    for ind in indicators:
        severity = ind.get("severity", "unknown")
        icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(severity, "⚪")

        with st.container():
            st.markdown(f"{icon} **{ind.get('type', 'Unknown').replace('_', ' ').title()}** — *{severity}*")
            st.write(ind.get("description", ""))
            if ind.get("evidence"):
                st.caption(f"Evidence: {', '.join(str(e) for e in ind['evidence'])}")
            st.divider()


def render_typology_results(results: dict[str, Any] | None) -> None:
    """Render typology analysis results summary."""
    if not results:
        st.info("No typology analysis results yet.")
        return

    st.subheader("🧪 Typology Analysis Results")

    aggregate = results.get("_aggregate", {})
    if aggregate:
        cols = st.columns(3)
        cols[0].metric("Agents Run", len(aggregate.get("agents_run", [])))
        cols[1].metric("Total Findings", aggregate.get("total_findings", 0))
        cols[2].metric("Avg Risk Score", f"{aggregate.get('average_risk_score', 0):.1%}")

    for agent_name, agent_result in results.items():
        if agent_name.startswith("_") or not isinstance(agent_result, dict):
            continue

        findings = agent_result.get("findings", [])
        risk_score = agent_result.get("risk_score", 0)

        with st.expander(f"🔬 {agent_name.replace('_', ' ').title()} (score: {risk_score:.1%}, {len(findings)} findings)"):
            if not findings:
                st.write("No suspicious findings.")
                continue
            for f in findings:
                sev = f.get("severity", "unknown")
                icon = {"critical": "🔴", "high": "🟠", "medium": "🟡"}.get(sev, "🔵")
                st.write(f"{icon} **{f.get('pattern', 'Unknown')}**: {f.get('detail', '')}")
