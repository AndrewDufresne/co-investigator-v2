"""Dispute Pattern Typology Agent — analyzes dispute/chargeback patterns for anomalies."""

from __future__ import annotations

import logging
from typing import Any

from src.core.state import SARState

logger = logging.getLogger(__name__)


def dispute_pattern_agent(state: SARState) -> dict[str, Any]:
    """Analyze alert and dispute patterns for suspicious indicators."""
    data = state.get("masked_data") or state.get("structured_data", {})
    alerts = data.get("alerts", [])
    txns = data.get("transactions", [])

    logger.info("Typology[dispute_pattern]: analyzing %d alerts", len(alerts))

    findings: list[dict[str, Any]] = []

    # Alert severity distribution
    severity_counts: dict[str, int] = {}
    for alert in alerts:
        sev = alert.get("severity", "unknown")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    high_severity_count = severity_counts.get("high", 0) + severity_counts.get("critical", 0)
    if high_severity_count >= 2:
        findings.append({
            "pattern": "multiple_high_severity_alerts",
            "severity": "high",
            "count": high_severity_count,
            "detail": (
                f"{high_severity_count} high/critical severity alerts triggered — "
                f"indicates systemic suspicious activity"
            ),
            "evidence": [
                a.get("alert_id") for a in alerts
                if a.get("severity") in ("high", "critical")
            ],
        })

    # Alert type diversity
    alert_types = set(a.get("type", "") for a in alerts)
    if len(alert_types) >= 3:
        findings.append({
            "pattern": "diverse_alert_types",
            "severity": "medium",
            "count": len(alert_types),
            "detail": (
                f"Alerts across {len(alert_types)} different categories: "
                f"{', '.join(alert_types)} — multi-faceted suspicious activity"
            ),
            "evidence": list(alert_types),
        })

    # Repeated alerts of same type
    type_counts: dict[str, int] = {}
    for a in alerts:
        t = a.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    for alert_type, count in type_counts.items():
        if count >= 3:
            findings.append({
                "pattern": "repeated_alert_type",
                "severity": "high",
                "count": count,
                "detail": f"Alert type '{alert_type}' triggered {count} times — persistent pattern",
                "evidence": [
                    a.get("alert_id") for a in alerts if a.get("type") == alert_type
                ],
            })

    return {
        "typology_results": {
            "dispute_pattern": {
                "findings": findings,
                "risk_score": min(len(findings) * 0.3, 1.0),
            },
        }
    }
