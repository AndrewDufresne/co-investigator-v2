"""Payment Velocity Typology Agent — detects abnormal transaction frequency patterns."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from src.core.state import SARState

logger = logging.getLogger(__name__)


def payment_velocity_agent(state: SARState) -> dict[str, Any]:
    """Analyze transaction frequency and velocity for anomalies."""
    data = state.get("masked_data") or state.get("structured_data", {})
    txns = data.get("transactions", [])

    logger.info("Typology[payment_velocity]: analyzing %d transactions", len(txns))

    findings: list[dict[str, Any]] = []

    # Group by date
    daily_txns: dict[str, list[dict]] = defaultdict(list)
    for t in txns:
        date = t.get("date", "")[:10]  # YYYY-MM-DD
        if date:
            daily_txns[date].append(t)

    # High-frequency days
    for date, day_txns in daily_txns.items():
        if len(day_txns) >= 3:
            total = sum(t.get("amount", 0) for t in day_txns)
            findings.append({
                "pattern": "high_daily_frequency",
                "severity": "high" if len(day_txns) >= 5 else "medium",
                "count": len(day_txns),
                "detail": (
                    f"{len(day_txns)} transactions on {date} totaling ${total:,.2f} "
                    f"— above normal daily threshold"
                ),
                "evidence": [t["txn_id"] for t in day_txns],
            })

    # Burst detection: consecutive same-type transactions
    sorted_txns = sorted(txns, key=lambda t: t.get("date", ""))
    if len(sorted_txns) >= 3:
        for i in range(len(sorted_txns) - 2):
            a, b, c = sorted_txns[i], sorted_txns[i + 1], sorted_txns[i + 2]
            if (a.get("type") == b.get("type") == c.get("type") and
                    a.get("date", "")[:10] == b.get("date", "")[:10] == c.get("date", "")[:10]):
                findings.append({
                    "pattern": "burst_activity",
                    "severity": "medium",
                    "count": 3,
                    "detail": (
                        f"Burst of 3+ same-type '{a.get('type')}' transactions "
                        f"on {a.get('date', '')[:10]}"
                    ),
                    "evidence": [a["txn_id"], b["txn_id"], c["txn_id"]],
                })
                break  # Report first burst only

    # Total velocity
    if daily_txns:
        avg_per_day = len(txns) / len(daily_txns)
        if avg_per_day >= 3:
            findings.append({
                "pattern": "elevated_average_velocity",
                "severity": "medium",
                "count": len(txns),
                "detail": f"Average {avg_per_day:.1f} transactions/day over {len(daily_txns)} active days",
                "evidence": [],
            })

    return {
        "typology_results": {
            "payment_velocity": {
                "findings": findings,
                "risk_score": min(len(findings) * 0.3, 1.0),
            },
        }
    }
