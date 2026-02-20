"""Geo Anomaly Typology Agent — detects location-based inconsistencies in transactions."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from src.core.state import SARState

logger = logging.getLogger(__name__)


def geo_anomaly_agent(state: SARState) -> dict[str, Any]:
    """Detect geographic anomalies: impossible travel, multi-jurisdiction inconsistencies."""
    data = state.get("masked_data") or state.get("structured_data", {})
    txns = data.get("transactions", [])
    subject = data.get("subject", {})

    logger.info("Typology[geo_anomaly]: analyzing %d transactions", len(txns))

    findings: list[dict[str, Any]] = []
    subject_address = subject.get("address", "")

    # Sort by date
    sorted_txns = sorted(txns, key=lambda t: t.get("date", ""))

    # Detect geographic spread
    locations = set()
    for t in sorted_txns:
        loc = t.get("location")
        if loc:
            locations.add(loc)

    # Multiple distinct locations within short time
    for i in range(len(sorted_txns) - 1):
        a = sorted_txns[i]
        b = sorted_txns[i + 1]
        loc_a = a.get("location")
        loc_b = b.get("location")
        date_a = a.get("date", "")[:10]
        date_b = b.get("date", "")[:10]

        if loc_a and loc_b and loc_a != loc_b and date_a == date_b:
            findings.append({
                "pattern": "impossible_travel",
                "severity": "high",
                "count": 1,
                "detail": (
                    f"Transactions from different locations on same day: "
                    f"'{loc_a}' and '{loc_b}' on {date_a}"
                ),
                "evidence": [a.get("txn_id"), b.get("txn_id")],
            })

    # Cross-border flow asymmetry
    country_in: dict[str, float] = defaultdict(float)
    country_out: dict[str, float] = defaultdict(float)
    for t in txns:
        if t.get("from_country"):
            country_in[t["from_country"]] += t.get("amount", 0)
        if t.get("to_country"):
            country_out[t["to_country"]] += t.get("amount", 0)

    total_cross_border = sum(country_out.values())
    domestic = sum(t.get("amount", 0) for t in txns if not t.get("to_country"))
    if total_cross_border > 0 and domestic > 0:
        ratio = total_cross_border / (total_cross_border + domestic)
        if ratio > 0.6:
            findings.append({
                "pattern": "high_cross_border_ratio",
                "severity": "medium",
                "count": 1,
                "detail": (
                    f"Cross-border transactions represent {ratio:.0%} of total volume "
                    f"(${total_cross_border:,.2f}) — unusually high for customer profile"
                ),
                "evidence": list(country_out.keys()),
            })

    # Geographic diversity
    if len(locations) >= 4:
        findings.append({
            "pattern": "high_geographic_diversity",
            "severity": "medium",
            "count": len(locations),
            "detail": f"Activity across {len(locations)} distinct locations: {', '.join(list(locations)[:5])}",
            "evidence": list(locations),
        })

    return {
        "typology_results": {
            "geo_anomaly": {
                "findings": findings,
                "risk_score": min(len(findings) * 0.3, 1.0),
            },
        }
    }
