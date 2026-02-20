"""Country Risk Typology Agent — evaluates transactions linked to high-risk jurisdictions."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from src.core.state import SARState

logger = logging.getLogger(__name__)

# FATF-aligned risk classification
HIGH_RISK_COUNTRIES = {
    "AF": ("Afghanistan", "FATF Black List"),
    "MM": ("Myanmar", "FATF Black List"),
    "KP": ("North Korea", "FATF Black List"),
    "IR": ("Iran", "FATF Black List"),
    "SY": ("Syria", "OFAC Sanctioned"),
}

MONITORED_COUNTRIES = {
    "BZ": ("Belize", "FATF Grey List"),
    "PA": ("Panama", "FATF Grey List / Tax Haven"),
    "VU": ("Vanuatu", "FATF Grey List"),
    "KY": ("Cayman Islands", "Tax Haven"),
    "VG": ("British Virgin Islands", "Tax Haven"),
    "JE": ("Jersey", "Tax Haven"),
    "GG": ("Guernsey", "Tax Haven"),
    "IM": ("Isle of Man", "Tax Haven"),
}


def country_risk_agent(state: SARState) -> dict[str, Any]:
    """Evaluate geographic risk from transaction flows."""
    data = state.get("masked_data") or state.get("structured_data", {})
    txns = data.get("transactions", [])
    entities = data.get("related_entities", [])

    logger.info("Typology[country_risk]: analyzing %d transactions", len(txns))

    findings: list[dict[str, Any]] = []
    country_flows: dict[str, dict[str, float]] = defaultdict(lambda: {"inflow": 0, "outflow": 0, "count": 0})

    for t in txns:
        from_c = t.get("from_country", "")
        to_c = t.get("to_country", "")
        amount = t.get("amount", 0)

        if from_c:
            country_flows[from_c]["inflow"] += amount
            country_flows[from_c]["count"] += 1
        if to_c:
            country_flows[to_c]["outflow"] += amount
            country_flows[to_c]["count"] += 1

    # Check high-risk countries
    for code, (name, designation) in HIGH_RISK_COUNTRIES.items():
        if code in country_flows:
            flow = country_flows[code]
            total = flow["inflow"] + flow["outflow"]
            findings.append({
                "pattern": "high_risk_jurisdiction",
                "severity": "critical",
                "count": int(flow["count"]),
                "detail": (
                    f"Transactions involving {name} ({code}) — {designation}: "
                    f"${total:,.2f} across {int(flow['count'])} transactions"
                ),
                "evidence": [code],
            })

    # Check monitored countries
    for code, (name, designation) in MONITORED_COUNTRIES.items():
        if code in country_flows:
            flow = country_flows[code]
            total = flow["inflow"] + flow["outflow"]
            findings.append({
                "pattern": "monitored_jurisdiction",
                "severity": "high",
                "count": int(flow["count"]),
                "detail": (
                    f"Transactions involving {name} ({code}) — {designation}: "
                    f"${total:,.2f} across {int(flow['count'])} transactions"
                ),
                "evidence": [code],
            })

    # Entity jurisdiction risk
    for entity in entities:
        j = entity.get("jurisdiction", "")
        if j in HIGH_RISK_COUNTRIES or j in MONITORED_COUNTRIES:
            lookup = HIGH_RISK_COUNTRIES.get(j) or MONITORED_COUNTRIES.get(j)
            name, designation = lookup if lookup else (j, "Unknown")
            findings.append({
                "pattern": "entity_jurisdiction_risk",
                "severity": "high",
                "count": 1,
                "detail": (
                    f"Related entity '{entity.get('entity_name')}' registered in "
                    f"{name} ({j}) — {designation}"
                ),
                "evidence": [entity.get("entity_name"), j],
            })

    return {
        "typology_results": {
            "country_risk": {
                "findings": findings,
                "risk_score": min(len(findings) * 0.35, 1.0),
            },
        }
    }
