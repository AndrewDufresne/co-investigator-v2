"""External Intelligence Agent — gathers external risk intelligence via MCP.

MVP: Returns mock external intelligence data.
Future: Integrate real MCP client for sanctions lists, news feeds, etc.
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.state import SARState

logger = logging.getLogger(__name__)


def external_intel_agent(state: SARState) -> dict[str, Any]:
    """Gather external intelligence relevant to the case.

    MVP: Generates mock intelligence based on case data.
    Future: Use MCP Client SDK to query external data sources.
    """
    data = state.get("masked_data", {})
    crime_types = state.get("crime_types", [])
    execution_plan = state.get("execution_plan", {})

    if not execution_plan.get("requires_external_intel", True):
        logger.info("External intel: skipped (not required by plan)")
        return {"external_intel": []}

    logger.info("External intel: gathering intelligence for case %s", data.get("case_id"))

    intel_results: list[dict[str, Any]] = []

    # ── Check related entities for risk signals ──
    for entity in data.get("related_entities", []):
        jurisdiction = entity.get("jurisdiction", "")
        risk_notes = entity.get("risk_notes", "")

        if risk_notes:
            intel_results.append({
                "source": "entity_risk_database",
                "entity": entity.get("entity_name"),
                "finding": risk_notes,
                "jurisdiction": jurisdiction,
                "relevance": "high",
            })

    # ── Check adverse media from KYC ──
    kyc = data.get("kyc", {})
    for hit in kyc.get("adverse_media_hits", []):
        intel_results.append({
            "source": f"media_{hit.get('source', 'unknown')}",
            "entity": data.get("subject", {}).get("name", "Unknown"),
            "finding": hit.get("summary", ""),
            "date": hit.get("date"),
            "relevance": "medium",
        })

    # ── Jurisdiction risk lookups ──
    high_risk_jurisdictions = {"BZ": "Belize", "PA": "Panama", "KY": "Cayman Islands"}
    countries = set()
    for txn in data.get("transactions", []):
        if txn.get("from_country"):
            countries.add(txn["from_country"])
        if txn.get("to_country"):
            countries.add(txn["to_country"])

    for code in countries:
        if code in high_risk_jurisdictions:
            intel_results.append({
                "source": "fatf_jurisdiction_monitor",
                "entity": high_risk_jurisdictions[code],
                "finding": f"{high_risk_jurisdictions[code]} ({code}) is on FATF monitored jurisdictions list",
                "relevance": "high",
            })

    logger.info("External intel: found %d intelligence items", len(intel_results))

    return {"external_intel": intel_results}
