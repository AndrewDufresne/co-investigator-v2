"""Data Ingestion Agent — parses raw JSON case data into a structured summary.

This agent performs pure Python data transformation (no LLM calls).
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.state import SARState

logger = logging.getLogger(__name__)


def data_ingestion_agent(state: SARState) -> dict[str, Any]:
    """Parse raw_data JSON into a normalized structured_data dict.

    Extracts and organizes:
    - Subject profile
    - Account summaries
    - Transaction timeline with risk flags
    - KYC assessment
    - Communication flags
    - Alert summaries
    - Related entity map
    """
    raw = state.get("raw_data")
    if raw is None:
        raise ValueError("Missing required state key: raw_data")
    logger.info("Ingesting case %s", raw.get("case_id", "unknown"))

    # ── Subject summary ──
    subject = raw.get("subject", {})
    subject_summary = {
        "name": subject.get("name", "Unknown"),
        "dob": subject.get("dob"),
        "ssn": subject.get("ssn"),
        "address": subject.get("address"),
        "phone": subject.get("phone"),
        "email": subject.get("email"),
        "occupation": subject.get("occupation"),
        "risk_rating": subject.get("risk_rating", "medium"),
        "customer_since": subject.get("customer_since"),
    }

    # ── Account summaries ──
    accounts = []
    for acc in raw.get("accounts", []):
        accounts.append({
            "account_id": acc.get("account_id"),
            "account_type": acc.get("account_type"),
            "balance": acc.get("balance", 0),
            "currency": acc.get("currency", "USD"),
            "branch": acc.get("branch"),
        })

    # ── Transaction timeline ──
    transactions = []
    total_in = 0.0
    total_out = 0.0
    all_risk_flags: list[str] = []
    for txn in raw.get("transactions", []):
        t = {
            "txn_id": txn.get("txn_id"),
            "date": txn.get("date"),
            "type": txn.get("type"),
            "amount": txn.get("amount", 0),
            "currency": txn.get("currency", "USD"),
            "from_account": txn.get("from_account"),
            "to_account": txn.get("to_account"),
            "from_entity": txn.get("from_entity"),
            "to_entity": txn.get("to_entity"),
            "from_country": txn.get("from_country"),
            "to_country": txn.get("to_country"),
            "location": txn.get("location"),
            "description": txn.get("description"),
            "risk_flags": txn.get("risk_flags", []),
        }
        transactions.append(t)
        all_risk_flags.extend(t["risk_flags"])

        # Aggregate amounts
        if "in" in t["type"] or "deposit" in t["type"]:
            total_in += t["amount"]
        elif "out" in t["type"] or "withdrawal" in t["type"]:
            total_out += t["amount"]

    # ── KYC summary ──
    kyc_raw = raw.get("kyc", {})
    kyc_summary = {
        "verification_status": kyc_raw.get("verification_status", "unknown"),
        "source_of_funds": kyc_raw.get("source_of_funds"),
        "expected_activity": kyc_raw.get("expected_activity"),
        "actual_activity_profile": kyc_raw.get("actual_activity_profile"),
        "pep_status": kyc_raw.get("pep_status", False),
        "adverse_media_count": len(kyc_raw.get("adverse_media_hits", [])),
        "adverse_media_hits": kyc_raw.get("adverse_media_hits", []),
        "activity_mismatch": (
            kyc_raw.get("expected_activity") != kyc_raw.get("actual_activity_profile")
        ),
    }

    # ── Communication flags ──
    flagged_comms = [
        {
            "date": c.get("date"),
            "channel": c.get("channel"),
            "content_snippet": c.get("content", "")[:200],
            "flag_reason": c.get("flag_reason"),
        }
        for c in raw.get("communications", [])
        if c.get("flagged", False)
    ]

    # ── Alerts ──
    alerts = [
        {
            "alert_id": a.get("alert_id"),
            "type": a.get("type"),
            "severity": a.get("severity"),
            "description": a.get("description"),
            "triggered_date": a.get("triggered_date"),
        }
        for a in raw.get("alerts", [])
    ]

    # ── Related entities ──
    entities = [
        {
            "entity_name": e.get("entity_name"),
            "entity_type": e.get("entity_type"),
            "jurisdiction": e.get("jurisdiction"),
            "relationship": e.get("relationship"),
            "risk_notes": e.get("risk_notes"),
        }
        for e in raw.get("related_entities", [])
    ]

    structured = {
        "case_id": raw.get("case_id"),
        "alert_date": raw.get("alert_date"),
        "priority": raw.get("priority", "medium"),
        "subject": subject_summary,
        "accounts": accounts,
        "transactions": transactions,
        "transaction_summary": {
            "count": len(transactions),
            "total_inflow": total_in,
            "total_outflow": total_out,
            "date_range": {
                "start": transactions[0]["date"] if transactions else None,
                "end": transactions[-1]["date"] if transactions else None,
            },
            "unique_risk_flags": list(set(all_risk_flags)),
        },
        "kyc": kyc_summary,
        "flagged_communications": flagged_comms,
        "alerts": alerts,
        "related_entities": entities,
    }

    logger.info(
        "Ingestion complete: %d transactions, %d alerts, %d risk flags",
        len(transactions),
        len(alerts),
        len(all_risk_flags),
    )

    return {"structured_data": structured}
