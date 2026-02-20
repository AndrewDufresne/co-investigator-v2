"""Account Health Typology Agent — assesses historical account behavior for anomalies."""

from __future__ import annotations

import logging
from typing import Any

from src.core.state import SARState

logger = logging.getLogger(__name__)


def account_health_agent(state: SARState) -> dict[str, Any]:
    """Analyze account characteristics and history for suspicious patterns."""
    data = state.get("masked_data") or state.get("structured_data", {})
    accounts = data.get("accounts", [])
    txns = data.get("transactions", [])
    kyc = data.get("kyc", {})

    logger.info("Typology[account_health]: analyzing %d accounts", len(accounts))

    findings: list[dict[str, Any]] = []

    # KYC/activity mismatch
    if kyc.get("activity_mismatch"):
        findings.append({
            "pattern": "profile_activity_mismatch",
            "severity": "high",
            "count": 1,
            "detail": (
                f"KYC declared activity '{kyc.get('expected_activity')}' does not match "
                f"actual behavior '{kyc.get('actual_activity_profile')}'"
            ),
            "evidence": ["activity_mismatch"],
        })

    # PEP status
    if kyc.get("pep_status"):
        findings.append({
            "pattern": "pep_involvement",
            "severity": "high",
            "count": 1,
            "detail": "Subject is a Politically Exposed Person (PEP) — enhanced due diligence required",
            "evidence": ["pep_status"],
        })

    # Multiple accounts
    if len(accounts) >= 3:
        findings.append({
            "pattern": "multiple_accounts",
            "severity": "medium",
            "count": len(accounts),
            "detail": f"Subject maintains {len(accounts)} accounts — possible fund distribution pattern",
            "evidence": [a.get("account_id") for a in accounts],
        })

    # High balance relative to stated activity
    for acc in accounts:
        balance = acc.get("balance", 0)
        if balance > 100000:
            findings.append({
                "pattern": "high_balance",
                "severity": "medium",
                "count": 1,
                "detail": (
                    f"Account {acc.get('account_id')} has balance ${balance:,.2f} "
                    f"— requires review against declared source of funds"
                ),
                "evidence": [acc.get("account_id")],
            })

    # Dormant account suddenly active (check if all transactions cluster)
    txn_dates = sorted(set(t.get("date", "")[:10] for t in txns if t.get("date")))
    if len(txn_dates) >= 2:
        active_days = len(txn_dates)
        if active_days <= 5 and len(txns) >= 10:
            findings.append({
                "pattern": "burst_after_dormancy",
                "severity": "high",
                "count": len(txns),
                "detail": (
                    f"{len(txns)} transactions concentrated within {active_days} days "
                    f"— possible previously dormant account now activated for illicit purpose"
                ),
                "evidence": txn_dates,
            })

    return {
        "typology_results": {
            "account_health": {
                "findings": findings,
                "risk_score": min(len(findings) * 0.25, 1.0),
            },
        }
    }
