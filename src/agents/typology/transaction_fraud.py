"""Transaction Fraud Typology Agent — analyzes transaction amount/frequency/counterparty patterns."""

from __future__ import annotations

import logging
from typing import Any

from src.core.state import SARState

logger = logging.getLogger(__name__)


def transaction_fraud_agent(state: SARState) -> dict[str, Any]:
    """Analyze transactions for fraud patterns: round amounts, structuring, splits."""
    data = state.get("masked_data") or state.get("structured_data", {})
    txns = data.get("transactions", [])

    logger.info("Typology[transaction_fraud]: analyzing %d transactions", len(txns))

    findings: list[dict[str, Any]] = []

    # Round amount detection
    round_amounts = [t for t in txns if t.get("amount", 0) % 1000 == 0 and t["amount"] > 0]
    if round_amounts:
        findings.append({
            "pattern": "round_amount_transactions",
            "severity": "medium",
            "count": len(round_amounts),
            "detail": f"{len(round_amounts)} transactions with round amounts (multiples of $1,000)",
            "evidence": [t["txn_id"] for t in round_amounts],
        })

    # Just-below-threshold detection (structuring)
    near_threshold = [t for t in txns if 8000 <= t.get("amount", 0) < 10000]
    if len(near_threshold) >= 2:
        total = sum(t["amount"] for t in near_threshold)
        findings.append({
            "pattern": "structuring_below_ctr_threshold",
            "severity": "high",
            "count": len(near_threshold),
            "detail": (
                f"{len(near_threshold)} transactions between $8,000-$10,000 "
                f"totaling ${total:,.2f} — possible structuring to avoid CTR"
            ),
            "evidence": [t["txn_id"] for t in near_threshold],
        })

    # Counterparty concentration
    counterparties: dict[str, float] = {}
    for t in txns:
        cp = t.get("to_entity") or t.get("from_entity") or "unknown"
        counterparties[cp] = counterparties.get(cp, 0) + t.get("amount", 0)

    for cp, total in counterparties.items():
        if total > 50000 and cp != "unknown":
            findings.append({
                "pattern": "counterparty_concentration",
                "severity": "medium",
                "count": 1,
                "detail": f"High volume with single counterparty '{cp}': ${total:,.2f}",
                "evidence": [cp],
            })

    return {
        "typology_results": {
            "transaction_fraud": {
                "findings": findings,
                "risk_score": min(len(findings) * 0.25, 1.0),
            },
        }
    }
