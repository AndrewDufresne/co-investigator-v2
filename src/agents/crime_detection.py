"""Crime Type Detection Agent — identifies financial crime typologies from case data.

Uses a rule-based risk indicator extraction engine (MVP).
Future: Add scikit-learn RF/GBM ensemble models.
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.state import SARState

logger = logging.getLogger(__name__)

# ── Rule-based risk indicator definitions ──

STRUCTURING_KEYWORDS = {"structured_amount", "just_below_threshold"}
LAYERING_KEYWORDS = {"layering_pattern", "rapid_movement", "internal_transfer"}
HIGH_RISK_JURISDICTIONS = {"AF", "MM", "KP", "IR", "SY", "BZ", "PA", "VU", "KY"}
SHELL_KEYWORDS = {"shell_company_indicator", "no_verifiable_operations", "minimal_history"}
FRAUD_KEYWORDS = {"identity_theft", "account_takeover", "unauthorized_access"}
VELOCITY_KEYWORDS = {"rapid_succession", "high_frequency", "burst_activity"}


def _extract_risk_indicators(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Scan structured data to extract risk indicators."""
    indicators: list[dict[str, Any]] = []
    transactions = data.get("transactions", [])

    # ── Structuring detection ──
    amounts_near_threshold = [
        t for t in transactions
        if 9000 <= t.get("amount", 0) < 10000
    ]
    if len(amounts_near_threshold) >= 2:
        indicators.append({
            "type": "structuring",
            "severity": "high",
            "description": (
                f"{len(amounts_near_threshold)} transactions between $9,000-$10,000 "
                f"detected — potential structuring to avoid CTR reporting"
            ),
            "evidence": [t["txn_id"] for t in amounts_near_threshold],
        })

    # ── Risk flag aggregation ──
    all_flags: list[str] = []
    for t in transactions:
        all_flags.extend(t.get("risk_flags", []))
    flag_set = set(all_flags)

    if flag_set & LAYERING_KEYWORDS:
        indicators.append({
            "type": "layering",
            "severity": "high",
            "description": "Transaction patterns consistent with layering — rapid internal transfers to obscure funds origin",
            "evidence": [f for f in all_flags if f in LAYERING_KEYWORDS],
        })

    if flag_set & SHELL_KEYWORDS:
        indicators.append({
            "type": "shell_company",
            "severity": "medium",
            "description": "Transactions involving entities with shell company characteristics",
            "evidence": [f for f in all_flags if f in SHELL_KEYWORDS],
        })

    if flag_set & VELOCITY_KEYWORDS:
        indicators.append({
            "type": "velocity_anomaly",
            "severity": "medium",
            "description": "Abnormal transaction velocity detected within a short time window",
            "evidence": [f for f in all_flags if f in VELOCITY_KEYWORDS],
        })

    # ── High-risk jurisdiction ──
    countries = set()
    for t in transactions:
        if t.get("from_country"):
            countries.add(t["from_country"])
        if t.get("to_country"):
            countries.add(t["to_country"])
    risky_countries = countries & HIGH_RISK_JURISDICTIONS
    if risky_countries:
        indicators.append({
            "type": "high_risk_jurisdiction",
            "severity": "high",
            "description": f"Transactions linked to high-risk jurisdictions: {', '.join(risky_countries)}",
            "evidence": list(risky_countries),
        })

    # ── KYC mismatch ──
    kyc = data.get("kyc", {})
    if kyc.get("activity_mismatch"):
        indicators.append({
            "type": "kyc_mismatch",
            "severity": "medium",
            "description": (
                f"Activity profile mismatch: expected '{kyc.get('expected_activity')}' "
                f"but actual is '{kyc.get('actual_activity_profile')}'"
            ),
            "evidence": ["activity_mismatch"],
        })

    # ── Adverse media ──
    if kyc.get("adverse_media_count", 0) > 0:
        indicators.append({
            "type": "adverse_media",
            "severity": "medium",
            "description": f"{kyc['adverse_media_count']} adverse media hit(s) found for subject",
            "evidence": [h.get("source", "") for h in kyc.get("adverse_media_hits", [])],
        })

    # ── Suspicious communications ──
    flagged_comms = data.get("flagged_communications", [])
    if flagged_comms:
        reasons = [c.get("flag_reason", "unknown") for c in flagged_comms]
        indicators.append({
            "type": "suspicious_communication",
            "severity": "medium",
            "description": f"Flagged communications detected: {', '.join(reasons)}",
            "evidence": reasons,
        })

    return indicators


def _classify_crime_types(
    indicators: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map risk indicators to crime type classifications with confidence scores."""
    type_scores: dict[str, float] = {}
    type_evidence: dict[str, list[str]] = {}

    scoring_rules = {
        "structuring": {"structuring": 0.45, "velocity_anomaly": 0.15, "kyc_mismatch": 0.10},
        "money_laundering_layering": {
            "layering": 0.40,
            "shell_company": 0.20,
            "high_risk_jurisdiction": 0.20,
            "structuring": 0.10,
        },
        "shell_company_activity": {
            "shell_company": 0.40,
            "high_risk_jurisdiction": 0.25,
            "layering": 0.10,
        },
        "fraud": {"suspicious_communication": 0.30, "kyc_mismatch": 0.20, "adverse_media": 0.20},
    }

    indicator_types = {ind["type"] for ind in indicators}

    for crime_type, rules in scoring_rules.items():
        score = 0.0
        evidence = []
        for indicator_type, weight in rules.items():
            if indicator_type in indicator_types:
                score += weight
                evidence.append(indicator_type)
        if score > 0.15:  # minimum threshold
            type_scores[crime_type] = min(score, 0.99)
            type_evidence[crime_type] = evidence

    # Sort by confidence descending
    results = [
        {"type": ct, "confidence": round(score, 2), "evidence": type_evidence[ct]}
        for ct, score in sorted(type_scores.items(), key=lambda x: -x[1])
    ]

    return results


def crime_detection_agent(state: SARState) -> dict[str, Any]:
    """Detect crime types from masked structured data.

    1. Extract risk indicators via rule engine
    2. Classify crime types with confidence scores
    """
    data = state.get("masked_data") or state.get("structured_data", {})
    logger.info("Crime detection: analyzing case %s", data.get("case_id", "unknown"))

    risk_indicators = _extract_risk_indicators(data)
    crime_types = _classify_crime_types(risk_indicators)

    logger.info(
        "Crime detection complete: %d indicators, %d crime types detected",
        len(risk_indicators),
        len(crime_types),
    )

    return {
        "risk_indicators": risk_indicators,
        "crime_types": crime_types,
    }
