"""Text Content Typology Agent — NLP analysis of communications and transaction notes."""

from __future__ import annotations

import logging
import re
from typing import Any

from src.core.state import SARState

logger = logging.getLogger(__name__)

# Suspicious keyword categories
SUSPICIOUS_KEYWORDS = {
    "urgency": ["urgent", "asap", "immediately", "hurry", "rush", "fast", "quick"],
    "secrecy": ["confidential", "secret", "don't tell", "between us", "private", "no one else"],
    "cash_related": ["cash", "withdraw", "untraceable", "off the books", "under the table"],
    "structuring": ["split", "break up", "smaller amounts", "below limit", "avoid reporting"],
    "offshore": ["offshore", "overseas account", "foreign account", "transfer abroad", "shell company"],
    "pressure": ["must do today", "can't wait", "deadline", "no questions", "just do it"],
}


def text_content_agent(state: SARState) -> dict[str, Any]:
    """Analyze textual content in communications and transaction descriptions for red flags."""
    data = state.get("masked_data") or state.get("structured_data", {})
    flagged_comms = data.get("flagged_communications", [])
    txns = data.get("transactions", [])

    logger.info(
        "Typology[text_content]: analyzing %d communications, %d transaction notes",
        len(flagged_comms),
        len(txns),
    )

    findings: list[dict[str, Any]] = []
    all_text_pieces: list[str] = []

    # Collect text from communications
    for comm in flagged_comms:
        snippet = comm.get("content_snippet", "")
        if snippet:
            all_text_pieces.append(snippet)

    # Collect text from transaction descriptions
    for txn in txns:
        desc = txn.get("description", "")
        if desc:
            all_text_pieces.append(desc)

    # Scan for suspicious keywords
    combined_text = " ".join(all_text_pieces).lower()

    for category, keywords in SUSPICIOUS_KEYWORDS.items():
        matched_kw = [kw for kw in keywords if kw in combined_text]
        if matched_kw:
            findings.append({
                "pattern": f"suspicious_language_{category}",
                "severity": "high" if category in ("structuring", "secrecy") else "medium",
                "count": len(matched_kw),
                "detail": f"Detected {category} language: {', '.join(matched_kw)}",
                "evidence": matched_kw,
            })

    # Check for coded language patterns (short, vague descriptions with amounts)
    vague_txns = [
        t for t in txns
        if t.get("description") and len(t["description"]) < 15 and t.get("amount", 0) > 5000
    ]
    if vague_txns:
        findings.append({
            "pattern": "vague_high_value_descriptions",
            "severity": "medium",
            "count": len(vague_txns),
            "detail": (
                f"{len(vague_txns)} high-value transactions with unusually brief descriptions "
                f"— potential coded communication"
            ),
            "evidence": [t["txn_id"] for t in vague_txns],
        })

    return {
        "typology_results": {
            "text_content": {
                "findings": findings,
                "risk_score": min(len(findings) * 0.25, 1.0),
            },
        }
    }
