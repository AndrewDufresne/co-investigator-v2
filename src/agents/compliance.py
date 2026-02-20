"""Compliance Validation Agent — Agent-as-a-Judge for SAR narrative quality.

Performs dual validation:
1. Rule-based checks for required SAR elements
2. LLM-based semantic coherence and regulatory alignment scoring
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.core.llm_gateway import get_llm
from src.core.state import SARState
from src.config import get_settings

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are a Compliance Validation Agent (Agent-as-a-Judge) evaluating a SAR narrative.

Evaluate the narrative across these dimensions, scoring each from 0.0 to 1.0:
1. completeness: Does the narrative cover all required SAR elements (who, what, when, where, how)?
2. accuracy: Are facts consistent with the source data? Are there any hallucinated details?
3. coherence: Is the narrative logically structured and easy to follow?
4. regulatory_alignment: Does it meet FinCEN narrative requirements?
5. evidence_citation: Are specific transactions, dates, and amounts referenced?
6. objectivity: Is the language factual and non-speculative? No tipping off?
7. actionability: Would this narrative be useful for law enforcement?

Respond ONLY with valid JSON:
{
    "scores": {
        "completeness": 0.0,
        "accuracy": 0.0,
        "coherence": 0.0,
        "regulatory_alignment": 0.0,
        "evidence_citation": 0.0,
        "objectivity": 0.0,
        "actionability": 0.0
    },
    "overall_score": 0.0,
    "issues": ["issue1", "issue2"],
    "improvement_suggestions": ["suggestion1", "suggestion2"]
}"""


def _rule_based_checks(state: SARState) -> list[dict[str, Any]]:
    """Perform deterministic rule-based compliance checks."""
    narrative = state.get("narrative_draft", "")
    data = state.get("masked_data") or state.get("structured_data", {})
    checks: list[dict[str, Any]] = []

    # Check 1: Narrative length
    word_count = len(narrative.split())
    checks.append({
        "dimension": "minimum_length",
        "passed": word_count >= 100,
        "score": min(word_count / 200, 1.0),
        "details": f"Narrative has {word_count} words (minimum recommended: 200)",
    })

    # Check 2: Date references
    date_pattern = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b", re.IGNORECASE)
    dates_found = date_pattern.findall(narrative)
    checks.append({
        "dimension": "date_references",
        "passed": len(dates_found) >= 1,
        "score": min(len(dates_found) / 3, 1.0),
        "details": f"Found {len(dates_found)} date reference(s) in narrative",
    })

    # Check 3: Amount references
    amount_pattern = re.compile(r"\$[\d,]+\.?\d*|\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b")
    amounts_found = amount_pattern.findall(narrative)
    checks.append({
        "dimension": "amount_references",
        "passed": len(amounts_found) >= 1,
        "score": min(len(amounts_found) / 5, 1.0),
        "details": f"Found {len(amounts_found)} amount reference(s) in narrative",
    })

    # Check 4: Transaction ID references
    txn_refs = [t.get("txn_id", "") for t in data.get("transactions", [])]
    referenced_txns = sum(1 for tid in txn_refs if tid and tid in narrative)
    total_txns = max(len(txn_refs), 1)
    checks.append({
        "dimension": "transaction_references",
        "passed": referenced_txns > 0,
        "score": referenced_txns / total_txns,
        "details": f"Referenced {referenced_txns}/{total_txns} transactions",
    })

    # Check 5: No tipping-off language
    tip_off_phrases = [
        "we are filing a SAR on you",
        "you are being investigated",
        "this report is about your suspicious",
        "we suspect you of",
    ]
    has_tip_off = any(phrase in narrative.lower() for phrase in tip_off_phrases)
    checks.append({
        "dimension": "no_tip_off",
        "passed": not has_tip_off,
        "score": 0.0 if has_tip_off else 1.0,
        "details": "No tipping-off language detected" if not has_tip_off else "WARNING: Potential tipping-off language found",
    })

    return checks


def compliance_validation_agent(state: SARState) -> dict[str, Any]:
    """Validate the SAR narrative using rule-based checks + LLM judge."""
    data = state.get("masked_data") or state.get("structured_data", {})
    case_id = data.get("case_id", "unknown")
    narrative = state.get("narrative_draft", "")
    settings = get_settings()

    logger.info("Compliance validation: evaluating narrative for case %s", case_id)

    # ── Part 1: Rule-based checks ──
    rule_checks = _rule_based_checks(state)
    rule_score = sum(c["score"] for c in rule_checks) / max(len(rule_checks), 1)

    # ── Part 2: LLM semantic evaluation ──
    llm_score = 0.0
    llm_issues: list[str] = []
    llm_suggestions: list[str] = []

    try:
        llm = get_llm(role="compliance_judge")

        eval_context = (
            f"=== NARRATIVE TO EVALUATE ===\n{narrative}\n\n"
            f"=== SOURCE DATA SUMMARY ===\n"
            f"Crime Types: {json.dumps(state.get('crime_types', []))}\n"
            f"Risk Indicators: {json.dumps(state.get('risk_indicators', []))}\n"
            f"Transaction Count: {len(data.get('transactions', []))}"
        )

        response = llm.invoke([
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=eval_context),
        ])

        raw_content = response.content
        if isinstance(raw_content, str):
            content = raw_content.strip()
        elif isinstance(raw_content, list):
            text_parts: list[str] = []
            for part in raw_content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
            content = "\n".join(text_parts).strip()
        else:
            content = str(raw_content).strip()

        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(content)

        llm_score = result.get("overall_score", 0.0)
        llm_issues = result.get("issues", [])
        llm_suggestions = result.get("improvement_suggestions", [])

    except Exception as e:
        logger.warning("LLM compliance evaluation failed (%s), using rule-based only", e)
        llm_score = rule_score  # fallback to rule score

    # ── Combine scores (60% LLM, 40% rule-based) ──
    overall_score = round(0.6 * llm_score + 0.4 * rule_score, 3)
    passed = overall_score >= settings.compliance_score_threshold

    compliance_result = {
        "status": "PASS" if passed else "FAIL",
        "overall_score": overall_score,
        "checks": rule_checks,
        "improvement_suggestions": llm_suggestions if not passed else [],
    }

    logger.info(
        "Compliance validation: %s (score=%.3f, threshold=%.2f)",
        compliance_result["status"],
        overall_score,
        settings.compliance_score_threshold,
    )

    return {
        "compliance_result": compliance_result,
        "compliance_score": overall_score,
    }
