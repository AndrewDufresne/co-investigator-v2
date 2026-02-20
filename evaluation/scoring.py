"""Evaluation Scoring — multi-dimensional quality assessment for SAR narratives.

Dimensions:
  1. Completeness  — Are all required SAR elements present?
  2. Accuracy      — Are facts consistent with source data?
  3. Coherence     — Is the narrative logically structured?
  4. Regulatory    — Does it meet FinCEN narrative requirements?
  5. Evidence      — Are specific transactions/amounts cited?
  6. Objectivity   — Is language factual and non-speculative?
  7. Actionability — Would law enforcement find this useful?
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DimensionScore:
    """Score for a single evaluation dimension."""
    dimension: str
    score: float  # 0.0 - 1.0
    weight: float
    details: str = ""


@dataclass
class EvaluationResult:
    """Complete evaluation result for a SAR narrative."""
    case_id: str
    overall_score: float
    dimensions: list[DimensionScore] = field(default_factory=list)
    passed: bool = False
    threshold: float = 0.70

    def summary(self) -> str:
        """Pretty-print the evaluation result."""
        lines = [
            f"=== Evaluation: {self.case_id} ===",
            f"Overall: {self.overall_score:.1%} ({'PASS' if self.passed else 'FAIL'})",
            f"Threshold: {self.threshold:.0%}",
            "",
        ]
        for d in self.dimensions:
            lines.append(f"  [{d.score:.1%}] {d.dimension} (w={d.weight:.0%}): {d.details}")
        return "\n".join(lines)


def evaluate_narrative(
    narrative: str,
    source_data: dict[str, Any],
    risk_indicators: list[dict[str, Any]] | None = None,
    crime_types: list[dict[str, Any]] | None = None,
    threshold: float = 0.70,
) -> EvaluationResult:
    """Evaluate a SAR narrative across multiple quality dimensions.

    This is a rule-based evaluation (no LLM calls). For LLM-based evaluation,
    use the compliance_validation_agent.

    Args:
        narrative: The SAR narrative text to evaluate.
        source_data: The structured case data.
        risk_indicators: Detected risk indicators.
        crime_types: Detected crime types.
        threshold: Minimum overall score to pass.

    Returns:
        EvaluationResult with dimension-level scores.
    """
    case_id = source_data.get("case_id", "unknown")
    risk_indicators = risk_indicators or []
    crime_types = crime_types or []

    dimensions: list[DimensionScore] = []

    # ── 1. Completeness (weight: 0.20) ──
    completeness_score = _score_completeness(narrative, source_data)
    dimensions.append(DimensionScore(
        dimension="completeness",
        score=completeness_score,
        weight=0.20,
        details=f"5W1H coverage score",
    ))

    # ── 2. Accuracy (weight: 0.20) ──
    accuracy_score = _score_accuracy(narrative, source_data)
    dimensions.append(DimensionScore(
        dimension="accuracy",
        score=accuracy_score,
        weight=0.20,
        details="Fact reference coverage",
    ))

    # ── 3. Coherence (weight: 0.10) ──
    coherence_score = _score_coherence(narrative)
    dimensions.append(DimensionScore(
        dimension="coherence",
        score=coherence_score,
        weight=0.10,
        details="Structure and length assessment",
    ))

    # ── 4. Regulatory alignment (weight: 0.15) ──
    regulatory_score = _score_regulatory(narrative)
    dimensions.append(DimensionScore(
        dimension="regulatory_alignment",
        score=regulatory_score,
        weight=0.15,
        details="FinCEN requirements check",
    ))

    # ── 5. Evidence citation (weight: 0.15) ──
    evidence_score = _score_evidence(narrative, source_data)
    dimensions.append(DimensionScore(
        dimension="evidence_citation",
        score=evidence_score,
        weight=0.15,
        details="Transaction/amount/date references",
    ))

    # ── 6. Objectivity (weight: 0.10) ──
    objectivity_score = _score_objectivity(narrative)
    dimensions.append(DimensionScore(
        dimension="objectivity",
        score=objectivity_score,
        weight=0.10,
        details="Language objectivity check",
    ))

    # ── 7. Actionability (weight: 0.10) ──
    actionability_score = _score_actionability(narrative, crime_types)
    dimensions.append(DimensionScore(
        dimension="actionability",
        score=actionability_score,
        weight=0.10,
        details="Law enforcement utility",
    ))

    # ── Weighted overall score ──
    overall = sum(d.score * d.weight for d in dimensions)
    overall = round(overall, 3)

    return EvaluationResult(
        case_id=case_id,
        overall_score=overall,
        dimensions=dimensions,
        passed=overall >= threshold,
        threshold=threshold,
    )


# ── Dimension scorers ──

def _score_completeness(narrative: str, data: dict) -> float:
    """Check if narrative covers the 5W1H elements."""
    score = 0.0
    n_lower = narrative.lower()

    # Who: subject name or reference
    subject_name = data.get("subject", {}).get("name", "")
    if subject_name and (subject_name.lower() in n_lower or "subject" in n_lower):
        score += 0.2

    # What: activity description
    if any(w in n_lower for w in ("transaction", "transfer", "wire", "payment", "activity")):
        score += 0.2

    # When: date references
    date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}|\w+\s+\d{1,2},?\s+\d{4}", re.IGNORECASE)
    if date_pattern.search(narrative):
        score += 0.2

    # Where: location/jurisdiction references
    if any(w in n_lower for w in ("chicago", "account", "branch", "jurisdiction", "country")):
        score += 0.2

    # How: method description
    if any(w in n_lower for w in ("structuring", "layering", "below", "threshold", "pattern")):
        score += 0.2

    return min(score, 1.0)


def _score_accuracy(narrative: str, data: dict) -> float:
    """Check if narrative references actual data from the case."""
    facts_found = 0
    facts_checked = 0

    # Check case ID
    case_id = data.get("case_id", "")
    if case_id:
        facts_checked += 1
        if case_id in narrative:
            facts_found += 1

    # Check transaction IDs
    txns = data.get("transactions", [])
    for txn in txns[:5]:
        txn_id = txn.get("txn_id", "")
        if txn_id:
            facts_checked += 1
            if txn_id in narrative:
                facts_found += 1

    # Check amounts
    for txn in txns[:5]:
        amount = txn.get("amount", 0)
        if amount > 0:
            facts_checked += 1
            # Check various number formats
            formatted = f"{amount:,.2f}"
            if formatted in narrative or str(int(amount)) in narrative:
                facts_found += 1

    return facts_found / max(facts_checked, 1)


def _score_coherence(narrative: str) -> float:
    """Score narrative structural coherence."""
    words = narrative.split()
    word_count = len(words)

    # Length score (200-2000 words is ideal)
    if word_count < 100:
        length_score = word_count / 100
    elif word_count <= 2000:
        length_score = 1.0
    else:
        length_score = max(0.7, 1.0 - (word_count - 2000) / 5000)

    # Paragraph structure
    paragraphs = [p.strip() for p in narrative.split("\n\n") if p.strip()]
    structure_score = min(len(paragraphs) / 3, 1.0)

    return (length_score + structure_score) / 2


def _score_regulatory(narrative: str) -> float:
    """Check FinCEN regulatory alignment."""
    n_lower = narrative.lower()
    score = 0.0

    # Required elements per FinCEN guidelines
    checks = [
        any(w in n_lower for w in ("suspicious", "unusual", "anomalous")),
        any(w in n_lower for w in ("account", "acc-")),
        any(w in n_lower for w in ("transaction", "transfer")),
        any(w in n_lower for w in ("$", "amount", "usd", "dollar")),
        any(w in n_lower for w in ("date", "period", "between", "from", "during")),
    ]

    score = sum(checks) / len(checks)
    return score


def _score_evidence(narrative: str, data: dict) -> float:
    """Check specific evidence citations in the narrative."""
    score = 0.0

    # Date references
    date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
    dates_found = len(date_pattern.findall(narrative))
    score += min(dates_found / 3, 0.33)

    # Amount references
    amount_pattern = re.compile(r"\$[\d,]+\.?\d*")
    amounts_found = len(amount_pattern.findall(narrative))
    score += min(amounts_found / 5, 0.34)

    # Transaction ID references
    txn_ids = [t.get("txn_id", "") for t in data.get("transactions", []) if t.get("txn_id")]
    referenced = sum(1 for tid in txn_ids if tid in narrative)
    score += min(referenced / max(len(txn_ids), 1), 0.33)

    return min(score, 1.0)


def _score_objectivity(narrative: str) -> float:
    """Check that language is objective and non-speculative."""
    n_lower = narrative.lower()
    score = 1.0

    # Penalize speculative language
    speculative_phrases = [
        "we believe", "we think", "it seems", "probably",
        "might be", "could be", "we suspect",
    ]
    for phrase in speculative_phrases:
        if phrase in n_lower:
            score -= 0.15

    # Penalize tipping-off language
    tip_off = [
        "you are being investigated",
        "we are filing a sar",
        "you are suspected",
    ]
    for phrase in tip_off:
        if phrase in n_lower:
            score -= 0.3

    return max(score, 0.0)


def _score_actionability(narrative: str, crime_types: list[dict]) -> float:
    """Score how actionable the narrative is for law enforcement."""
    n_lower = narrative.lower()
    score = 0.0

    # Crime type mentioned
    for ct in crime_types:
        ct_name = ct.get("type", "").replace("_", " ")
        if ct_name and ct_name in n_lower:
            score += 0.25

    # Specific patterns described
    if any(w in n_lower for w in ("pattern", "scheme", "method", "technique")):
        score += 0.25

    # Recommendations or conclusions
    if any(w in n_lower for w in ("recommend", "conclusion", "further investigation", "law enforcement")):
        score += 0.25

    # Severity/urgency
    if any(w in n_lower for w in ("high risk", "critical", "significant", "substantial")):
        score += 0.25

    return min(score, 1.0)
