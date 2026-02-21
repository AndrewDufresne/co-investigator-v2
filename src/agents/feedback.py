"""Feedback Agent — integrates compliance and human feedback for narrative refinement."""

from __future__ import annotations

import logging
from typing import Any

from src.core.state import SARState

logger = logging.getLogger(__name__)


def feedback_agent(state: SARState) -> dict[str, Any]:
    """Consolidate compliance validation feedback and human input.

    Increments iteration_count and prepares revision instructions
    for the next narrative generation cycle.
    """
    compliance = state.get("compliance_result", {})
    human_feedback = state.get("human_feedback")
    iteration = state.get("iteration_count", 0)

    logger.info("Feedback agent: preparing revision instructions (iteration %d)", iteration + 1)

    # Build combined feedback
    feedback_parts: list[str] = []

    # From compliance validation
    suggestions = compliance.get("improvement_suggestions", [])
    if suggestions:
        feedback_parts.append("Compliance Validator Feedback:")
        for i, s in enumerate(suggestions, 1):
            feedback_parts.append(f"  {i}. {s}")

    # Failed checks
    for check in compliance.get("checks", []):
        if not check.get("passed", True):
            feedback_parts.append(f"  - Fix: {check.get('dimension')}: {check.get('details')}")

    # From human investigator
    if human_feedback:
        feedback_parts.append(f"\nInvestigator Feedback:\n{human_feedback}")

    combined = "\n".join(feedback_parts) if feedback_parts else "No specific feedback provided."

    logger.info("Feedback prepared: %d chars of revision instructions", len(combined))

    return {
        "human_feedback": combined,
        "iteration_count": iteration + 1,
    }
