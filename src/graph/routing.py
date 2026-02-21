"""Routing functions for LangGraph conditional edges."""

from __future__ import annotations

from typing import Literal

from src.config import get_settings
from src.core.state import SARState


def compliance_router(
    state: SARState,
) -> Literal["unmask", "feedback"]:
    """Route based on compliance validation result.

    Returns:
        "unmask"   — narrative passed compliance → proceed to unmask + finalize
        "feedback" — narrative failed → loop back through feedback + regeneration
    """
    score = state.get("compliance_score", 0.0)
    iteration = state.get("iteration_count", 0)
    settings = get_settings()

    # Pass if score meets threshold OR max iterations exhausted
    if score >= settings.compliance_score_threshold:
        return "unmask"
    if iteration >= settings.max_iterations:
        return "unmask"  # force output after max iterations
    return "feedback"


def external_intel_router(
    state: SARState,
) -> Literal["external_intel", "narrative"]:
    """Decide whether to gather external intel or skip to narrative."""
    plan = state.get("execution_plan", {})
    if plan.get("requires_external_intel", False):
        return "external_intel"
    return "narrative"
