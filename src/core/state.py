"""SARState — LangGraph global state schema.

All agents (graph nodes) read from and write to this shared state.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import add_messages


def _merge_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Reducer that merges two dicts (used for typology_results from parallel agents)."""
    merged = {**left} if left else {}
    if right:
        merged.update(right)
    return merged


class CrimeTypeResult(TypedDict):
    """A single detected crime type with confidence score."""
    type: str
    confidence: float
    evidence: list[str]


class ComplianceCheck(TypedDict):
    """A single compliance check result."""
    dimension: str
    passed: bool
    score: float
    details: str


class ComplianceResult(TypedDict):
    """Full compliance validation result."""
    status: Literal["PASS", "FAIL"]
    overall_score: float
    checks: list[ComplianceCheck]
    improvement_suggestions: list[str]


class ExecutionPlan(TypedDict):
    """Planning Agent output — which agents to activate and narrative strategy."""
    active_typology_agents: list[str]
    requires_external_intel: bool
    narrative_focus: str
    narrative_structure: list[str]


class SARState(TypedDict, total=False):
    """Global state flowing through the SAR generation LangGraph.

    Uses `total=False` so agents only need to return the fields they update.
    """

    # ── Input data ──
    case_id: str
    raw_data: dict[str, Any]
    structured_data: dict[str, Any]
    masked_data: dict[str, Any]
    mask_mapping: dict[str, str]

    # ── Crime type detection ──
    risk_indicators: list[dict[str, Any]]
    crime_types: list[CrimeTypeResult]

    # ── Planning ──
    execution_plan: ExecutionPlan
    active_typology_agents: list[str]

    # ── Typology results ──
    typology_results: Annotated[dict[str, Any], _merge_dicts]

    # ── External intelligence ──
    external_intel: list[dict[str, Any]]

    # ── Narrative generation ──
    narrative_draft: str
    narrative_intro: str
    chain_of_thought: list[str]

    # ── Compliance validation ──
    compliance_result: ComplianceResult
    compliance_score: float

    # ── Human-in-the-loop ──
    human_feedback: str | None
    iteration_count: int
    max_iterations: int

    # ── Final output ──
    final_narrative: str
    status: Literal["processing", "review", "approved", "rejected"]

    # ── Message log ──
    messages: Annotated[list, add_messages]
