"""SAR Main Graph — orchestrates the full SAR generation pipeline via LangGraph.

Graph flow:
  ingest → mask → crime_detect → plan → typology_subgraph → [external_intel] →
  narrative → compliance → (pass → unmask → END | fail → feedback → narrative loop)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.config import get_settings
from src.core.state import SARState

# Agent imports
from src.agents.ingestion import data_ingestion_agent
from src.agents.privacy_guard import privacy_mask_agent, privacy_unmask_agent
from src.agents.crime_detection import crime_detection_agent
from src.agents.planning import planning_agent
from src.agents.narrative import narrative_generation_agent
from src.agents.compliance import compliance_validation_agent
from src.agents.feedback import feedback_agent
from src.agents.external_intel import external_intel_agent

# Graph components
from src.graph.routing import compliance_router, external_intel_router
from src.graph.typology_subgraph import build_typology_subgraph

logger = logging.getLogger(__name__)


def build_sar_graph(
    checkpointer: Any | None = None,
    interrupt_before: list[str] | None = None,
) -> Any:
    """Build and compile the main SAR generation StateGraph.

    Args:
        checkpointer: LangGraph checkpointer for state persistence.
                      Defaults to MemorySaver if None.
        interrupt_before: List of node names to pause before (HITL breakpoints).
                         Defaults to ["unmask"] for human review.

    Returns:
        Compiled LangGraph application.
    """
    settings = get_settings()

    graph = StateGraph(SARState)

    # ── Register nodes ──
    graph.add_node("ingest", data_ingestion_agent)
    graph.add_node("mask", privacy_mask_agent)
    graph.add_node("crime_detect", crime_detection_agent)
    graph.add_node("plan", planning_agent)

    # Typology subgraph (compiled inline)
    typology_subgraph = build_typology_subgraph()
    graph.add_node("typology", typology_subgraph.compile())

    graph.add_node("external_intel", external_intel_agent)
    graph.add_node("narrative", narrative_generation_agent)
    graph.add_node("compliance", compliance_validation_agent)
    graph.add_node("feedback", feedback_agent)
    graph.add_node("unmask", privacy_unmask_agent)

    # ── Define edges ──
    # Linear flow: ingest → mask → crime_detect → plan → typology
    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "mask")
    graph.add_edge("mask", "crime_detect")
    graph.add_edge("crime_detect", "plan")
    graph.add_edge("plan", "typology")

    # After typology: conditionally gather external intel or go straight to narrative
    graph.add_conditional_edges(
        "typology",
        external_intel_router,
        {
            "external_intel": "external_intel",
            "narrative": "narrative",
        },
    )
    graph.add_edge("external_intel", "narrative")

    # After narrative: compliance check
    graph.add_edge("narrative", "compliance")

    # Compliance routing: pass → unmask, fail → feedback loop
    graph.add_conditional_edges(
        "compliance",
        compliance_router,
        {
            "unmask": "unmask",
            "feedback": "feedback",
        },
    )

    # Feedback loops back to narrative for revision
    graph.add_edge("feedback", "narrative")

    # Unmask → END
    graph.add_edge("unmask", END)

    # ── Compile ──
    if checkpointer is None:
        checkpointer = MemorySaver()

    if interrupt_before is None:
        interrupt_before = ["unmask"]

    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
    )

    logger.info(
        "SAR graph compiled: %d nodes, interrupt_before=%s",
        len(graph.nodes),
        interrupt_before,
    )

    return compiled
