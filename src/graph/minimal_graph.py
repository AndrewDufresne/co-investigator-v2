"""Phase 1 Minimal Graph — 3-node linear pipeline for end-to-end SAR generation.

Graph flow: ingest → crime_detect → narrative → END

This graph bypasses privacy masking, planning, typology agents, external intel,
compliance validation, and feedback loops. It is the simplest end-to-end pipeline
that can produce a SAR narrative draft from raw case data via a single LLM call.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.core.state import SARState

# Agent imports
from src.agents.ingestion import data_ingestion_agent
from src.agents.crime_detection import crime_detection_agent
from src.agents.narrative import narrative_generation_agent

logger = logging.getLogger(__name__)


def build_minimal_graph(checkpointer: Any | None = None) -> Any:
    """Build and compile the Phase 1 minimal 3-node SAR graph.

    Flow: ingest → crime_detect → narrative → END

    No privacy masking, no planning, no typology subgraph, no compliance loop.
    The narrative agent falls back to structured_data when masked_data is absent.

    Returns:
        Compiled LangGraph application.
    """
    graph = StateGraph(SARState)

    # ── Register nodes ──
    graph.add_node("ingest", data_ingestion_agent)
    graph.add_node("crime_detect", crime_detection_agent)
    graph.add_node("narrative", narrative_generation_agent)

    # ── Linear edges ──
    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "crime_detect")
    graph.add_edge("crime_detect", "narrative")
    graph.add_edge("narrative", END)

    # ── Compile ──
    if checkpointer is None:
        checkpointer = MemorySaver()

    compiled = graph.compile(checkpointer=checkpointer)

    logger.info("Minimal graph compiled: 3 nodes (ingest → crime_detect → narrative → END)")
    return compiled
