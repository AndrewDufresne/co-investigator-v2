"""Typology Subgraph — dynamic parallel execution of typology agents via Send API."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.constants import Send
from langgraph.graph import END, StateGraph

from src.core.state import SARState

# Import all typology agents
from src.agents.typology.transaction_fraud import transaction_fraud_agent
from src.agents.typology.payment_velocity import payment_velocity_agent
from src.agents.typology.country_risk import country_risk_agent
from src.agents.typology.text_content import text_content_agent
from src.agents.typology.geo_anomaly import geo_anomaly_agent
from src.agents.typology.account_health import account_health_agent
from src.agents.typology.dispute_pattern import dispute_pattern_agent

logger = logging.getLogger(__name__)

# Registry mapping agent names to functions
TYPOLOGY_REGISTRY: dict[str, Any] = {
    "transaction_fraud": transaction_fraud_agent,
    "payment_velocity": payment_velocity_agent,
    "country_risk": country_risk_agent,
    "text_content": text_content_agent,
    "geo_anomaly": geo_anomaly_agent,
    "account_health": account_health_agent,
    "dispute_pattern": dispute_pattern_agent,
}


def typology_dispatcher(state: SARState) -> list[Send]:
    """Fan-out: dispatch Send messages to activate selected typology agents.

    Uses the active_typology_agents list from the planning agent to
    dynamically decide which typology agents to run in parallel.
    """
    active = state.get("active_typology_agents", [])
    logger.info("Typology dispatcher: activating agents %s", active)

    sends = []
    for agent_name in active:
        if agent_name in TYPOLOGY_REGISTRY:
            sends.append(Send(agent_name, state))
        else:
            logger.warning("Unknown typology agent: %s (skipped)", agent_name)

    if not sends:
        logger.warning("No valid typology agents selected, using defaults")
        sends = [
            Send("transaction_fraud", state),
            Send("country_risk", state),
            Send("account_health", state),
        ]

    return sends


def typology_aggregator(state: SARState) -> dict[str, Any]:
    """Fan-in: merge all typology results into a consolidated summary.

    This node runs after all parallel typology agents complete.
    The _merge_dicts reducer on typology_results has already merged
    each agent's output; we only add the _aggregate summary here.
    """
    results = state.get("typology_results", {})
    agent_results = {
        k: v for k, v in results.items()
        if not k.startswith("_") and isinstance(v, dict)
    }
    logger.info("Typology aggregator: merging results from %d agents", len(agent_results))

    # Calculate aggregate risk score
    scores = [v.get("risk_score", 0) for v in agent_results.values()]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "typology_results": {
            "_aggregate": {
                "agents_run": list(agent_results.keys()),
                "average_risk_score": round(avg_score, 3),
                "total_findings": sum(
                    len(v.get("findings", []))
                    for v in agent_results.values()
                    if "findings" in v
                ),
            },
        }
    }


def build_typology_subgraph() -> StateGraph:
    """Build the typology analysis subgraph with dynamic parallel execution.

    Graph structure:
        dispatcher → [agent_1, agent_2, ...] (parallel via Send) → aggregator → END
    """
    graph = StateGraph(SARState)

    # Register all typology agent nodes
    for name, func in TYPOLOGY_REGISTRY.items():
        graph.add_node(name, func)

    # Aggregator node
    graph.add_node("aggregator", typology_aggregator)

    # Entry: dispatcher fans out via Send API
    graph.set_conditional_entry_point(typology_dispatcher)

    # Each typology agent flows to aggregator
    for name in TYPOLOGY_REGISTRY:
        graph.add_edge(name, "aggregator")

    graph.add_edge("aggregator", END)

    return graph
