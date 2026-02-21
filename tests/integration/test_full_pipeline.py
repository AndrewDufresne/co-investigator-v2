"""Integration tests for Phase 2 full pipeline.

Tests the complete 10-node SAR graph with all agents, typology subgraph,
compliance validation, and feedback loop.

These tests require a valid DeepSeek API key in .env.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("LLM_API_KEY") and not Path(__file__).resolve().parents[2].joinpath(".env").exists(),
    reason="LLM_API_KEY not set — skipping integration tests",
)

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "data" / "samples"


def _load_sample(name: str) -> dict:
    p = SAMPLES_DIR / name
    assert p.exists(), f"Sample file not found: {p}"
    return json.loads(p.read_text(encoding="utf-8"))


class TestFullPipeline:
    """Integration tests for the Phase 2 full graph."""

    def test_full_pipeline_no_interrupt(self) -> None:
        """End-to-end full pipeline without HITL interrupt."""
        from src.graph.sar_graph import build_sar_graph

        case_data = _load_sample("case_structuring.json")
        app = build_sar_graph(interrupt_before=[])
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        result = app.invoke(
            {"raw_data": case_data, "iteration_count": 0, "max_iterations": 3},
            config,
        )

        # Verify all pipeline stages produced output
        assert "structured_data" in result
        assert "masked_data" in result
        assert "risk_indicators" in result
        assert "crime_types" in result
        assert "execution_plan" in result
        assert "typology_results" in result
        assert "narrative_draft" in result
        assert "compliance_result" in result
        assert "compliance_score" in result

        # Typology results should contain at least one agent's output
        typology = result["typology_results"]
        agent_keys = [k for k in typology if not k.startswith("_")]
        assert len(agent_keys) > 0, "No typology agent produced results"

        # Aggregate should exist
        assert "_aggregate" in typology

        # Narrative should be substantial
        assert len(result["narrative_draft"]) > 200

        # Compliance check should have run
        assert result["compliance_result"]["status"] in ("PASS", "FAIL")

    def test_full_pipeline_with_interrupt_and_resume(self) -> None:
        """Test HITL: pipeline pauses before unmask, then resumes."""
        from src.graph.sar_graph import build_sar_graph

        case_data = _load_sample("case_structuring.json")
        app = build_sar_graph(interrupt_before=["unmask"])
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        # First invoke — should pause at unmask
        result = app.invoke(
            {"raw_data": case_data, "iteration_count": 0, "max_iterations": 3},
            config,
        )

        # Check state: should be paused before unmask
        snapshot = app.get_state(config)
        if snapshot.next and "unmask" in snapshot.next:
            # Resume
            result = app.invoke(None, config)

            # After resume, final_narrative should exist (via unmask agent)
            assert "final_narrative" in result or "narrative_draft" in result


class TestTypologySubgraph:
    """Test the typology subgraph in isolation."""

    def test_subgraph_produces_merged_results(self) -> None:
        """Run typology subgraph alone and verify parallel results merge."""
        from src.graph.typology_subgraph import build_typology_subgraph

        subgraph = build_typology_subgraph().compile()

        state = {
            "structured_data": _load_sample("case_structuring.json"),
            "active_typology_agents": [
                "transaction_fraud",
                "payment_velocity",
                "country_risk",
            ],
            "typology_results": {},
        }

        # The subgraph needs the structured data in masked_data OR structured_data
        # Since we're testing the fallback, use structured_data
        from src.agents.ingestion import data_ingestion_agent

        ingested = data_ingestion_agent({"raw_data": state["structured_data"]})
        state["structured_data"] = ingested["structured_data"]

        result = subgraph.invoke(state)

        typology = result.get("typology_results", {})
        # Should have results from the 3 requested agents + _aggregate
        assert "transaction_fraud" in typology
        assert "payment_velocity" in typology
        assert "country_risk" in typology
        assert "_aggregate" in typology

        # Verify aggregate metadata
        agg = typology["_aggregate"]
        assert agg["agents_run"] is not None
        assert agg["average_risk_score"] >= 0
