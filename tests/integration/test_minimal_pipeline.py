"""Integration tests for Phase 1 minimal pipeline.

These tests require a valid DeepSeek API key in .env to run the LLM-based agents.
Mark with pytest.mark.integration so they can be skipped in fast local runs.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest

# Skip if no API key available
pytestmark = pytest.mark.skipif(
    not os.getenv("LLM_API_KEY") and not Path(__file__).resolve().parents[2].joinpath(".env").exists(),
    reason="LLM_API_KEY not set — skipping integration tests",
)

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "data" / "samples"


def _load_sample(name: str) -> dict:
    p = SAMPLES_DIR / name
    assert p.exists(), f"Sample file not found: {p}"
    return json.loads(p.read_text(encoding="utf-8"))


class TestMinimalPipeline:
    """Integration tests for the Phase 1 minimal graph: ingest → crime_detect → narrative."""

    def test_minimal_pipeline_produces_narrative(self) -> None:
        """End-to-end: raw JSON → SAR narrative via minimal 3-node graph."""
        from src.graph.minimal_graph import build_minimal_graph

        case_data = _load_sample("case_structuring.json")
        app = build_minimal_graph()
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        result = app.invoke(
            {"raw_data": case_data, "iteration_count": 0, "max_iterations": 3},
            config,
        )

        # Verify structured_data was produced by ingestion
        assert "structured_data" in result
        assert result["structured_data"]["case_id"] == "CASE-2024-STR-001"

        # Verify crime detection ran
        assert "risk_indicators" in result
        assert len(result["risk_indicators"]) > 0
        assert "crime_types" in result
        assert len(result["crime_types"]) > 0

        # Verify narrative was generated
        assert "narrative_draft" in result
        narrative = result["narrative_draft"]
        assert len(narrative) > 100, f"Narrative too short ({len(narrative)} chars)"

        # Verify chain of thought was captured
        assert "chain_of_thought" in result

    def test_minimal_pipeline_elder_exploit(self) -> None:
        """Test with elder exploitation case."""
        from src.graph.minimal_graph import build_minimal_graph

        case_data = _load_sample("case_elder_exploit.json")
        app = build_minimal_graph()
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        result = app.invoke(
            {"raw_data": case_data, "iteration_count": 0, "max_iterations": 3},
            config,
        )

        assert "narrative_draft" in result
        assert len(result["narrative_draft"]) > 100

    def test_minimal_pipeline_shell_company(self) -> None:
        """Test with shell company case."""
        from src.graph.minimal_graph import build_minimal_graph

        case_data = _load_sample("case_shell_company.json")
        app = build_minimal_graph()
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        result = app.invoke(
            {"raw_data": case_data, "iteration_count": 0, "max_iterations": 3},
            config,
        )

        assert "narrative_draft" in result
        assert len(result["narrative_draft"]) > 100


class TestMinimalPipelineNoLLM:
    """Tests that run the non-LLM nodes only (ingestion + crime detection)."""

    def test_ingestion_and_crime_detection(self) -> None:
        """Test ingest + crime_detect without the narrative (LLM) step."""
        from src.agents.ingestion import data_ingestion_agent
        from src.agents.crime_detection import crime_detection_agent

        case_data = _load_sample("case_structuring.json")

        # Step 1: Ingestion
        ingest_result = data_ingestion_agent({"raw_data": case_data})
        structured = ingest_result["structured_data"]
        assert structured["case_id"] == "CASE-2024-STR-001"

        # Step 2: Crime detection (using structured_data, no masked_data)
        detect_result = crime_detection_agent({"structured_data": structured})
        assert len(detect_result["risk_indicators"]) > 0
        assert len(detect_result["crime_types"]) > 0

        # Verify structuring detected
        crime_type_names = [ct["type"] for ct in detect_result["crime_types"]]
        assert "structuring" in crime_type_names
