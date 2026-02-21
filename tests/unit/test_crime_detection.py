"""Unit tests for Crime Detection Agent."""

from __future__ import annotations

import pytest
from src.agents.crime_detection import (
    _extract_risk_indicators,
    _classify_crime_types,
    crime_detection_agent,
)


@pytest.fixture
def structuring_data() -> dict:
    """Case data with structuring patterns."""
    return {
        "case_id": "TEST-STR",
        "transactions": [
            {"txn_id": "T1", "amount": 9500, "type": "wire_out", "risk_flags": ["structured_amount"]},
            {"txn_id": "T2", "amount": 9800, "type": "wire_out", "risk_flags": ["structured_amount"]},
            {"txn_id": "T3", "amount": 9200, "type": "wire_out", "risk_flags": []},
        ],
        "kyc": {
            "expected_activity": "low",
            "actual_activity_profile": "high",
            "activity_mismatch": True,
            "adverse_media_hits": [],
        },
        "flagged_communications": [],
        "related_entities": [],
    }


@pytest.fixture
def layering_data() -> dict:
    """Case data with layering + shell company patterns."""
    return {
        "case_id": "TEST-LAY",
        "transactions": [
            {
                "txn_id": "T1", "amount": 50000, "type": "wire_out",
                "risk_flags": ["layering_pattern", "rapid_movement"],
                "from_country": "US", "to_country": "PA",
            },
            {
                "txn_id": "T2", "amount": 48000, "type": "wire_in",
                "risk_flags": ["shell_company_indicator"],
                "from_country": "BZ", "to_country": "US",
            },
        ],
        "kyc": {"adverse_media_hits": [], "activity_mismatch": False},
        "flagged_communications": [],
        "related_entities": [],
    }


@pytest.fixture
def clean_data() -> dict:
    """Clean case data with no risk indicators."""
    return {
        "case_id": "TEST-CLEAN",
        "transactions": [
            {"txn_id": "T1", "amount": 500, "type": "deposit", "risk_flags": []},
            {"txn_id": "T2", "amount": 200, "type": "withdrawal", "risk_flags": []},
        ],
        "kyc": {
            "expected_activity": "low",
            "actual_activity_profile": "low",
            "activity_mismatch": False,
            "adverse_media_hits": [],
        },
        "flagged_communications": [],
        "related_entities": [],
    }


class TestRiskIndicatorExtraction:
    """Tests for _extract_risk_indicators."""

    def test_detects_structuring(self, structuring_data: dict) -> None:
        indicators = _extract_risk_indicators(structuring_data)
        types = [i["type"] for i in indicators]
        assert "structuring" in types

    def test_detects_layering(self, layering_data: dict) -> None:
        indicators = _extract_risk_indicators(layering_data)
        types = [i["type"] for i in indicators]
        assert "layering" in types

    def test_detects_shell_company(self, layering_data: dict) -> None:
        indicators = _extract_risk_indicators(layering_data)
        types = [i["type"] for i in indicators]
        assert "shell_company" in types

    def test_detects_high_risk_jurisdiction(self, layering_data: dict) -> None:
        indicators = _extract_risk_indicators(layering_data)
        types = [i["type"] for i in indicators]
        assert "high_risk_jurisdiction" in types

    def test_detects_kyc_mismatch(self, structuring_data: dict) -> None:
        indicators = _extract_risk_indicators(structuring_data)
        types = [i["type"] for i in indicators]
        assert "kyc_mismatch" in types

    def test_clean_data_minimal_indicators(self, clean_data: dict) -> None:
        indicators = _extract_risk_indicators(clean_data)
        assert len(indicators) == 0

    def test_indicator_severity(self, structuring_data: dict) -> None:
        indicators = _extract_risk_indicators(structuring_data)
        structuring = [i for i in indicators if i["type"] == "structuring"][0]
        assert structuring["severity"] == "high"

    def test_indicator_has_evidence(self, structuring_data: dict) -> None:
        indicators = _extract_risk_indicators(structuring_data)
        for ind in indicators:
            assert "evidence" in ind
            assert isinstance(ind["evidence"], list)


class TestCrimeTypeClassification:
    """Tests for _classify_crime_types."""

    def test_classifies_structuring(self) -> None:
        indicators = [
            {"type": "structuring", "severity": "high", "description": "", "evidence": []},
            {"type": "kyc_mismatch", "severity": "medium", "description": "", "evidence": []},
        ]
        results = _classify_crime_types(indicators)
        types = [r["type"] for r in results]
        assert "structuring" in types

    def test_classifies_money_laundering(self) -> None:
        indicators = [
            {"type": "layering", "severity": "high", "description": "", "evidence": []},
            {"type": "shell_company", "severity": "medium", "description": "", "evidence": []},
            {"type": "high_risk_jurisdiction", "severity": "high", "description": "", "evidence": []},
        ]
        results = _classify_crime_types(indicators)
        types = [r["type"] for r in results]
        assert "money_laundering_layering" in types

    def test_confidence_bounded(self) -> None:
        indicators = [
            {"type": "layering", "severity": "high", "description": "", "evidence": []},
            {"type": "shell_company", "severity": "medium", "description": "", "evidence": []},
            {"type": "high_risk_jurisdiction", "severity": "high", "description": "", "evidence": []},
            {"type": "structuring", "severity": "high", "description": "", "evidence": []},
        ]
        results = _classify_crime_types(indicators)
        for r in results:
            assert 0 < r["confidence"] <= 0.99

    def test_no_classification_for_clean(self) -> None:
        results = _classify_crime_types([])
        assert len(results) == 0

    def test_results_sorted_by_confidence(self) -> None:
        indicators = [
            {"type": "structuring", "severity": "high", "description": "", "evidence": []},
            {"type": "layering", "severity": "high", "description": "", "evidence": []},
            {"type": "shell_company", "severity": "medium", "description": "", "evidence": []},
        ]
        results = _classify_crime_types(indicators)
        confidences = [r["confidence"] for r in results]
        assert confidences == sorted(confidences, reverse=True)


class TestCrimeDetectionAgent:
    """Tests for the full crime_detection_agent function."""

    def test_uses_structured_data_fallback(self, structuring_data: dict) -> None:
        """Agent uses structured_data when masked_data is absent."""
        state = {"structured_data": structuring_data}
        result = crime_detection_agent(state)
        assert "risk_indicators" in result
        assert "crime_types" in result

    def test_prefers_masked_data(self, structuring_data: dict) -> None:
        """Agent uses masked_data when available."""
        state = {"masked_data": structuring_data, "structured_data": {}}
        result = crime_detection_agent(state)
        assert len(result["risk_indicators"]) > 0

    def test_output_structure(self, structuring_data: dict) -> None:
        state = {"structured_data": structuring_data}
        result = crime_detection_agent(state)
        for ind in result["risk_indicators"]:
            assert "type" in ind
            assert "severity" in ind
            assert "description" in ind
            assert "evidence" in ind
        for ct in result["crime_types"]:
            assert "type" in ct
            assert "confidence" in ct
            assert "evidence" in ct
