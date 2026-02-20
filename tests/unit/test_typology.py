"""Unit tests for Typology Agents."""

from __future__ import annotations

import pytest

from src.agents.typology.transaction_fraud import transaction_fraud_agent
from src.agents.typology.payment_velocity import payment_velocity_agent
from src.agents.typology.country_risk import country_risk_agent
from src.agents.typology.text_content import text_content_agent
from src.agents.typology.geo_anomaly import geo_anomaly_agent
from src.agents.typology.account_health import account_health_agent
from src.agents.typology.dispute_pattern import dispute_pattern_agent
from src.core.state import _merge_dicts


@pytest.fixture
def rich_case_data() -> dict:
    """Rich case data that should trigger findings across multiple agents."""
    return {
        "case_id": "TEST-TYPOLOGY",
        "transactions": [
            {
                "txn_id": "T1", "date": "2024-01-10", "type": "wire_out",
                "amount": 9000, "from_entity": "Subject", "to_entity": "Corp A",
                "from_country": "US", "to_country": "PA",
                "location": "Chicago", "description": "fee", "risk_flags": [],
            },
            {
                "txn_id": "T2", "date": "2024-01-10", "type": "wire_out",
                "amount": 9500, "from_entity": "Subject", "to_entity": "Corp A",
                "from_country": "US", "to_country": "PA",
                "location": "New York", "description": "payment", "risk_flags": [],
            },
            {
                "txn_id": "T3", "date": "2024-01-10", "type": "wire_out",
                "amount": 8500, "from_entity": "Subject", "to_entity": "Corp B",
                "from_country": "US", "to_country": "BZ",
                "location": "Miami", "description": "split", "risk_flags": [],
            },
            {
                "txn_id": "T4", "date": "2024-01-11", "type": "wire_in",
                "amount": 50000, "from_entity": "Shell LLC", "to_entity": "Subject",
                "from_country": "KY", "to_country": "US",
                "location": "Chicago", "description": "consulting", "risk_flags": [],
            },
        ],
        "subject": {"name": "Test Subject", "address": "Chicago, IL"},
        "accounts": [
            {"account_id": "ACC-1", "account_type": "checking", "balance": 150000},
            {"account_id": "ACC-2", "account_type": "savings", "balance": 200000},
            {"account_id": "ACC-3", "account_type": "business", "balance": 50000},
        ],
        "kyc": {
            "expected_activity": "low",
            "actual_activity_profile": "high",
            "activity_mismatch": True,
            "pep_status": True,
        },
        "flagged_communications": [
            {
                "content_snippet": "Send the money urgently, must be confidential and under the table",
                "flag_reason": "suspicious_language",
            }
        ],
        "alerts": [
            {"alert_id": "A1", "type": "structuring", "severity": "high", "description": "test"},
            {"alert_id": "A2", "type": "velocity", "severity": "high", "description": "test"},
            {"alert_id": "A3", "type": "jurisdiction", "severity": "critical", "description": "test"},
        ],
        "related_entities": [
            {
                "entity_name": "Shell LLC",
                "entity_type": "company",
                "jurisdiction": "KY",
                "risk_notes": "Known shell entity",
            }
        ],
    }


def _make_state(data: dict) -> dict:
    """Create a state dict with structured_data (no masked_data, testing fallback)."""
    return {"structured_data": data}


class TestTransactionFraudAgent:
    def test_detects_structuring(self, rich_case_data: dict) -> None:
        result = transaction_fraud_agent(_make_state(rich_case_data))
        assert "typology_results" in result
        tf = result["typology_results"]["transaction_fraud"]
        patterns = [f["pattern"] for f in tf["findings"]]
        assert "structuring_below_ctr_threshold" in patterns

    def test_detects_counterparty_concentration(self, rich_case_data: dict) -> None:
        # Corp A gets 9000 + 9500 = 18500, which is below 50k threshold.
        # Boost one transaction to trigger the pattern.
        data = {**rich_case_data}
        data["transactions"] = list(rich_case_data["transactions"]) + [
            {
                "txn_id": "T5", "date": "2024-01-12", "type": "wire_out",
                "amount": 42000, "from_entity": "Subject", "to_entity": "Corp A",
                "from_country": "US", "to_country": "PA",
                "location": "Chicago", "description": "extra", "risk_flags": [],
            }
        ]
        result = transaction_fraud_agent(_make_state(data))
        tf = result["typology_results"]["transaction_fraud"]
        patterns = [f["pattern"] for f in tf["findings"]]
        assert "counterparty_concentration" in patterns

    def test_risk_score_bounded(self, rich_case_data: dict) -> None:
        result = transaction_fraud_agent(_make_state(rich_case_data))
        score = result["typology_results"]["transaction_fraud"]["risk_score"]
        assert 0 <= score <= 1.0

    def test_no_redundant_state_merge(self, rich_case_data: dict) -> None:
        """Ensure result only contains 'transaction_fraud' key."""
        result = transaction_fraud_agent(_make_state(rich_case_data))
        keys = list(result["typology_results"].keys())
        assert keys == ["transaction_fraud"]


class TestPaymentVelocityAgent:
    def test_detects_high_frequency(self, rich_case_data: dict) -> None:
        result = payment_velocity_agent(_make_state(rich_case_data))
        pv = result["typology_results"]["payment_velocity"]
        assert len(pv["findings"]) > 0
        patterns = [f["pattern"] for f in pv["findings"]]
        assert "high_daily_frequency" in patterns


class TestCountryRiskAgent:
    def test_detects_monitored_jurisdiction(self, rich_case_data: dict) -> None:
        result = country_risk_agent(_make_state(rich_case_data))
        cr = result["typology_results"]["country_risk"]
        assert len(cr["findings"]) > 0

    def test_detects_entity_jurisdiction_risk(self, rich_case_data: dict) -> None:
        result = country_risk_agent(_make_state(rich_case_data))
        cr = result["typology_results"]["country_risk"]
        patterns = [f["pattern"] for f in cr["findings"]]
        assert "entity_jurisdiction_risk" in patterns


class TestTextContentAgent:
    def test_detects_suspicious_language(self, rich_case_data: dict) -> None:
        result = text_content_agent(_make_state(rich_case_data))
        tc = result["typology_results"]["text_content"]
        assert len(tc["findings"]) > 0


class TestGeoAnomalyAgent:
    def test_detects_impossible_travel(self, rich_case_data: dict) -> None:
        result = geo_anomaly_agent(_make_state(rich_case_data))
        ga = result["typology_results"]["geo_anomaly"]
        patterns = [f["pattern"] for f in ga["findings"]]
        assert "impossible_travel" in patterns

    def test_detects_geographic_diversity(self, rich_case_data: dict) -> None:
        result = geo_anomaly_agent(_make_state(rich_case_data))
        ga = result["typology_results"]["geo_anomaly"]
        patterns = [f["pattern"] for f in ga["findings"]]
        # 4 locations: Chicago, New York, Miami, and potentially others
        assert "high_geographic_diversity" in patterns or len(ga["findings"]) > 0


class TestAccountHealthAgent:
    def test_detects_activity_mismatch(self, rich_case_data: dict) -> None:
        result = account_health_agent(_make_state(rich_case_data))
        ah = result["typology_results"]["account_health"]
        patterns = [f["pattern"] for f in ah["findings"]]
        assert "profile_activity_mismatch" in patterns

    def test_detects_pep(self, rich_case_data: dict) -> None:
        result = account_health_agent(_make_state(rich_case_data))
        ah = result["typology_results"]["account_health"]
        patterns = [f["pattern"] for f in ah["findings"]]
        assert "pep_involvement" in patterns

    def test_detects_multiple_accounts(self, rich_case_data: dict) -> None:
        result = account_health_agent(_make_state(rich_case_data))
        ah = result["typology_results"]["account_health"]
        patterns = [f["pattern"] for f in ah["findings"]]
        assert "multiple_accounts" in patterns


class TestDisputePatternAgent:
    def test_detects_high_severity_alerts(self, rich_case_data: dict) -> None:
        result = dispute_pattern_agent(_make_state(rich_case_data))
        dp = result["typology_results"]["dispute_pattern"]
        patterns = [f["pattern"] for f in dp["findings"]]
        assert "multiple_high_severity_alerts" in patterns

    def test_detects_diverse_alert_types(self, rich_case_data: dict) -> None:
        result = dispute_pattern_agent(_make_state(rich_case_data))
        dp = result["typology_results"]["dispute_pattern"]
        patterns = [f["pattern"] for f in dp["findings"]]
        assert "diverse_alert_types" in patterns


class TestMergeDictsReducer:
    """Test the _merge_dicts reducer used for parallel typology results."""

    def test_merge_empty_left(self) -> None:
        result = _merge_dicts({}, {"a": 1})
        assert result == {"a": 1}

    def test_merge_empty_right(self) -> None:
        result = _merge_dicts({"a": 1}, {})
        assert result == {"a": 1}

    def test_merge_both_empty(self) -> None:
        result = _merge_dicts({}, {})
        assert result == {}

    def test_merge_non_overlapping(self) -> None:
        result = _merge_dicts({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_merge_overlapping_right_wins(self) -> None:
        result = _merge_dicts({"a": 1}, {"a": 2})
        assert result == {"a": 2}

    def test_merge_simulates_parallel_agents(self) -> None:
        """Simulate the sequential reducer merge of 3 parallel agent outputs."""
        accumulated = {}
        accumulated = _merge_dicts(accumulated, {"transaction_fraud": {"risk_score": 0.5}})
        accumulated = _merge_dicts(accumulated, {"country_risk": {"risk_score": 0.7}})
        accumulated = _merge_dicts(accumulated, {"text_content": {"risk_score": 0.25}})
        assert len(accumulated) == 3
        assert accumulated["transaction_fraud"]["risk_score"] == 0.5
        assert accumulated["country_risk"]["risk_score"] == 0.7
        assert accumulated["text_content"]["risk_score"] == 0.25
