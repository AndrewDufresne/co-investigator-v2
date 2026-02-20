"""Unit tests for Data Ingestion Agent."""

from __future__ import annotations

import pytest
from src.agents.ingestion import data_ingestion_agent


@pytest.fixture
def sample_raw_data() -> dict:
    """Minimal raw case data for testing."""
    return {
        "case_id": "TEST-001",
        "alert_date": "2024-01-15",
        "priority": "high",
        "subject": {
            "name": "Test Subject",
            "dob": "1990-01-01",
            "ssn": "123-45-6789",
            "address": "123 Main St, Anytown, US 12345",
            "phone": "+1-555-0100",
            "email": "test@example.com",
            "occupation": "Consultant",
            "risk_rating": "medium",
            "customer_since": "2020-01-01",
        },
        "accounts": [
            {
                "account_id": "ACC-001",
                "account_type": "checking",
                "balance": 50000.00,
                "currency": "USD",
                "branch": "Main Branch",
            }
        ],
        "transactions": [
            {
                "txn_id": "TXN-001",
                "date": "2024-01-10",
                "type": "wire_out",
                "amount": 9500.00,
                "currency": "USD",
                "from_account": "ACC-001",
                "to_account": "EXT-001",
                "from_entity": "Test Subject",
                "to_entity": "Offshore Corp",
                "from_country": "US",
                "to_country": "PA",
                "location": "Chicago",
                "description": "Investment transfer",
                "risk_flags": ["structured_amount"],
            },
            {
                "txn_id": "TXN-002",
                "date": "2024-01-11",
                "type": "wire_out",
                "amount": 9800.00,
                "currency": "USD",
                "from_account": "ACC-001",
                "to_account": "EXT-002",
                "from_entity": "Test Subject",
                "to_entity": "Shell LLC",
                "from_country": "US",
                "to_country": "BZ",
                "location": "Chicago",
                "description": "Consulting fee",
                "risk_flags": ["structured_amount", "layering_pattern"],
            },
        ],
        "kyc": {
            "verification_status": "verified",
            "source_of_funds": "salary",
            "expected_activity": "low",
            "actual_activity_profile": "high",
            "pep_status": False,
            "adverse_media_hits": [],
        },
        "communications": [
            {
                "date": "2024-01-09",
                "channel": "email",
                "content": "Send the wire immediately, must be under 10k",
                "flagged": True,
                "flag_reason": "structuring_language",
            }
        ],
        "alerts": [
            {
                "alert_id": "ALT-001",
                "type": "structuring",
                "severity": "high",
                "description": "Multiple transactions just below $10,000",
                "triggered_date": "2024-01-12",
            }
        ],
        "related_entities": [
            {
                "entity_name": "Offshore Corp",
                "entity_type": "company",
                "jurisdiction": "PA",
                "relationship": "counterparty",
                "risk_notes": "Shell company in Panama",
            }
        ],
    }


class TestDataIngestionAgent:
    """Tests for data_ingestion_agent."""

    def test_returns_structured_data(self, sample_raw_data: dict) -> None:
        state = {"raw_data": sample_raw_data}
        result = data_ingestion_agent(state)
        assert "structured_data" in result
        sd = result["structured_data"]
        assert sd["case_id"] == "TEST-001"
        assert sd["alert_date"] == "2024-01-15"
        assert sd["priority"] == "high"

    def test_subject_extraction(self, sample_raw_data: dict) -> None:
        result = data_ingestion_agent({"raw_data": sample_raw_data})
        subject = result["structured_data"]["subject"]
        assert subject["name"] == "Test Subject"
        assert subject["ssn"] == "123-45-6789"
        assert subject["risk_rating"] == "medium"

    def test_transaction_extraction(self, sample_raw_data: dict) -> None:
        result = data_ingestion_agent({"raw_data": sample_raw_data})
        txns = result["structured_data"]["transactions"]
        assert len(txns) == 2
        assert txns[0]["txn_id"] == "TXN-001"
        assert txns[0]["amount"] == 9500.00
        assert txns[1]["risk_flags"] == ["structured_amount", "layering_pattern"]

    def test_transaction_summary(self, sample_raw_data: dict) -> None:
        result = data_ingestion_agent({"raw_data": sample_raw_data})
        summary = result["structured_data"]["transaction_summary"]
        assert summary["count"] == 2
        assert summary["total_outflow"] == 19300.00

    def test_kyc_summary_activity_mismatch(self, sample_raw_data: dict) -> None:
        result = data_ingestion_agent({"raw_data": sample_raw_data})
        kyc = result["structured_data"]["kyc"]
        assert kyc["activity_mismatch"] is True
        assert kyc["expected_activity"] == "low"
        assert kyc["actual_activity_profile"] == "high"

    def test_flagged_communications(self, sample_raw_data: dict) -> None:
        result = data_ingestion_agent({"raw_data": sample_raw_data})
        comms = result["structured_data"]["flagged_communications"]
        assert len(comms) == 1
        assert comms[0]["flag_reason"] == "structuring_language"

    def test_alerts_extraction(self, sample_raw_data: dict) -> None:
        result = data_ingestion_agent({"raw_data": sample_raw_data})
        alerts = result["structured_data"]["alerts"]
        assert len(alerts) == 1
        assert alerts[0]["alert_id"] == "ALT-001"

    def test_related_entities(self, sample_raw_data: dict) -> None:
        result = data_ingestion_agent({"raw_data": sample_raw_data})
        entities = result["structured_data"]["related_entities"]
        assert len(entities) == 1
        assert entities[0]["jurisdiction"] == "PA"

    def test_empty_transactions(self) -> None:
        state = {"raw_data": {"case_id": "EMPTY", "transactions": []}}
        result = data_ingestion_agent(state)
        sd = result["structured_data"]
        assert sd["transaction_summary"]["count"] == 0
        assert len(sd["transactions"]) == 0

    def test_risk_flags_aggregation(self, sample_raw_data: dict) -> None:
        result = data_ingestion_agent({"raw_data": sample_raw_data})
        flags = result["structured_data"]["transaction_summary"]["unique_risk_flags"]
        assert "structured_amount" in flags
        assert "layering_pattern" in flags
