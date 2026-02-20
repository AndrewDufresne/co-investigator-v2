"""Pydantic data models for input/output validation and serialization."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Input Models (JSON case data) ──


class Subject(BaseModel):
    name: str
    dob: str | None = None
    ssn: str | None = None
    address: str | None = None
    phone: str | None = None
    email: str | None = None
    occupation: str | None = None
    risk_rating: str = "medium"
    customer_since: str | None = None


class Account(BaseModel):
    account_id: str
    account_type: str
    opened_date: str | None = None
    balance: float = 0.0
    currency: str = "USD"
    branch: str | None = None


class Transaction(BaseModel):
    txn_id: str
    date: str
    type: str
    amount: float
    currency: str = "USD"
    from_account: str | None = None
    to_account: str | None = None
    from_entity: str | None = None
    to_entity: str | None = None
    from_country: str | None = None
    to_country: str | None = None
    location: str | None = None
    description: str | None = None
    risk_flags: list[str] = Field(default_factory=list)


class AdverseMediaHit(BaseModel):
    date: str
    source: str
    summary: str


class KYC(BaseModel):
    verification_status: str = "pending"
    last_review_date: str | None = None
    source_of_funds: str | None = None
    expected_activity: str | None = None
    actual_activity_profile: str | None = None
    pep_status: bool = False
    adverse_media_hits: list[AdverseMediaHit] = Field(default_factory=list)


class Communication(BaseModel):
    date: str
    channel: str
    direction: str
    content: str
    flagged: bool = False
    flag_reason: str | None = None


class Alert(BaseModel):
    alert_id: str
    type: str
    severity: str = "medium"
    description: str
    triggered_date: str


class RelatedEntity(BaseModel):
    entity_name: str
    entity_type: str
    jurisdiction: str | None = None
    relationship: str
    risk_notes: str | None = None


class CaseData(BaseModel):
    """Top-level input: a complete AML case in JSON format."""

    case_id: str
    alert_date: str
    priority: str = "medium"
    subject: Subject
    accounts: list[Account] = Field(default_factory=list)
    transactions: list[Transaction] = Field(default_factory=list)
    kyc: KYC | None = None
    communications: list[Communication] = Field(default_factory=list)
    alerts: list[Alert] = Field(default_factory=list)
    related_entities: list[RelatedEntity] = Field(default_factory=list)


# ── Output Models ──


class CrimeTypeDetected(BaseModel):
    type: str
    confidence: float


class NarrativeOutput(BaseModel):
    intro: str
    body: str
    conclusion: str


class ComplianceValidation(BaseModel):
    score: float
    status: str
    checks: dict[str, Any] = Field(default_factory=dict)


class SAROutput(BaseModel):
    """Complete SAR generation output."""

    sar_id: str
    case_id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "review"
    crime_types_detected: list[CrimeTypeDetected] = Field(default_factory=list)
    narrative: NarrativeOutput | None = None
    compliance_validation: ComplianceValidation | None = None
    chain_of_thought: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
