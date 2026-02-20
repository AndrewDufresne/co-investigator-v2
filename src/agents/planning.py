"""Planning Agent — central orchestrator that decides which agents to activate.

Calls DeepSeek to generate an execution plan based on detected crime types.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.core.llm_gateway import get_llm
from src.core.state import SARState

logger = logging.getLogger(__name__)

ALL_TYPOLOGY_AGENTS = [
    "transaction_fraud",
    "payment_velocity",
    "country_risk",
    "text_content",
    "geo_anomaly",
    "account_health",
    "dispute_pattern",
]

PLANNING_SYSTEM_PROMPT = """You are the Planning Agent for an AML compliance investigation system.
Given the detected crime types and risk indicators, you must decide:
1. Which specialized typology detection agents to activate (from the available list).
2. Whether external intelligence gathering is needed.
3. The narrative focus and structure for the SAR report.

Available typology agents:
- transaction_fraud: Analyzes transaction amount/frequency/counterparty patterns
- payment_velocity: Detects abnormal transaction frequencies or high-volume activity
- country_risk: Evaluates transactions linked to high-risk jurisdictions
- text_content: NLP analysis of communications and transaction notes
- geo_anomaly: Detects location-based inconsistencies
- account_health: Assesses historical account behavior for anomalies
- dispute_pattern: Analyzes dispute/chargeback patterns

Respond ONLY with valid JSON in this exact format:
{
    "active_typology_agents": ["agent_name1", "agent_name2"],
    "requires_external_intel": true/false,
    "narrative_focus": "Brief description of what the narrative should emphasize",
    "narrative_structure": ["section1", "section2", "section3"]
}"""


def planning_agent(state: SARState) -> dict[str, Any]:
    """Generate an execution plan based on crime type detection results."""
    crime_types = state.get("crime_types", [])
    risk_indicators = state.get("risk_indicators", [])
    case_id = state.get("masked_data", {}).get("case_id", "unknown")

    logger.info("Planning agent: creating execution plan for case %s", case_id)

    # Build context message
    context = (
        f"Case ID: {case_id}\n\n"
        f"Detected Crime Types:\n{json.dumps(crime_types, indent=2)}\n\n"
        f"Risk Indicators:\n{json.dumps(risk_indicators, indent=2)}"
    )

    try:
        llm = get_llm(role="planning")
        response = llm.invoke([
            SystemMessage(content=PLANNING_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ])

        # Parse JSON from response
        raw_content = response.content
        if isinstance(raw_content, str):
            content = raw_content.strip()
        elif isinstance(raw_content, list):
            parts: list[str] = []
            for item in raw_content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            content = "\n".join(parts).strip()
        else:
            content = str(raw_content).strip()

        # Handle markdown code blocks
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        plan = json.loads(content)

    except Exception as e:
        logger.warning("Planning agent LLM call failed (%s), using fallback plan", e)
        # Fallback: activate agents based on detected indicators
        active_agents = set()
        for indicator in risk_indicators:
            ind_type = indicator.get("type", "")
            if ind_type in ("structuring", "layering"):
                active_agents.update(["transaction_fraud", "payment_velocity"])
            if ind_type == "high_risk_jurisdiction":
                active_agents.add("country_risk")
            if ind_type == "shell_company":
                active_agents.add("account_health")
            if ind_type == "suspicious_communication":
                active_agents.add("text_content")
            if ind_type == "velocity_anomaly":
                active_agents.add("payment_velocity")
        if not active_agents:
            active_agents = {"transaction_fraud", "country_risk", "account_health"}

        plan = {
            "active_typology_agents": list(active_agents),
            "requires_external_intel": True,
            "narrative_focus": "Comprehensive analysis of detected suspicious activity patterns",
            "narrative_structure": [
                "Subject and account identification",
                "Suspicious activity description",
                "Supporting evidence and analysis",
                "Conclusion and risk assessment",
            ],
        }

    # Validate agent names
    valid_agents = [a for a in plan.get("active_typology_agents", []) if a in ALL_TYPOLOGY_AGENTS]
    if not valid_agents:
        valid_agents = ["transaction_fraud", "country_risk", "account_health"]

    execution_plan = {
        "active_typology_agents": valid_agents,
        "requires_external_intel": plan.get("requires_external_intel", True),
        "narrative_focus": plan.get("narrative_focus", ""),
        "narrative_structure": plan.get("narrative_structure", []),
    }

    logger.info("Planning complete: activating agents %s", valid_agents)

    return {
        "execution_plan": execution_plan,
        "active_typology_agents": valid_agents,
    }
