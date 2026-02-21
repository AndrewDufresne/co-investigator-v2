"""Narrative Generation Agent — synthesizes SAR narrative drafts using CoT prompting.

Calls DeepSeek with Chain-of-Thought to produce FinCEN-compliant SAR narratives.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.core.llm_gateway import get_llm
from src.core.state import SARState

logger = logging.getLogger(__name__)

NARRATIVE_SYSTEM_PROMPT = """You are a senior AML compliance analyst generating a Suspicious Activity Report (SAR) narrative.

Your task: produce a clear, factual, regulatory-compliant SAR narrative following FinCEN guidelines.

RULES:
1. Use ONLY facts from the provided case data — NEVER invent or hallucinate details.
2. Follow the 5W1H framework: Who, What, When, Where, Why, How.
3. Be specific with amounts, dates, account numbers, and entity names (use the masked placeholders as-is).
4. Do NOT "tip off" the subject — maintain objective, factual language.
5. Structure the narrative with clear sections: Intro, Activity Description, Analysis, Conclusion.
6. Reference specific transaction IDs and risk indicators as evidence.

CHAIN-OF-THOUGHT: Before writing the narrative, first outline your reasoning step by step, prefixed with "REASONING:" on separate lines. Then write the narrative prefixed with "NARRATIVE:".

FORMAT:
REASONING:
- Step 1: ...
- Step 2: ...
...

NARRATIVE_INTRO:
[One paragraph introducing the SAR filing — who is being reported, for what activity, during what period]

NARRATIVE_BODY:
[Detailed narrative covering all suspicious activity, evidence, and analysis]

NARRATIVE_CONCLUSION:
[Summary of findings and why activity is suspicious]
"""


def _build_context(state: SARState) -> str:
    """Build the context message from state for the LLM."""
    data = state.get("masked_data") or state.get("structured_data", {})
    crime_types = state.get("crime_types", [])
    risk_indicators = state.get("risk_indicators", [])
    typology_results = state.get("typology_results", {})
    external_intel = state.get("external_intel", [])
    execution_plan = state.get("execution_plan", {})
    human_feedback = state.get("human_feedback")
    iteration = state.get("iteration_count", 0)

    parts = [
        "=== CASE DATA ===",
        f"Case ID: {data.get('case_id', 'N/A')}",
        f"Alert Date: {data.get('alert_date', 'N/A')}",
        f"Priority: {data.get('priority', 'N/A')}",
        "",
        f"Subject: {json.dumps(data.get('subject', {}), indent=2)}",
        "",
        f"Accounts: {json.dumps(data.get('accounts', []), indent=2)}",
        "",
        f"Transaction Summary: {json.dumps(data.get('transaction_summary', {}), indent=2)}",
        "",
        "Transactions:",
        json.dumps(data.get("transactions", []), indent=2),
        "",
        f"KYC: {json.dumps(data.get('kyc', {}), indent=2)}",
        "",
        f"Flagged Communications: {json.dumps(data.get('flagged_communications', []), indent=2)}",
        "",
        f"Alerts: {json.dumps(data.get('alerts', []), indent=2)}",
        "",
        f"Related Entities: {json.dumps(data.get('related_entities', []), indent=2)}",
        "",
        "=== ANALYSIS RESULTS ===",
        f"Crime Types Detected: {json.dumps(crime_types, indent=2)}",
        "",
        f"Risk Indicators: {json.dumps(risk_indicators, indent=2)}",
    ]

    if typology_results:
        parts.extend([
            "",
            f"Typology Agent Results: {json.dumps(typology_results, indent=2)}",
        ])

    if external_intel:
        parts.extend([
            "",
            f"External Intelligence: {json.dumps(external_intel, indent=2)}",
        ])

    if execution_plan:
        parts.extend([
            "",
            f"Narrative Focus: {execution_plan.get('narrative_focus', '')}",
            f"Narrative Structure: {json.dumps(execution_plan.get('narrative_structure', []))}",
        ])

    if human_feedback and iteration > 0:
        parts.extend([
            "",
            f"=== INVESTIGATOR FEEDBACK (iteration {iteration}) ===",
            human_feedback,
            "",
            "Please revise the narrative based on this feedback while maintaining compliance.",
        ])

    return "\n".join(parts)


def _parse_response(content: str) -> tuple[list[str], str, str]:
    """Parse the LLM response into reasoning chain, intro, and body."""
    reasoning_lines: list[str] = []
    intro = ""
    body = ""

    sections = content.split("NARRATIVE_INTRO:")
    if len(sections) >= 2:
        # Extract reasoning
        reasoning_part = sections[0]
        if "REASONING:" in reasoning_part:
            reasoning_text = reasoning_part.split("REASONING:", 1)[1].strip()
            reasoning_lines = [
                line.strip().lstrip("- ")
                for line in reasoning_text.split("\n")
                if line.strip()
            ]

        # Extract intro and body
        rest = sections[1]
        if "NARRATIVE_BODY:" in rest:
            intro_body = rest.split("NARRATIVE_BODY:", 1)
            intro = intro_body[0].strip()
            remainder = intro_body[1]
            if "NARRATIVE_CONCLUSION:" in remainder:
                body_conclusion = remainder.split("NARRATIVE_CONCLUSION:", 1)
                body = body_conclusion[0].strip() + "\n\n" + body_conclusion[1].strip()
            else:
                body = remainder.strip()
        else:
            intro = rest.strip()
    else:
        # Fallback: treat entire response as narrative body
        body = content.strip()

    return reasoning_lines, intro, body


def narrative_generation_agent(state: SARState) -> dict[str, Any]:
    """Generate SAR narrative draft using Chain-of-Thought prompting."""
    data = state.get("masked_data") or state.get("structured_data", {})
    case_id = data.get("case_id", "unknown")
    iteration = state.get("iteration_count", 0)

    logger.info(
        "Narrative generation: drafting SAR for case %s (iteration %d)",
        case_id,
        iteration,
    )

    context = _build_context(state)

    try:
        llm = get_llm(role="narrative")
        response = llm.invoke([
            SystemMessage(content=NARRATIVE_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ])

        raw_content = response.content
        if isinstance(raw_content, str):
            content = raw_content
        elif isinstance(raw_content, list):
            parts: list[str] = []
            for item in raw_content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            content = "\n".join(parts)
        else:
            content = str(raw_content)

        reasoning, intro, body = _parse_response(content)

        # Combine into full narrative draft
        full_draft = f"{intro}\n\n{body}" if intro else body

    except Exception as e:
        logger.error("Narrative generation failed: %s", e)
        reasoning = [f"Error during generation: {e}"]
        intro = "Error: narrative generation failed. Please retry."
        full_draft = intro

    logger.info("Narrative generation complete (%d chars)", len(full_draft))

    return {
        "narrative_draft": full_draft,
        "narrative_intro": intro,
        "chain_of_thought": reasoning,
    }
