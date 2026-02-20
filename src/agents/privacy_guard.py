"""AI-Privacy Guard Agent — masks/unmasks PII before/after LLM processing.

MVP: Simple regex + pattern-based PII detection.
Future: Replace with RoBERTa + CRF model per the paper.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any

from src.core.state import SARState

logger = logging.getLogger(__name__)

# PII patterns for MVP detection
PII_PATTERNS = {
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "PHONE": re.compile(r"\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
}


def _generate_placeholder(entity_type: str, index: int) -> str:
    """Generate a deterministic placeholder token."""
    return f"[{entity_type}_{index:03d}]"


def _deep_mask(obj: Any, mapping: dict[str, str], reverse_map: dict[str, str]) -> Any:
    """Recursively walk a data structure and replace PII values with placeholders."""
    if isinstance(obj, str):
        result = obj
        for pii_type, pattern in PII_PATTERNS.items():
            for match in pattern.finditer(result):
                original = match.group()
                if original not in mapping:
                    idx = len(mapping)
                    placeholder = _generate_placeholder(pii_type, idx)
                    mapping[original] = placeholder
                    reverse_map[placeholder] = original
                result = result.replace(original, mapping[original])
        return result
    elif isinstance(obj, dict):
        # Mask specific known PII field names
        pii_field_names = {"ssn", "name", "address", "phone", "email", "dob"}
        masked = {}
        for k, v in obj.items():
            if k.lower() in pii_field_names and isinstance(v, str) and v:
                if v not in mapping:
                    idx = len(mapping)
                    field_type = k.upper()
                    placeholder = _generate_placeholder(field_type, idx)
                    mapping[v] = placeholder
                    reverse_map[placeholder] = v
                masked[k] = mapping[v]
            else:
                masked[k] = _deep_mask(v, mapping, reverse_map)
        return masked
    elif isinstance(obj, list):
        return [_deep_mask(item, mapping, reverse_map) for item in obj]
    else:
        return obj


def _deep_unmask(obj: Any, reverse_map: dict[str, str]) -> Any:
    """Recursively restore original values from placeholders."""
    if isinstance(obj, str):
        result = obj
        for placeholder, original in reverse_map.items():
            result = result.replace(placeholder, original)
        return result
    elif isinstance(obj, dict):
        return {k: _deep_unmask(v, reverse_map) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_deep_unmask(item, reverse_map) for item in obj]
    else:
        return obj


def privacy_mask_agent(state: SARState) -> dict[str, Any]:
    """Mask PII in structured_data before sending to LLM agents.

    Returns masked_data and mask_mapping for later restoration.
    """
    structured = state.get("structured_data")
    if structured is None:
        raise ValueError("Missing required state key: structured_data")
    logger.info("Privacy Guard: masking PII for case %s", structured.get("case_id", "unknown"))

    mapping: dict[str, str] = {}        # original → placeholder
    reverse_map: dict[str, str] = {}    # placeholder → original

    masked = _deep_mask(structured, mapping, reverse_map)

    logger.info("Privacy Guard: masked %d PII entities", len(mapping))

    return {
        "masked_data": masked,
        "mask_mapping": reverse_map,  # store reverse map for unmask
    }


def privacy_unmask_agent(state: SARState) -> dict[str, Any]:
    """Restore real PII values in the final narrative draft.

    Uses the mask_mapping (reverse map) created during masking.
    """
    narrative = state.get("narrative_draft", "")
    intro = state.get("narrative_intro", "")
    reverse_map = state.get("mask_mapping", {})

    logger.info("Privacy Guard: unmasking narrative (%d mappings)", len(reverse_map))

    unmasked_narrative = _deep_unmask(narrative, reverse_map)
    unmasked_intro = _deep_unmask(intro, reverse_map)

    return {
        "final_narrative": unmasked_narrative,
        "narrative_intro": unmasked_intro,
        "narrative_draft": unmasked_narrative,
        "status": "review",
    }
