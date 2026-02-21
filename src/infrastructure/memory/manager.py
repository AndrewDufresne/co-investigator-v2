"""Dynamic Memory Manager — manages regulatory, historical, and typology memory stores.

MVP: In-memory stores using dicts. Future: ChromaDB vector store + SQLite.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.config import get_settings

logger = logging.getLogger(__name__)


class MemoryManager:
    """Three-tier memory system for the Argus agents.

    Stores:
        - regulatory: AML rules, FinCEN guidelines, compliance templates
        - historical: Past SAR narratives and case patterns
        - typology:   Crime type templates and detection patterns
    """

    def __init__(self) -> None:
        self._regulatory: dict[str, Any] = {}
        self._historical: list[dict[str, Any]] = []
        self._typology: dict[str, Any] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Load memory stores from disk or initialize defaults."""
        if self._initialized:
            return

        settings = get_settings()
        db_dir = settings.db_dir

        # Load regulatory memory
        reg_path = db_dir / "regulatory.json"
        if reg_path.exists():
            self._regulatory = json.loads(reg_path.read_text(encoding="utf-8"))
        else:
            self._regulatory = {
                "ctr_threshold": 10000,
                "sar_filing_days": 30,
                "required_narrative_elements": [
                    "subject_identification",
                    "suspicious_activity_description",
                    "time_period",
                    "amounts_involved",
                    "how_activity_conducted",
                    "why_suspicious",
                ],
                "prohibited_language": [
                    "we suspect you",
                    "you are being reported",
                    "this SAR is about you",
                ],
            }

        # Load historical memory
        hist_path = db_dir / "historical.json"
        if hist_path.exists():
            self._historical = json.loads(hist_path.read_text(encoding="utf-8"))

        # Load typology memory
        typ_path = db_dir / "typology.json"
        if typ_path.exists():
            self._typology = json.loads(typ_path.read_text(encoding="utf-8"))
        else:
            self._typology = {
                "structuring": {
                    "description": "Breaking large transactions into smaller amounts to avoid CTR reporting",
                    "indicators": ["amounts just below $10,000", "multiple transactions same day", "round amounts"],
                },
                "layering": {
                    "description": "Moving funds through multiple accounts/entities to obscure origin",
                    "indicators": ["rapid internal transfers", "shell companies", "multiple jurisdictions"],
                },
                "trade_based_ml": {
                    "description": "Disguising money laundering through international trade",
                    "indicators": ["over/under invoicing", "phantom shipments", "misrepresented goods"],
                },
            }

        self._initialized = True
        logger.info("Memory manager initialized")

    def get_regulatory_rules(self) -> dict[str, Any]:
        """Return regulatory compliance rules."""
        self.initialize()
        return self._regulatory

    def get_typology_template(self, crime_type: str) -> dict[str, Any] | None:
        """Return typology template for a given crime type."""
        self.initialize()
        return self._typology.get(crime_type)

    def store_historical_case(self, case_summary: dict[str, Any]) -> None:
        """Store a completed case for future reference."""
        self.initialize()
        self._historical.append(case_summary)
        # Persist
        settings = get_settings()
        db_dir = settings.db_dir
        db_dir.mkdir(parents=True, exist_ok=True)
        (db_dir / "historical.json").write_text(
            json.dumps(self._historical, indent=2), encoding="utf-8"
        )

    def search_historical(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search historical cases by simple keyword match (MVP)."""
        self.initialize()
        results = []
        query_lower = query.lower()
        for case in self._historical:
            text = json.dumps(case).lower()
            if query_lower in text:
                results.append(case)
                if len(results) >= limit:
                    break
        return results


# Singleton instance
_memory_manager: MemoryManager | None = None


def get_memory_manager() -> MemoryManager:
    """Return the singleton MemoryManager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
