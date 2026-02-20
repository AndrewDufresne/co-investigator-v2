"""Evaluation Runner — batch evaluation across sample cases.

Usage:
    python -m evaluation.runner --mode minimal
    python -m evaluation.runner --mode full
    python -m evaluation.runner --mode offline   # no LLM, evaluate crime detection only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.scoring import evaluate_narrative, EvaluationResult

SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

logger = logging.getLogger(__name__)


def _load_all_samples() -> list[tuple[str, dict]]:
    """Load all JSON sample cases."""
    cases = []
    for p in sorted(SAMPLES_DIR.glob("*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        cases.append((p.name, data))
    return cases


def evaluate_minimal(cases: list[tuple[str, dict]]) -> list[EvaluationResult]:
    """Run Phase 1 minimal pipeline on all cases and evaluate."""
    from src.graph.minimal_graph import build_minimal_graph

    app = build_minimal_graph()
    results: list[EvaluationResult] = []

    for filename, case_data in cases:
        print(f"\n[EVAL] {filename}...")
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        try:
            state = app.invoke(
                {"raw_data": case_data, "iteration_count": 0, "max_iterations": 3},
                config,
            )

            narrative = state.get("narrative_draft", "")
            structured = state.get("structured_data", {})
            risk_indicators = state.get("risk_indicators", [])
            crime_types = state.get("crime_types", [])

            ev = evaluate_narrative(
                narrative=narrative,
                source_data=structured,
                risk_indicators=risk_indicators,
                crime_types=crime_types,
            )
            results.append(ev)
            print(ev.summary())

        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append(EvaluationResult(
                case_id=case_data.get("case_id", filename),
                overall_score=0.0,
                passed=False,
            ))

    return results


def evaluate_full(cases: list[tuple[str, dict]]) -> list[EvaluationResult]:
    """Run Phase 2 full pipeline on all cases and evaluate."""
    from src.graph.sar_graph import build_sar_graph

    app = build_sar_graph(interrupt_before=[])
    results: list[EvaluationResult] = []

    for filename, case_data in cases:
        print(f"\n[EVAL] {filename}...")
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        try:
            state = app.invoke(
                {"raw_data": case_data, "iteration_count": 0, "max_iterations": 3},
                config,
            )

            narrative = state.get("narrative_draft", "")
            structured = state.get("structured_data", {})
            risk_indicators = state.get("risk_indicators", [])
            crime_types = state.get("crime_types", [])

            ev = evaluate_narrative(
                narrative=narrative,
                source_data=structured,
                risk_indicators=risk_indicators,
                crime_types=crime_types,
            )
            results.append(ev)
            print(ev.summary())

        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append(EvaluationResult(
                case_id=case_data.get("case_id", filename),
                overall_score=0.0,
                passed=False,
            ))

    return results


def evaluate_offline(cases: list[tuple[str, dict]]) -> list[EvaluationResult]:
    """Evaluate non-LLM agents only (ingestion + crime detection)."""
    from src.agents.ingestion import data_ingestion_agent
    from src.agents.crime_detection import crime_detection_agent

    results: list[EvaluationResult] = []

    for filename, case_data in cases:
        print(f"\n[EVAL] {filename} (offline)...")

        try:
            ingested = data_ingestion_agent({"raw_data": case_data})
            structured = ingested["structured_data"]

            detected = crime_detection_agent({"structured_data": structured})
            risk_indicators = detected["risk_indicators"]
            crime_types = detected["crime_types"]

            print(f"  Risk indicators: {len(risk_indicators)}")
            print(f"  Crime types: {[ct['type'] for ct in crime_types]}")

            # For offline mode, create a dummy evaluation
            ev = EvaluationResult(
                case_id=structured.get("case_id", filename),
                overall_score=1.0 if crime_types else 0.0,
                passed=len(crime_types) > 0,
            )
            results.append(ev)

        except Exception as e:
            print(f"  [ERROR] {e}")

    return results


def _print_summary(results: list[EvaluationResult]) -> None:
    """Print batch evaluation summary."""
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    avg_score = sum(r.overall_score for r in results) / max(total, 1)

    print(f"  Cases evaluated: {total}")
    print(f"  Passed: {passed}/{total}")
    print(f"  Average score: {avg_score:.1%}")

    for r in results:
        icon = "✅" if r.passed else "❌"
        print(f"  {icon} {r.case_id}: {r.overall_score:.1%}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="SAR Pipeline Evaluation Runner")
    parser.add_argument(
        "--mode",
        choices=["minimal", "full", "offline"],
        default="offline",
        help="Evaluation mode",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    cases = _load_all_samples()
    print(f"[OK] Loaded {len(cases)} sample cases")

    if args.mode == "minimal":
        results = evaluate_minimal(cases)
    elif args.mode == "full":
        results = evaluate_full(cases)
    else:
        results = evaluate_offline(cases)

    _print_summary(results)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"eval_{args.mode}.json"
    out_data = [
        {"case_id": r.case_id, "overall_score": r.overall_score, "passed": r.passed}
        for r in results
    ]
    out_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    print(f"[OK] Results saved to {out_path}")


if __name__ == "__main__":
    main()
