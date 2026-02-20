"""CLI Runner — execute the SAR pipeline from the command line.

Usage:
    # Phase 1 minimal pipeline (ingest → crime_detect → narrative):
    python -m src.runner --mode minimal --case data/samples/case_structuring.json

    # Phase 2 full pipeline (all agents, compliance loop):
    python -m src.runner --mode full --case data/samples/case_structuring.json

    # Full pipeline without HITL interrupt:
    python -m src.runner --mode full --case data/samples/case_structuring.json --no-interrupt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def _load_case(path: str) -> dict:
    """Load a JSON case file."""
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] Case file not found: {p}")
        sys.exit(1)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[OK] Loaded case: {data.get('case_id', 'unknown')} from {p.name}")
    return data


def run_minimal(case_data: dict, verbose: bool = False) -> dict:
    """Execute Phase 1 minimal pipeline: ingest → crime_detect → narrative."""
    from src.graph.minimal_graph import build_minimal_graph

    print("\n" + "=" * 70)
    print("  PHASE 1 — Minimal Pipeline (3 nodes)")
    print("  ingest → crime_detect → narrative → END")
    print("=" * 70 + "\n")

    app = build_minimal_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "raw_data": case_data,
        "iteration_count": 0,
        "max_iterations": 3,
    }

    print("[RUN] Starting minimal pipeline...")
    result = app.invoke(initial_state, config)

    _print_results(result)
    return result


def run_full(case_data: dict, interrupt: bool = True, verbose: bool = False) -> dict:
    """Execute Phase 2 full pipeline with all agents and compliance loop."""
    from src.graph.sar_graph import build_sar_graph

    print("\n" + "=" * 70)
    print("  PHASE 2 — Full Pipeline (10 nodes + typology subgraph)")
    print("  ingest → mask → crime_detect → plan → typology →")
    print("  [external_intel] → narrative → compliance → unmask → END")
    print("=" * 70 + "\n")

    interrupt_before = ["unmask"] if interrupt else []
    app = build_sar_graph(interrupt_before=interrupt_before)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "raw_data": case_data,
        "iteration_count": 0,
        "max_iterations": 3,
    }

    print("[RUN] Starting full pipeline...")
    if interrupt:
        print("[INFO] Pipeline will pause at 'unmask' for human review.")
        print("[INFO] Use --no-interrupt to skip the HITL pause.\n")

    result = app.invoke(initial_state, config)

    # If interrupted at unmask, resume
    if interrupt:
        snapshot = app.get_state(config)
        if snapshot.next and "unmask" in snapshot.next:
            print("\n[PAUSE] Pipeline paused before unmask. Resuming automatically...\n")
            result = app.invoke(None, config)

    _print_results(result)
    return result


def _print_results(result: dict) -> None:
    """Pretty-print pipeline results."""
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    # Crime types
    crime_types = result.get("crime_types", [])
    if crime_types:
        print(f"\n📋 Crime Types Detected ({len(crime_types)}):")
        for ct in crime_types:
            print(f"   • {ct['type']} (confidence: {ct['confidence']:.0%})")
            print(f"     Evidence: {', '.join(ct.get('evidence', []))}")

    # Risk indicators
    indicators = result.get("risk_indicators", [])
    if indicators:
        print(f"\n⚠️  Risk Indicators ({len(indicators)}):")
        for ind in indicators:
            print(f"   [{ind.get('severity', '?').upper()}] {ind['type']}: {ind['description']}")

    # Typology results
    typology = result.get("typology_results", {})
    if typology:
        print(f"\n🔬 Typology Analysis:")
        for agent_name, agent_result in typology.items():
            if agent_name.startswith("_"):
                continue
            if isinstance(agent_result, dict):
                findings = agent_result.get("findings", [])
                score = agent_result.get("risk_score", 0)
                print(f"   {agent_name}: {len(findings)} findings, risk_score={score:.2f}")
                for f in findings[:3]:  # show top 3
                    print(f"     • [{f.get('severity', '?')}] {f.get('pattern')}: {f.get('detail', '')[:80]}")
        agg = typology.get("_aggregate", {})
        if agg:
            print(f"   ── Aggregate: avg_risk={agg.get('average_risk_score', 0):.3f}, "
                  f"total_findings={agg.get('total_findings', 0)}, "
                  f"agents={agg.get('agents_run', [])}")

    # Compliance
    compliance = result.get("compliance_result", {})
    if compliance:
        status = compliance.get("status", "N/A")
        score = result.get("compliance_score", 0)
        emoji = "✅" if status == "PASS" else "❌"
        print(f"\n{emoji} Compliance: {status} (score: {score:.1%})")
        for check in compliance.get("checks", []):
            icon = "✓" if check["passed"] else "✗"
            print(f"   [{icon}] {check['dimension']}: {check['details']}")

    # Narrative
    narrative = result.get("narrative_draft", "")
    if narrative:
        print(f"\n📝 SAR Narrative ({len(narrative)} chars):")
        print("─" * 60)
        # Print first 2000 chars
        if len(narrative) > 2000:
            print(narrative[:2000])
            print(f"\n... [truncated, {len(narrative) - 2000} more chars]")
        else:
            print(narrative)
        print("─" * 60)

    # Chain of thought
    cot = result.get("chain_of_thought", [])
    if cot:
        print(f"\n🧠 Chain of Thought ({len(cot)} steps):")
        for step in cot[:5]:
            print(f"   → {step}")

    iterations = result.get("iteration_count", 0)
    if iterations:
        print(f"\n🔄 Iterations: {iterations}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Co-Investigator SAR Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.runner --mode minimal --case data/samples/case_structuring.json
  python -m src.runner --mode full --case data/samples/case_structuring.json --no-interrupt
  python -m src.runner --mode full --case data/samples/case_elder_exploit.json -v
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["minimal", "full"],
        default="minimal",
        help="Pipeline mode: 'minimal' (Phase 1, 3 nodes) or 'full' (Phase 2, all agents)",
    )
    parser.add_argument(
        "--case",
        required=True,
        help="Path to JSON case file",
    )
    parser.add_argument(
        "--no-interrupt",
        action="store_true",
        help="Disable HITL interrupt before unmask (full mode only)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--output",
        help="Save full narrative to this file path",
    )

    args = parser.parse_args()
    _setup_logging(args.verbose)

    case_data = _load_case(args.case)

    if args.mode == "minimal":
        result = run_minimal(case_data, verbose=args.verbose)
    else:
        result = run_full(case_data, interrupt=not args.no_interrupt, verbose=args.verbose)

    # Optionally save narrative to file
    if args.output:
        narrative = result.get("narrative_draft", "")
        p = Path(args.output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(narrative, encoding="utf-8")
        print(f"[OK] Narrative saved to {p}")

    print("[DONE]")


if __name__ == "__main__":
    main()
