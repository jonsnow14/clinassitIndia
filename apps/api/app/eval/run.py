"""Eval CLI. PR-1 implements `--suite unit_ops` (ops_judge over agents.jsonl)."""
from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .agents.ops_judge import judge_ops
from .load import load_suite
from .schema import GoldAgent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ClinAssistIndia eval runner")
    parser.add_argument(
        "--suite",
        type=str,
        choices=["unit_ops", "retrieval", "clinical", "full", "journeys"],
        default="unit_ops",
        help="Suite to run (only unit_ops is implemented in PR-1)",
    )
    parser.add_argument("--mode", type=str, choices=["unit", "replay", "live", "chroma"], default="unit")
    parser.add_argument("--baseline", type=str, help="Path to baseline json")
    parser.add_argument("--case", type=str, help="Single case ID to run")
    parser.add_argument("--dump-notes", action="store_true", help="Dump worker notes to traces")
    parser.add_argument("--judge", type=str, choices=["none", "sarvam", "external"], default="none")
    parser.add_argument("--strict-judge", action="store_true", help="Treat judge timeout as critical fail")
    return parser


def run_unit_ops(case_id: str | None = None) -> tuple[int, list[dict]]:
    """Load agent gold, judge each row, return (exit_code, scorecard rows)."""
    try:
        rows = load_suite("agents")
    except (OSError, ValueError) as exc:
        return 2, [{"error": f"gold load failed: {exc}"}]

    agents = [r for r in rows if isinstance(r, GoldAgent)]
    if case_id:
        agents = [r for r in agents if r.id == case_id]
        if not agents:
            return 2, [{"error": f"case {case_id} not found in agents.jsonl"}]

    scorecard: list[dict] = []
    for row in agents:
        result = judge_ops(row.fn, row.args, row.expect)
        scorecard.append(
            {
                "id": row.id,
                "fn": row.fn,
                "status": result["status"],
                "evidence": result.get("evidence") or {},
            }
        )

    total = len(scorecard)
    passed = sum(1 for row in scorecard if row["status"] == "pass")
    if total == 0 or passed != total:
        return 1, scorecard
    return 0, scorecard


def _print_scorecard(scorecard: list[dict]) -> None:
    if len(scorecard) == 1 and "error" in scorecard[0]:
        print(scorecard[0]["error"], file=sys.stderr)
        return
    for row in scorecard:
        reason = (row.get("evidence") or {}).get("reason", "")
        extra = f"\t{reason}" if reason else ""
        print(f"{row['id']}\t{row['status']}{extra}")
    total = len(scorecard)
    passed = sum(1 for row in scorecard if row.get("status") == "pass")
    rate = (passed / total) if total else 0.0
    print(f"ops_assertion_pass_rate\t{rate:.2f}\t{passed}/{total}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.suite != "unit_ops":
        print(f"Suite {args.suite} not implemented in PR-1 skeleton.", file=sys.stderr)
        return 2
    code, scorecard = run_unit_ops(case_id=args.case)
    _print_scorecard(scorecard)
    return code


if __name__ == "__main__":
    sys.exit(main())
