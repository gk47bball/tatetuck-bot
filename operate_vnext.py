"""
operate_vnext.py — end-to-end operational runner for the vNext platform.

Archives the current universe, replays archived snapshots, materializes labels,
optionally evaluates the model, and prints a readiness audit in one pass.
"""

from __future__ import annotations

import argparse
import time

import prepare

from biopharma_agent.vnext import VNextSettings
from biopharma_agent.vnext.pipeline import run_vnext_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the end-to-end vNext operating pipeline.")
    parser.add_argument("--ticker", help="Restrict the run to a single ticker.")
    parser.add_argument("--company-name", help="Optional company name when running a single ticker.")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of benchmark names archived this run.")
    parser.add_argument("--replay-limit", type=int, default=0, help="Only replay the most recent N archived snapshots.")
    parser.add_argument("--include-literature", action="store_true", help="Include literature synthesis during archive.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip walk-forward evaluation in the orchestrated run.")
    return parser.parse_args()


def resolve_universe(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.ticker:
        ticker = args.ticker.upper()
        return [(ticker, args.company_name or ticker)]
    universe = list(prepare.BENCHMARK_TICKERS)
    if args.limit and args.limit > 0:
        return universe[: args.limit]
    return universe


def main() -> None:
    args = parse_args()
    settings = VNextSettings.from_env()
    universe = resolve_universe(args)
    start = time.time()
    summary = run_vnext_pipeline(
        universe=universe,
        settings=settings,
        include_literature=args.include_literature or settings.include_literature,
        replay_limit=args.replay_limit,
        run_evaluation=not args.skip_eval,
    )

    print("=" * 72)
    print("  TATETUCK BOT vNEXT — OPERATING PIPELINE")
    print("=" * 72)
    print(f"store_dir:                 {summary.store_dir}")
    print(f"archived_companies:        {summary.archive.archived_companies}")
    print(f"replayed_snapshots:        {summary.replay.replayed_snapshots}")
    print(f"snapshot_label_rows:       {summary.labels.snapshot_label_rows}")
    print(f"matured_return_90d_rows:   {summary.labels.matured_return_90d_rows}")
    print(f"walkforward_windows:       {summary.evaluation.num_windows}")
    print(f"readiness_status:          {summary.readiness.status}")
    print(f"readiness_blockers:        {len(summary.readiness.blockers)}")

    print("\n[blockers]")
    if summary.readiness.blockers:
        for blocker in summary.readiness.blockers:
            print(f"- {blocker}")
    else:
        print("- none")

    print("\n[top ideas]")
    for idea in summary.archive.top_ideas:
        print(
            f"{idea['ticker']:5} | "
            f"wt={float(idea['target_weight']):>4.1f}% | "
            f"horizon={idea['thesis_horizon']} | "
            f"scenario={idea['scenario']}"
        )

    print(f"\n[summary] elapsed_seconds={time.time() - start:.1f}")


if __name__ == "__main__":
    main()
