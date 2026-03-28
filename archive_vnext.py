"""
archive_vnext.py — archive current vNext snapshots into the local research store.

Run this on a recurring basis to accumulate dated company snapshots, features,
and predictions for future walk-forward evaluation.
"""

from __future__ import annotations

import argparse
import time

import prepare

from biopharma_agent.vnext import TatetuckPlatform, VNextSettings, archive_universe, record_pipeline_run
from biopharma_agent.vnext.ops import utc_now_iso
from biopharma_agent.vnext.storage import LocalResearchStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive Tatetuck vNext company snapshots into the local store.")
    parser.add_argument("--ticker", help="Archive a single ticker instead of the benchmark universe.")
    parser.add_argument("--company-name", help="Optional company name when archiving a single ticker.")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of benchmark names archived in this run.")
    parser.add_argument(
        "--include-literature",
        action="store_true",
        help="Include literature synthesis during the archive run.",
    )
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
    universe = resolve_universe(args)
    settings = VNextSettings.from_env()
    store = LocalResearchStore(settings.store_dir)
    platform = TatetuckPlatform(store=store)
    started_at = utc_now_iso()
    start = time.time()

    try:
        _, summary = archive_universe(
            platform,
            universe,
            include_literature=args.include_literature or settings.include_literature,
        )
        finished_at = utc_now_iso()
        record_pipeline_run(
            store=store,
            job_name="archive_vnext",
            status="success",
            started_at=started_at,
            finished_at=finished_at,
            metrics={
                "archived_companies": summary.archived_companies,
                "sec_enriched_companies": summary.sec_enriched_companies,
                "snapshot_rows": summary.snapshot_rows,
                "feature_rows": summary.feature_rows,
                "prediction_rows": summary.prediction_rows,
            },
            config={
                "ticker": args.ticker,
                "limit": args.limit,
                "include_literature": args.include_literature or settings.include_literature,
                "store_dir": settings.store_dir,
            },
        )
    except Exception as exc:
        finished_at = utc_now_iso()
        record_pipeline_run(
            store=store,
            job_name="archive_vnext",
            status="failed",
            started_at=started_at,
            finished_at=finished_at,
            metrics={"archived_companies": 0},
            config={"ticker": args.ticker, "limit": args.limit, "store_dir": settings.store_dir},
            notes=f"{type(exc).__name__}: {exc}",
        )
        raise

    print("=" * 72)
    print("  TATETUCK BOT vNEXT — SNAPSHOT ARCHIVE")
    print("=" * 72)
    print(f"archived_companies:        {summary.archived_companies}")
    print(f"archived_at:               {summary.archived_at}")
    print(f"sec_enriched_companies:    {summary.sec_enriched_companies}")
    print(f"financing_flagged_names:   {summary.financing_flagged_companies}")
    print(f"snapshot_rows:             {summary.snapshot_rows}")
    print(f"feature_rows:              {summary.feature_rows}")
    print(f"prediction_rows:           {summary.prediction_rows}")
    print(f"store_dir:                 {summary.store_dir}")
    print("\n[top ideas]")
    for idea in summary.top_ideas:
        print(
            f"{idea['ticker']:5} | "
            f"wt={float(idea['target_weight']):>4.1f}% | "
            f"horizon={idea['thesis_horizon']} | "
            f"scenario={idea['scenario']}"
        )
    print(f"\n[summary] elapsed_seconds={time.time() - start:.1f}")


if __name__ == "__main__":
    main()
