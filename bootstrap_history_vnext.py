"""
bootstrap_history_vnext.py — synthesize dated historical snapshots from archived
SEC filings and market history, then rebuild labels and evaluation artifacts.
"""

from __future__ import annotations

import argparse
import time

from biopharma_agent.vnext import HistoricalReplayEngine, PointInTimeLabeler, VNextSettings, record_pipeline_run
from biopharma_agent.vnext.evaluation import WalkForwardEvaluator
from biopharma_agent.vnext.history import HistoricalSnapshotBootstrapper
from biopharma_agent.vnext.ops import utc_now_iso
from biopharma_agent.vnext.storage import LocalResearchStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap historical vNext snapshots from archived SEC and price data.")
    parser.add_argument("--ticker", help="Restrict history synthesis to a single ticker.")
    parser.add_argument("--ticker-limit", type=int, default=0, help="Limit the number of tickers synthesized.")
    parser.add_argument("--max-anchors-per-ticker", type=int, default=8, help="Maximum historical filing anchors per ticker.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip walk-forward evaluation after rebuilding labels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = VNextSettings.from_env()
    store = LocalResearchStore(settings.store_dir)
    start = time.time()
    started_at = utc_now_iso()

    try:
        bootstrapper = HistoricalSnapshotBootstrapper(store=store)
        bootstrap_summary = bootstrapper.materialize(
            ticker=args.ticker,
            ticker_limit=args.ticker_limit,
            max_anchors_per_ticker=args.max_anchors_per_ticker,
        )

        replay = HistoricalReplayEngine(store=store)
        replay_summary = replay.rebuild_from_archived_snapshots(ticker=args.ticker, limit=0)

        labeler = PointInTimeLabeler(store=store)
        label_summary = labeler.materialize_labels()

        evaluation = None
        if not args.skip_eval:
            evaluation = WalkForwardEvaluator(store=store, settings=settings).evaluate()

        finished_at = utc_now_iso()
        record_pipeline_run(
            store=store,
            job_name="bootstrap_history_vnext",
            status="success",
            started_at=started_at,
            finished_at=finished_at,
            metrics={
                "generated_snapshots": bootstrap_summary.generated_snapshots,
                "distinct_anchor_dates": bootstrap_summary.distinct_anchor_dates,
                "replayed_snapshots": replay_summary.replayed_snapshots,
                "matured_return_90d_rows": label_summary.matured_return_90d_rows,
                "walkforward_windows": evaluation.num_windows if evaluation else 0,
            },
            config={
                "ticker": args.ticker,
                "ticker_limit": args.ticker_limit,
                "max_anchors_per_ticker": args.max_anchors_per_ticker,
                "skip_eval": args.skip_eval,
                "store_dir": settings.store_dir,
            },
        )
    except Exception as exc:
        finished_at = utc_now_iso()
        record_pipeline_run(
            store=store,
            job_name="bootstrap_history_vnext",
            status="failed",
            started_at=started_at,
            finished_at=finished_at,
            metrics={},
            config={
                "ticker": args.ticker,
                "ticker_limit": args.ticker_limit,
                "max_anchors_per_ticker": args.max_anchors_per_ticker,
                "skip_eval": args.skip_eval,
                "store_dir": settings.store_dir,
            },
            notes=f"{type(exc).__name__}: {exc}",
        )
        raise

    print("=" * 72)
    print("  TATETUCK BOT vNEXT — HISTORY BOOTSTRAP")
    print("=" * 72)
    print(f"generated_snapshots:       {bootstrap_summary.generated_snapshots}")
    print(f"tickers_processed:         {bootstrap_summary.tickers_processed}")
    print(f"tickers_with_history:      {bootstrap_summary.tickers_with_history}")
    print(f"skipped_tickers:           {bootstrap_summary.skipped_tickers}")
    print(f"distinct_anchor_dates:     {bootstrap_summary.distinct_anchor_dates}")
    print(f"earliest_as_of:            {bootstrap_summary.earliest_as_of}")
    print(f"latest_as_of:              {bootstrap_summary.latest_as_of}")
    print(f"replayed_snapshots:        {replay_summary.replayed_snapshots}")
    print(f"snapshot_label_rows:       {label_summary.snapshot_label_rows}")
    print(f"event_label_rows:          {label_summary.event_label_rows}")
    print(f"matured_return_90d_rows:   {label_summary.matured_return_90d_rows}")
    print(f"matured_event_rows:        {label_summary.matured_event_rows}")

    if evaluation is not None:
        print("\n[walk-forward]")
        print(f"rows:                     {evaluation.num_rows}")
        print(f"windows:                  {evaluation.num_windows}")
        print(f"rank_ic:                  {evaluation.rank_ic:.4f}")
        print(f"hit_rate:                 {evaluation.hit_rate:.4f}")
        print(f"top_bottom_spread:        {evaluation.top_bottom_spread:.4f}")
        print(f"turnover:                 {evaluation.turnover:.4f}")
        print(f"max_drawdown:             {evaluation.max_drawdown:.4f}")
        print(f"beta_adj_return:          {evaluation.beta_adjusted_return:.4f}")
        print(f"brier:                    {evaluation.calibrated_brier:.4f}")
        print(f"leakage_passed:           {evaluation.leakage_passed}")
        if evaluation.message:
            print(f"message:                  {evaluation.message}")

    print(f"\n[summary] elapsed_seconds={time.time() - start:.1f}")


if __name__ == "__main__":
    main()
