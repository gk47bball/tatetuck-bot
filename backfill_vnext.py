"""
backfill_vnext.py — rebuild historical vNext tables and labels from archived snapshots.

This script does not hit live company ingestion endpoints. It replays the raw
snapshot archive already stored in `.tatetuck_store/raw/snapshots`, rebuilds
features and predictions, and materializes forward-return/event-window labels.
"""

from __future__ import annotations

import argparse
import time

from biopharma_agent.vnext import HistoricalReplayEngine, PointInTimeLabeler, VNextSettings, record_pipeline_run
from biopharma_agent.vnext.evaluation import WalkForwardEvaluator
from biopharma_agent.vnext.ops import utc_now_iso
from biopharma_agent.vnext.storage import LocalResearchStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill vNext tables and labels from archived raw snapshots.")
    parser.add_argument("--ticker", help="Restrict replay to a single ticker.")
    parser.add_argument("--limit", type=int, default=0, help="Only replay the most recent N archived snapshots.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip the walk-forward evaluation summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = VNextSettings.from_env()
    store = LocalResearchStore(settings.store_dir)
    start = time.time()
    started_at = utc_now_iso()

    try:
        replay = HistoricalReplayEngine(store=store)
        replay_summary = replay.rebuild_from_archived_snapshots(ticker=args.ticker, limit=args.limit)

        labeler = PointInTimeLabeler(store=replay.store)
        label_summary = labeler.materialize_labels()
        summary = None
        if not args.skip_eval:
            evaluator = WalkForwardEvaluator(store=replay.store)
            summary = evaluator.evaluate()
        finished_at = utc_now_iso()
        record_pipeline_run(
            store=store,
            job_name="backfill_vnext",
            status="success",
            started_at=started_at,
            finished_at=finished_at,
            metrics={
                "replayed_snapshots": replay_summary.replayed_snapshots,
                "snapshot_label_rows": label_summary.snapshot_label_rows,
                "matured_return_90d_rows": label_summary.matured_return_90d_rows,
                "walkforward_windows": summary.num_windows if summary else 0,
            },
            config={"ticker": args.ticker, "limit": args.limit, "skip_eval": args.skip_eval, "store_dir": settings.store_dir},
        )
    except Exception as exc:
        finished_at = utc_now_iso()
        record_pipeline_run(
            store=store,
            job_name="backfill_vnext",
            status="failed",
            started_at=started_at,
            finished_at=finished_at,
            metrics={},
            config={"ticker": args.ticker, "limit": args.limit, "skip_eval": args.skip_eval, "store_dir": settings.store_dir},
            notes=f"{type(exc).__name__}: {exc}",
        )
        raise

    print("=" * 72)
    print("  TATETUCK BOT vNEXT — HISTORICAL BACKFILL")
    print("=" * 72)
    print(f"replayed_snapshots:        {replay_summary.replayed_snapshots}")
    print(f"replayed_tickers:          {replay_summary.replayed_tickers}")
    print(f"feature_rows_written:      {replay_summary.feature_rows_written}")
    print(f"prediction_rows_written:   {replay_summary.prediction_rows_written}")
    print(f"earliest_as_of:            {replay_summary.earliest_as_of}")
    print(f"latest_as_of:              {replay_summary.latest_as_of}")
    print(f"snapshot_label_rows:       {label_summary.snapshot_label_rows}")
    print(f"event_label_rows:          {label_summary.event_label_rows}")
    print(f"matured_return_90d_rows:   {label_summary.matured_return_90d_rows}")
    print(f"matured_event_rows:        {label_summary.matured_event_rows}")

    if not args.skip_eval:
        print("\n[walk-forward]")
        print(f"rows:                     {summary.num_rows}")
        print(f"windows:                  {summary.num_windows}")
        print(f"rank_ic:                  {summary.rank_ic:.4f}")
        print(f"hit_rate:                 {summary.hit_rate:.4f}")
        print(f"top_bottom_spread:        {summary.top_bottom_spread:.4f}")
        print(f"turnover:                 {summary.turnover:.4f}")
        print(f"max_drawdown:             {summary.max_drawdown:.4f}")
        print(f"beta_adj_return:          {summary.beta_adjusted_return:.4f}")
        print(f"brier:                    {summary.calibrated_brier:.4f}")
        print(f"leakage_passed:           {summary.leakage_passed}")
        if summary.message:
            print(f"message:                  {summary.message}")

    print(f"\n[summary] elapsed_seconds={time.time() - start:.1f}")


if __name__ == "__main__":
    main()
