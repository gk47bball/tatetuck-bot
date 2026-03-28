"""
backfill_vnext.py — rebuild historical vNext tables and labels from archived snapshots.

This script does not hit live company ingestion endpoints. It replays the raw
snapshot archive already stored in `.tatetuck_store/raw/snapshots`, rebuilds
features and predictions, and materializes forward-return/event-window labels.
"""

from __future__ import annotations

import argparse
import time

from biopharma_agent.vnext import HistoricalReplayEngine, PointInTimeLabeler
from biopharma_agent.vnext.evaluation import WalkForwardEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill vNext tables and labels from archived raw snapshots.")
    parser.add_argument("--ticker", help="Restrict replay to a single ticker.")
    parser.add_argument("--limit", type=int, default=0, help="Only replay the most recent N archived snapshots.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip the walk-forward evaluation summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()

    replay = HistoricalReplayEngine()
    replay_summary = replay.rebuild_from_archived_snapshots(ticker=args.ticker, limit=args.limit)

    labeler = PointInTimeLabeler(store=replay.store)
    label_summary = labeler.materialize_labels()

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
        evaluator = WalkForwardEvaluator(store=replay.store)
        summary = evaluator.evaluate()
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
