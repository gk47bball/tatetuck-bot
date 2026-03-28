"""
evaluate_vnext.py — Event-driven Tatetuck platform evaluator.

Builds current company snapshots into the local research store, then runs the
new walk-forward evaluator when enough historical snapshots exist.
"""

import time

import prepare

from biopharma_agent.vnext import TatetuckPlatform, VNextSettings, record_pipeline_run
from biopharma_agent.vnext.evaluation import WalkForwardEvaluator
from biopharma_agent.vnext.ops import utc_now_iso
from biopharma_agent.vnext.storage import LocalResearchStore


def main():
    print("=" * 72)
    print("  TATETUCK BOT vNEXT — EVENT-DRIVEN BIOPHARMA ALPHA PLATFORM")
    print("=" * 72)
    settings = VNextSettings.from_env()
    store = LocalResearchStore(settings.store_dir)
    started_at = utc_now_iso()
    start = time.time()

    try:
        platform = TatetuckPlatform(store=store)
        print("\n[ingest] Building company-program-catalyst snapshots...")
        analyses = platform.analyze_universe(prepare.BENCHMARK_TICKERS, include_literature=settings.include_literature)

        print("\n[signals] Current top event-driven ideas")
        ranked = sorted(analyses, key=lambda item: item.portfolio.target_weight, reverse=True)
        for analysis in ranked[:10]:
            print(
                f"{analysis.snapshot.ticker:5} | "
                f"wt={analysis.portfolio.target_weight:>4.1f}% | "
                f"exp_ret={analysis.signal.expected_return:+.3f} | "
                f"cat_prob={analysis.signal.catalyst_success_prob:.3f} | "
                f"scenario={analysis.portfolio.scenario}"
            )

        evaluator = WalkForwardEvaluator(store=platform.store)
        summary = evaluator.evaluate()
        finished_at = utc_now_iso()
        record_pipeline_run(
            store=store,
            job_name="evaluate_vnext",
            status="success",
            started_at=started_at,
            finished_at=finished_at,
            metrics={
                "rows": summary.num_rows,
                "windows": summary.num_windows,
                "rank_ic": summary.rank_ic,
                "hit_rate": summary.hit_rate,
                "top_bottom_spread": summary.top_bottom_spread,
                "leakage_passed": summary.leakage_passed,
            },
            config={"store_dir": settings.store_dir, "include_literature": settings.include_literature},
        )
    except Exception as exc:
        finished_at = utc_now_iso()
        record_pipeline_run(
            store=store,
            job_name="evaluate_vnext",
            status="failed",
            started_at=started_at,
            finished_at=finished_at,
            metrics={},
            config={"store_dir": settings.store_dir, "include_literature": settings.include_literature},
            notes=f"{type(exc).__name__}: {exc}",
        )
        raise

    print("\n[walk-forward] Primary evaluator")
    print(f"rows:             {summary.num_rows}")
    print(f"windows:          {summary.num_windows}")
    print(f"rank_ic:          {summary.rank_ic:.4f}")
    print(f"hit_rate:         {summary.hit_rate:.4f}")
    print(f"top_bottom_spread:{summary.top_bottom_spread:.4f}")
    print(f"turnover:         {summary.turnover:.4f}")
    print(f"max_drawdown:     {summary.max_drawdown:.4f}")
    print(f"beta_adj_return:  {summary.beta_adjusted_return:.4f}")
    print(f"brier:            {summary.calibrated_brier:.4f}")
    print(f"leakage_passed:   {summary.leakage_passed}")
    if summary.message:
        print(f"message:          {summary.message}")

    print(f"\n[summary] elapsed_seconds={time.time() - start:.1f}")


if __name__ == "__main__":
    main()
