"""
evaluate_vnext.py — Event-driven Tatetuck platform evaluator.

Builds current company snapshots into the local research store, then runs the
new walk-forward evaluator when enough historical snapshots exist.
"""

import argparse
import time

from biopharma_agent.vnext import TatetuckPlatform, VNextSettings, record_pipeline_run
from biopharma_agent.vnext.evaluation import WalkForwardEvaluator
from biopharma_agent.vnext.ops import utc_now_iso
from biopharma_agent.vnext.storage import LocalResearchStore
from biopharma_agent.vnext.universe import UniverseResolver


def parse_args():
    parser = argparse.ArgumentParser(description="Run the vNext event-driven evaluator.")
    parser.add_argument(
        "--prefer-archive",
        action="store_true",
        help="Use archived snapshots when available instead of refreshing every ticker live first.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 72)
    print("  TATETUCK BOT vNEXT — EVENT-DRIVEN BIOPHARMA ALPHA PLATFORM")
    print("=" * 72)
    settings = VNextSettings.from_env()
    store = LocalResearchStore(settings.store_dir)
    started_at = utc_now_iso()
    start = time.time()

    try:
        platform = TatetuckPlatform(store=store)
        universe = UniverseResolver(store=store).resolve_default_universe(prefer_archive=args.prefer_archive)
        print("\n[ingest] Building company-program-catalyst snapshots...")
        analyses = platform.analyze_universe(
            universe,
            include_literature=settings.include_literature,
            prefer_archive=args.prefer_archive,
            fallback_to_archive=True,
        )

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

        evaluator = WalkForwardEvaluator(store=platform.store, settings=settings)
        summary = evaluator.evaluate()
        from_failure_universe_rate = evaluator._from_failure_universe_rate
        store.write_raw_payload(
            "validation_audits",
            "latest_walkforward_audit",
            {
                "generated_at": utc_now_iso(),
                "rows": summary.num_rows,
                "windows": summary.num_windows,
                "rank_ic": summary.rank_ic,
                "strict_rank_ic": summary.strict_rank_ic,
                "pm_context_coverage": summary.pm_context_coverage,
                "exact_primary_event_rate": summary.exact_primary_event_rate,
                "synthetic_primary_event_rate": summary.synthetic_primary_event_rate,
                "institutional_blockers": summary.institutional_blockers,
                "company_state_scorecards": summary.company_state_scorecards,
                "setup_type_scorecards": summary.setup_type_scorecards,
                "state_setup_scorecards": summary.state_setup_scorecards,
                "factor_attribution": summary.factor_attribution,
                "momentum_ablation": summary.momentum_ablation,
                "latest_window_top_trades": summary.latest_window_top_trades,
            },
        )
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
                "strict_rank_ic": summary.strict_rank_ic,
                "hit_rate": summary.hit_rate,
                "strict_hit_rate": summary.strict_hit_rate,
                "top_bottom_spread": summary.top_bottom_spread,
                "strict_top_bottom_spread": summary.strict_top_bottom_spread,
                "turnover": summary.turnover,
                "brier": summary.calibrated_brier,
                "leakage_passed": summary.leakage_passed,
                "pm_context_coverage": summary.pm_context_coverage,
                "exact_primary_event_rate": summary.exact_primary_event_rate,
                "synthetic_primary_event_rate": summary.synthetic_primary_event_rate,
                "institutional_blockers": summary.institutional_blockers,
                "event_type_scorecards": summary.event_type_scorecards,
                "company_state_scorecards": summary.company_state_scorecards,
                "setup_type_scorecards": summary.setup_type_scorecards,
                "state_setup_scorecards": summary.state_setup_scorecards,
                "factor_attribution": summary.factor_attribution,
                "momentum_ablation": summary.momentum_ablation,
                "from_failure_universe_rate": from_failure_universe_rate,
            },
            config={
                "store_dir": settings.store_dir,
                "include_literature": settings.include_literature,
                "prefer_archive": args.prefer_archive,
                "universe_size": len(universe),
            },
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
            config={
                "store_dir": settings.store_dir,
                "include_literature": settings.include_literature,
                "prefer_archive": args.prefer_archive,
            },
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
    print(f"strict_rank_ic:   {summary.strict_rank_ic:.4f}")
    print(f"strict_hit_rate:  {summary.strict_hit_rate:.4f}")
    print(f"strict_spread:    {summary.strict_top_bottom_spread:.4f}")
    print(f"pm_context_cov:   {summary.pm_context_coverage:.4f}")
    print(f"exact_event_rate: {summary.exact_primary_event_rate:.4f}")
    print(f"synthetic_rate:   {summary.synthetic_primary_event_rate:.4f}")
    if summary.momentum_ablation:
        print(f"momentum_only_ic: {summary.momentum_ablation.get('momentum_only_rank_ic', 0.0):.4f}")
        print(f"no_momentum_ic:   {summary.momentum_ablation.get('no_momentum_rank_ic', 0.0):.4f}")
    if summary.message:
        print(f"message:          {summary.message}")
    print("\n[institutional_blockers]")
    if summary.institutional_blockers:
        for blocker in summary.institutional_blockers:
            print(f"- {blocker}")
    else:
        print("- none")
    if summary.event_type_scorecards:
        print("\n[event_scorecards]")
        for event_type, metrics in sorted(
            summary.event_type_scorecards.items(),
            key=lambda item: item[1].get("top_bottom_spread", 0.0),
            reverse=True,
        ):
            print(
                f"{event_type:18} | "
                f"bucket={metrics.get('event_bucket', 'other'):10} | "
                f"rows={int(metrics.get('rows', 0)):>4} | "
                f"hit={metrics.get('hit_rate', 0.0):.3f} | "
                f"spread={metrics.get('top_bottom_spread', 0.0):+.3f} | "
                f"brier={metrics.get('calibrated_brier', 1.0):.3f}"
            )
    if summary.setup_type_scorecards:
        print("\n[setup_scorecards]")
        for setup_type, metrics in sorted(
            summary.setup_type_scorecards.items(),
            key=lambda item: item[1].get("top_bottom_spread", 0.0),
            reverse=True,
        ):
            print(
                f"{setup_type:28} | "
                f"rows={int(metrics.get('rows', 0)):>4} | "
                f"hit={metrics.get('hit_rate', 0.0):.3f} | "
                f"spread={metrics.get('top_bottom_spread', 0.0):+.3f} | "
                f"ic={metrics.get('rank_ic', 0.0):+.3f}"
            )
    if summary.company_state_scorecards:
        print("\n[state_scorecards]")
        for company_state, metrics in sorted(
            summary.company_state_scorecards.items(),
            key=lambda item: item[1].get("top_bottom_spread", 0.0),
            reverse=True,
        ):
            print(
                f"{company_state:28} | "
                f"rows={int(metrics.get('rows', 0)):>4} | "
                f"hit={metrics.get('hit_rate', 0.0):.3f} | "
                f"spread={metrics.get('top_bottom_spread', 0.0):+.3f} | "
                f"ic={metrics.get('rank_ic', 0.0):+.3f}"
            )
    if summary.factor_attribution:
        print("\n[factor_attribution]")
        for family, metrics in sorted(
            summary.factor_attribution.items(),
            key=lambda item: item[1].get("rank_ic_delta", 0.0),
            reverse=True,
        ):
            print(
                f"{family:20} | "
                f"ic_delta={metrics.get('rank_ic_delta', 0.0):+.3f} | "
                f"spread_delta={metrics.get('spread_delta', 0.0):+.3f} | "
                f"signal_corr={metrics.get('signal_correlation', 0.0):.3f}"
            )
    if summary.latest_window_top_trades:
        print("\n[latest_window_audit]")
        for row in summary.latest_window_top_trades[:10]:
            event_date = row.get("primary_event_date") or "TBD"
            event_status = row.get("primary_event_status") or "unknown"
            print(
                f"{str(row.get('ticker')):5} | "
                f"wt={float(row.get('target_weight', 0.0)):>4.1f}% | "
                f"state={row.get('company_state') or 'na'} | "
                f"setup={row.get('setup_type') or 'na'} | "
                f"event={row.get('primary_event_type') or 'none'} | "
                f"date={event_date} | "
                f"status={event_status} | "
                f"synthetic={bool(row.get('primary_event_synthetic', False))}"
            )

    print(f"\n[summary] elapsed_seconds={time.time() - start:.1f}")


if __name__ == "__main__":
    main()
