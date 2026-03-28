"""
readiness_vnext.py — operational readiness audit for the vNext platform.
"""

from __future__ import annotations

from biopharma_agent.vnext.ops import build_readiness_report
from biopharma_agent.vnext.settings import VNextSettings
from biopharma_agent.vnext.storage import LocalResearchStore


def main() -> None:
    settings = VNextSettings.from_env()
    store = LocalResearchStore(settings.store_dir)
    report = build_readiness_report(store=store, settings=settings)

    print("=" * 72)
    print("  TATETUCK BOT vNEXT — READINESS AUDIT")
    print("=" * 72)
    print(f"status:                    {report.status}")
    print(f"generated_at:              {report.generated_at}")
    print(f"store_dir:                 {report.store_dir}")
    print(f"eodhd_configured:          {report.eodhd_configured}")
    print(f"sec_user_agent_configured: {report.sec_user_agent_configured}")
    print(f"snapshot_rows:             {report.snapshot_rows}")
    print(f"distinct_snapshot_dates:   {report.distinct_snapshot_dates}")
    print(f"latest_snapshot_age_hours: {report.latest_snapshot_age_hours}")
    print(f"label_rows:                {report.label_rows}")
    print(f"event_label_rows:          {report.event_label_rows}")
    print(f"matured_return_90d_rows:   {report.matured_return_90d_rows}")
    print(f"matured_event_rows:        {report.matured_event_rows}")
    print(f"archive_runs:              {report.successful_archive_runs}/{report.archive_run_count}")
    print(f"backfill_runs:             {report.successful_backfill_runs}/{report.backfill_run_count}")
    print(f"evaluate_runs:             {report.successful_evaluate_runs}/{report.evaluate_run_count}")
    print(f"eodhd_cache_files:         {report.eodhd_cache_files}")
    print(f"walkforward_rows:          {report.walkforward_rows}")
    print(f"walkforward_windows:       {report.walkforward_windows}")
    print(f"leakage_passed:            {report.leakage_passed}")
    print(f"evaluation_message:        {report.evaluation_message}")

    print("\n[blockers]")
    if report.blockers:
        for blocker in report.blockers:
            print(f"- {blocker}")
    else:
        print("- none")

    print("\n[warnings]")
    if report.warnings:
        for warning in report.warnings:
            print(f"- {warning}")
    else:
        print("- none")


if __name__ == "__main__":
    main()
