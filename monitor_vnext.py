"""
monitor_vnext.py — lightweight autonomy loop for Tatetuck vNext.

Watches the live book, recent exact catalyst tape, and autonomy health to
trigger targeted single-name refreshes and optional paper-trade actions without
waiting for a full manual PM run.
"""

from __future__ import annotations

import argparse

from biopharma_agent.vnext.monitor import AutonomyMonitor
from biopharma_agent.vnext.settings import VNextSettings
from biopharma_agent.vnext.storage import LocalResearchStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Tatetuck vNext autonomy monitor.")
    parser.add_argument(
        "--ticker",
        action="append",
        dest="tickers",
        help="Force the monitor to refresh a specific ticker. Can be passed multiple times.",
    )
    parser.add_argument("--submit", action="store_true", help="Submit actionable paper orders.")
    parser.add_argument(
        "--allow-blocked-readiness",
        action="store_true",
        help="Allow the monitor to submit even when readiness blockers remain.",
    )
    parser.add_argument(
        "--include-literature",
        action="store_true",
        help="Include literature synthesis in triggered refreshes.",
    )
    parser.add_argument(
        "--prefer-archive",
        action="store_true",
        help="Prefer archived snapshots instead of live ingestion during triggered refreshes.",
    )
    parser.add_argument("--loop", action="store_true", help="Run continuously instead of a single cycle.")
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=0,
        help="Loop sleep interval. Defaults to the monitor setting if omitted.",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=0,
        help="Cap the number of symbols refreshed in a single cycle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = VNextSettings.from_env()
    store = LocalResearchStore(settings.store_dir)
    monitor = AutonomyMonitor(settings=settings, store=store)

    if args.loop:
        monitor.run_forever(
            interval_seconds=args.interval_seconds or settings.monitor_loop_interval_seconds,
            submit=bool(args.submit),
            allow_blocked_readiness=bool(args.allow_blocked_readiness),
            include_literature=bool(args.include_literature or settings.include_literature),
            prefer_live=not bool(args.prefer_archive),
        )
        return

    result = monitor.run_once(
        submit=bool(args.submit),
        allow_blocked_readiness=bool(args.allow_blocked_readiness),
        include_literature=bool(args.include_literature or settings.include_literature),
        prefer_live=not bool(args.prefer_archive),
        manual_symbols=[ticker.upper() for ticker in (args.tickers or [])],
        max_symbols=args.max_symbols or None,
    )

    print("=" * 72)
    print("  TATETUCK BOT vNEXT — AUTONOMY MONITOR")
    print("=" * 72)
    print(f"generated_at:              {result.generated_at}")
    print(f"status:                    {result.status}")
    print(f"trigger_count:             {result.trigger_count}")
    print(f"refreshed_symbols:         {', '.join(result.refreshed_symbols) if result.refreshed_symbols else 'none'}")
    print(f"actionable_orders:         {result.actionable_orders}")
    print(f"submitted_orders:          {result.submitted_orders}")
    print(f"note:                      {result.note or 'none'}")

    print("\n[blockers]")
    if result.blockers:
        for blocker in result.blockers:
            print(f"- {blocker}")
    else:
        print("- none")

    print("\n[warnings]")
    if result.warnings:
        for warning in result.warnings:
            print(f"- {warning}")
    else:
        print("- none")


if __name__ == "__main__":
    main()
