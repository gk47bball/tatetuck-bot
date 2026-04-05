"""
refresh_research_book_vnext.py — rebuild the PM research book without touching broker state.

Uses the latest archived snapshot per ticker by default so the dashboard can be
refreshed locally even when live data ingestion is unavailable.
"""

from __future__ import annotations

import argparse
import time

from biopharma_agent.vnext import TatetuckPlatform, UniverseResolver, VNextSettings, record_pipeline_run
from biopharma_agent.vnext.ops import utc_now_iso
from biopharma_agent.vnext.storage import LocalResearchStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh Tatetuck vNext research artifacts for the PM dashboard.")
    parser.add_argument("--ticker", help="Refresh a single ticker.")
    parser.add_argument("--company-name", help="Optional company name when using --ticker.")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of universe names refreshed.")
    parser.add_argument(
        "--prefer-live",
        action="store_true",
        help="Prefer live ingestion instead of the latest archived snapshot when both are available.",
    )
    parser.add_argument(
        "--include-literature",
        action="store_true",
        help="Include literature synthesis during refresh runs.",
    )
    return parser.parse_args()


def resolve_universe(args: argparse.Namespace, store: LocalResearchStore) -> list[tuple[str, str]]:
    if args.ticker:
        ticker = args.ticker.upper()
        return [(ticker, args.company_name or ticker)]
    resolver = UniverseResolver(store=store)
    return resolver.resolve_default_universe(limit=args.limit, prefer_archive=not args.prefer_live)


def main() -> None:
    args = parse_args()
    settings = VNextSettings.from_env()
    store = LocalResearchStore(settings.store_dir)
    platform = TatetuckPlatform(store=store)
    universe = resolve_universe(args, store=store)
    started_at = utc_now_iso()
    start = time.time()

    analyses = platform.analyze_universe(
        universe,
        include_literature=args.include_literature or settings.include_literature,
        prefer_archive=not args.prefer_live,
        fallback_to_archive=True,
        persist=True,
    )

    finished_at = utc_now_iso()
    latest_as_of = max((item.signal.as_of for item in analyses), default=None)
    avg_confidence = (
        sum(float(item.signal.confidence) for item in analyses) / len(analyses)
        if analyses
        else 0.0
    )
    record_pipeline_run(
        store=store,
        job_name="refresh_research_book_vnext",
        status="success",
        started_at=started_at,
        finished_at=finished_at,
        metrics={
            "analyses": len(analyses),
            "long_or_watch_names": len([item for item in analyses if item.portfolio.stance in {"long", "avoid"}]),
            "short_names": len([item for item in analyses if item.portfolio.stance == "short"]),
            "avg_confidence": round(avg_confidence, 4),
        },
        config={
            "prefer_live": bool(args.prefer_live),
            "include_literature": bool(args.include_literature or settings.include_literature),
            "limit": int(args.limit),
            "ticker": args.ticker.upper() if args.ticker else None,
            "store_dir": settings.store_dir,
        },
        notes=f"latest_signal_as_of={latest_as_of}",
    )

    print("=" * 72)
    print("  TATETUCK BOT vNEXT — RESEARCH BOOK REFRESH")
    print("=" * 72)
    print(f"names_refreshed:            {len(analyses)}")
    print(f"prefer_live:               {bool(args.prefer_live)}")
    print(f"latest_signal_as_of:       {latest_as_of or 'none'}")
    print(f"avg_confidence:            {avg_confidence:.3f}")
    print(f"elapsed_seconds:           {time.time() - start:.1f}")
    print("\n[top refreshed names]")
    for item in sorted(analyses, key=lambda analysis: analysis.signal.confidence, reverse=True)[:10]:
        print(
            f"- {item.snapshot.ticker:5} | "
            f"{item.portfolio.stance:5} | "
            f"conf={item.signal.confidence:.3f} | "
            f"ret={item.signal.expected_return:+.3f} | "
            f"as_of={item.signal.as_of}"
        )


if __name__ == "__main__":
    main()
