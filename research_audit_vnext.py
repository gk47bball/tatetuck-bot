"""
research_audit_vnext.py — PM-grade research sanity report for Tatetuck vNext.

Reads the latest archived universe and reports whether coverage, dispersion, and
idea quality are strong enough to trust the research stack operationally today.
"""

from __future__ import annotations

import argparse

from biopharma_agent.vnext import ResearchAuditBuilder, VNextSettings
from biopharma_agent.vnext.storage import LocalResearchStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Tatetuck vNext research audit.")
    parser.add_argument("--top", type=int, default=10, help="How many top ideas to print.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Recompute company-level signals and recommendations from the latest archived snapshots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = VNextSettings.from_env()
    store = LocalResearchStore(settings.store_dir)
    audit = ResearchAuditBuilder(store=store).build(top_n=args.top, refresh_company_views=args.refresh)

    print("=" * 72)
    print("  TATETUCK BOT vNEXT — RESEARCH AUDIT")
    print("=" * 72)
    print(f"store_dir:                  {audit.store_dir}")
    print(f"latest_snapshot_at:         {audit.latest_snapshot_at}")
    print(f"latest_ticker_count:        {audit.latest_ticker_count}")
    print(f"signal_ticker_count:        {audit.signal_ticker_count}")
    print(f"recommendation_count:       {audit.recommendation_ticker_count}")
    print(f"sec_enriched_pct:           {audit.sec_enriched_pct:.1%}")
    print(f"catalyst_coverage_pct:      {audit.catalyst_coverage_pct:.1%}")
    print(f"near_term_catalyst_pct:     {audit.near_term_catalyst_pct:.1%}")
    print(f"approved_product_pct:       {audit.approved_product_pct:.1%}")
    print(f"revenue_positive_pct:       {audit.revenue_positive_pct:.1%}")
    print(f"financing_flagged_pct:      {audit.financing_flagged_pct:.1%}")
    print(f"zero_market_cap_pct:        {audit.zero_market_cap_pct:.1%}")
    print(f"expected_return_std:        {audit.expected_return_std:.4f}")
    print(f"confidence_std:             {audit.confidence_std:.4f}")
    print(f"target_weight_std:          {audit.target_weight_std:.4f}")
    print(f"actionable_count:           {audit.actionable_count}")
    print(f"scenario_diversity:         {audit.scenario_diversity}")
    print(f"archetype_diversity:        {audit.archetype_diversity}")

    print("\n[scenario_counts]")
    if audit.scenario_counts:
        for scenario, count in audit.scenario_counts.items():
            print(f"- {scenario}: {count}")
    else:
        print("- none")

    print("\n[archetype_counts]")
    if audit.archetype_counts:
        for archetype, count in audit.archetype_counts.items():
            print(f"- {archetype}: {count}")
    else:
        print("- none")

    print("\n[primary_event_types]")
    if audit.primary_event_type_counts:
        for event_type, count in audit.primary_event_type_counts.items():
            print(f"- {event_type}: {count}")
    else:
        print("- none")

    print("\n[blockers]")
    if audit.blockers:
        for blocker in audit.blockers:
            print(f"- {blocker}")
    else:
        print("- none")

    print("\n[warnings]")
    if audit.warnings:
        for warning in audit.warnings:
            print(f"- {warning}")
    else:
        print("- none")

    print("\n[top_ideas]")
    if audit.top_ideas:
        for idea in audit.top_ideas:
            print(
                f"- {idea['ticker']}: scenario={idea['scenario']} "
                f"weight={idea['target_weight']:.2f}% exp_ret={idea['expected_return']:.4f} "
                f"conf={idea['confidence']:.3f} archetype={idea['archetype']} "
                f"event={idea['primary_event_type']}"
            )
    else:
        print("- none")


if __name__ == "__main__":
    main()
