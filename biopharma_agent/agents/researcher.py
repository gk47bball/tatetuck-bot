"""
BiopharmaResearcher compatibility layer.

The legacy CLI/reporting stack now delegates to the vNext event-driven Tatetuck
platform so old entrypoints can keep working while the new research core owns
ingestion, normalization, feature engineering, storage, and signal generation.
"""

from __future__ import annotations

from typing import Any

from ..vnext import TatetuckPlatform


class BiopharmaResearcher:
    def __init__(self):
        self.platform = TatetuckPlatform()

    def run_research(self, ticker: str, company_name: str | None = None) -> dict[str, Any]:
        print(f"\n{'='*72}")
        print(f"  TATETUCK vNEXT RESEARCH WORKSTATION: {ticker}")
        print(f"{'='*72}\n")
        report = self.platform.build_legacy_report(ticker, company_name=company_name)

        signal = report.get("signal_artifact", {})
        portfolio = report.get("portfolio_recommendation", {})
        print(f"[snapshot] {report['company']} ({report['ticker']})")
        print(f"  → Programs normalized: {report['company_snapshot']['metadata'].get('num_trials', 0)} active trials")
        print(f"  → Expected return:    {signal.get('expected_return', 0.0):+.3f}")
        print(f"  → Catalyst success:   {signal.get('catalyst_success_prob', 0.0):.3f}")
        print(f"  → Confidence:         {signal.get('confidence', 0.0):.3f}")
        print(f"  → Scenario:           {portfolio.get('scenario', 'watchlist only')}")
        print(f"  → Target weight:      {portfolio.get('target_weight', 0.0)}%")

        return report
