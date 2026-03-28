from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import tempfile
import unittest

from biopharma_agent.vnext.audit import ResearchAuditBuilder
from biopharma_agent.vnext.features import FeatureEngineer
from biopharma_agent.vnext.graph import build_company_snapshot
from biopharma_agent.vnext.models import EventDrivenEnsemble
from biopharma_agent.vnext.portfolio import PortfolioConstructor, aggregate_signal
from biopharma_agent.vnext.storage import LocalResearchStore


def make_raw(
    ticker: str,
    *,
    revenue: float,
    market_cap: float,
    cash: float,
    debt: float,
    best_phase: str,
    condition: str,
    intervention: str,
    description: str,
) -> dict:
    phase_map = {
        "PHASE3": ["Phase 3"],
        "PHASE2": ["Phase 2"],
        "PHASE1": ["Phase 1"],
        "APPROVED": ["Phase 3"],
    }
    return {
        "ticker": ticker,
        "company_name": f"{ticker} Therapeutics",
        "finance": {
            "marketCap": market_cap,
            "enterpriseValue": max(market_cap - cash + debt, 0.0),
            "totalRevenue": revenue,
            "cash": cash,
            "debt": debt,
            "momentum_3mo": 0.15,
            "trailing_6mo_return": 0.05,
            "volatility": 0.04,
            "description": description,
        },
        "trials": [
            {
                "nct_id": f"NCT-{ticker}-1",
                "title": f"{intervention} study",
                "overall_status": "RECRUITING",
                "phase": phase_map[best_phase],
                "conditions": [condition],
                "interventions": [intervention],
                "primary_outcomes": ["overall response"],
                "enrollment": 180,
            }
        ],
        "num_trials": 1,
        "best_phase": best_phase,
        "num_papers": 3,
        "pubmed_papers": [{"pmid": f"{ticker}-1", "title": f"{ticker} paper", "abstract": "Supporting evidence."}],
        "conditions": [condition],
    }


def write_research_state(store: LocalResearchStore, raw: dict, as_of: datetime) -> None:
    snapshot = build_company_snapshot(raw, as_of=as_of)
    snapshot.metadata["sec_cik"] = f"0000{raw['ticker']}"
    raw_key = f"{snapshot.ticker}_{snapshot.as_of.replace(':', '-')}"
    store.write_raw_payload("snapshots", raw_key, asdict(snapshot))
    store.write_snapshot(snapshot)

    vectors = FeatureEngineer().build_all(snapshot)
    store.write_feature_vectors(vectors)
    predictions = EventDrivenEnsemble(store=store).score([item for item in vectors if not item.metadata.get("aggregate")])
    signal = aggregate_signal(
        ticker=snapshot.ticker,
        as_of=snapshot.as_of,
        predictions=predictions,
        evidence_rationale=["test"],
        evidence=snapshot.evidence[:3],
    )
    recommendation = PortfolioConstructor().recommend(signal)
    store.write_signal_artifacts([signal])
    store.write_portfolio_recommendations([recommendation])


class TestVNextAudit(unittest.TestCase):
    def test_audit_builds_pm_grade_summary(self):
        as_of = datetime(2025, 1, 15, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            write_research_state(
                store,
                make_raw(
                    "CRSP",
                    revenue=3_500_000,
                    market_cap=4_000_000_000,
                    cash=1_900_000_000,
                    debt=0.0,
                    best_phase="PHASE3",
                    condition="Sickle Cell Disease",
                    intervention="CASGEVY",
                    description="Gene editing platform with late-stage hematology catalysts.",
                ),
                as_of,
            )
            write_research_state(
                store,
                make_raw(
                    "NVAX",
                    revenue=1_100_000_000,
                    market_cap=1_400_000_000,
                    cash=900_000_000,
                    debt=500_000_000,
                    best_phase="PHASE2",
                    condition="COVID-19",
                    intervention="NVX-CoV",
                    description="Commercial vaccine franchise with earnings and pipeline updates.",
                ),
                as_of,
            )
            write_research_state(
                store,
                make_raw(
                    "PRME",
                    revenue=0.0,
                    market_cap=650_000_000,
                    cash=220_000_000,
                    debt=0.0,
                    best_phase="PHASE1",
                    condition="Muscular Dystrophy",
                    intervention="Prime Editor 1",
                    description="Prime editing platform in early clinical development.",
                ),
                as_of,
            )

            audit = ResearchAuditBuilder(store=store).build(top_n=5)

            self.assertEqual(audit.latest_ticker_count, 3)
            self.assertEqual(audit.signal_ticker_count, 3)
            self.assertTrue(audit.top_ideas)
            self.assertGreaterEqual(audit.scenario_diversity, 1)
            self.assertGreaterEqual(audit.archetype_diversity, 2)
            self.assertGreater(audit.expected_return_std, 0.0)
            self.assertFalse(store.read_table("research_audits").empty)

    def test_audit_flags_collapsed_signal_dispersion(self):
        as_of = "2025-01-15T00:00:00+00:00"
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            for ticker in ("AAA", "BBB"):
                snapshot = build_company_snapshot(
                    make_raw(
                        ticker,
                        revenue=100_000_000,
                        market_cap=1_000_000_000,
                        cash=300_000_000,
                        debt=0.0,
                        best_phase="PHASE2",
                        condition="Renal Cell Carcinoma",
                        intervention=f"{ticker}-101",
                        description="Oncology company with catalysts.",
                    ),
                    as_of=datetime(2025, 1, 15, tzinfo=timezone.utc),
                )
                snapshot.metadata["sec_cik"] = f"0000{ticker}"
                raw_key = f"{snapshot.ticker}_{snapshot.as_of.replace(':', '-')}"
                store.write_raw_payload("snapshots", raw_key, asdict(snapshot))
                store.write_snapshot(snapshot)
                store.write_signal_artifacts(
                    [
                        aggregate_signal(
                            ticker=snapshot.ticker,
                            as_of=snapshot.as_of,
                            predictions=[],
                            evidence_rationale=["flat"],
                            evidence=snapshot.evidence,
                        )
                    ]
                )
                store.write_portfolio_recommendations(
                    [
                        PortfolioConstructor().recommend(
                            aggregate_signal(
                                ticker=snapshot.ticker,
                                as_of=snapshot.as_of,
                                predictions=[],
                                evidence_rationale=["flat"],
                                evidence=snapshot.evidence,
                            )
                        )
                    ]
                )

            audit = ResearchAuditBuilder(store=store).build(top_n=5)
            blocker_text = " ".join(audit.blockers)
            self.assertIn("Expected-return dispersion is too low", blocker_text)
            self.assertIn("Portfolio target-weight dispersion is too low", blocker_text)


if __name__ == "__main__":
    unittest.main()
