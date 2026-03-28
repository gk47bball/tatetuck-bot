import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from biopharma_agent.vnext import TatetuckPlatform
from biopharma_agent.vnext.evaluation import WalkForwardEvaluator
from biopharma_agent.vnext.features import FeatureEngineer
from biopharma_agent.vnext.graph import build_company_snapshot
from biopharma_agent.vnext.portfolio import PortfolioConstructor, aggregate_signal
from biopharma_agent.vnext.storage import LocalResearchStore


def make_raw_company() -> dict:
    return {
        "ticker": "TEST",
        "company_name": "Test Therapeutics",
        "finance": {
            "marketCap": 1_500_000_000,
            "enterpriseValue": 1_250_000_000,
            "totalRevenue": 80_000_000,
            "cash": 500_000_000,
            "debt": 50_000_000,
            "momentum_3mo": 0.22,
            "trailing_6mo_return": 0.10,
            "volatility": 0.04,
            "description": "Precision oncology and gene editing company with a late-stage RCC program.",
        },
        "trials": [
            {
                "nct_id": "NCT-1",
                "title": "Lead RCC trial",
                "overall_status": "RECRUITING",
                "phase": ["Phase 3"],
                "conditions": ["Clear Cell Renal Cell Carcinoma"],
                "interventions": ["TTX-101"],
                "primary_outcomes": ["overall survival"],
                "enrollment": 420,
            },
            {
                "nct_id": "NCT-2",
                "title": "Expansion cohort",
                "overall_status": "ACTIVE_NOT_RECRUITING",
                "phase": ["Phase 2"],
                "conditions": ["Clear Cell Renal Cell Carcinoma"],
                "interventions": ["TTX-101"],
                "primary_outcomes": ["response rate"],
                "enrollment": 115,
            },
        ],
        "num_trials": 2,
        "best_phase": "PHASE3",
        "num_papers": 6,
        "pubmed_papers": [
            {"pmid": "1", "title": "TTX-101 data", "abstract": "Strong oncology signal."},
        ],
    }


class TestVNextPlatform(unittest.TestCase):
    def test_graph_builder_creates_programs_and_financing_events(self):
        raw = make_raw_company()
        snapshot = build_company_snapshot(raw)
        self.assertEqual(snapshot.ticker, "TEST")
        self.assertTrue(snapshot.programs)
        self.assertEqual(snapshot.programs[0].phase, "PHASE3")
        self.assertTrue(snapshot.catalyst_events)
        self.assertTrue(snapshot.approved_products)

    def test_feature_engineer_outputs_non_leaky_vectors(self):
        snapshot = build_company_snapshot(make_raw_company())
        engineer = FeatureEngineer()
        vectors = engineer.build_all(snapshot)
        self.assertTrue(vectors)
        program_vector = vectors[0]
        self.assertNotIn("market_flow_return_6mo_legacy", program_vector.feature_family)
        self.assertIn("catalyst_timing_probability", program_vector.feature_family)

    def test_portfolio_constructor_flags_financing_overhang(self):
        snapshot = build_company_snapshot(make_raw_company())
        engineer = FeatureEngineer()
        vectors = engineer.build_all(snapshot)
        prediction_rows = []
        for idx, vector in enumerate(vectors[:-1]):
            prediction_rows.append(
                {
                    "entity_id": vector.entity_id,
                    "ticker": vector.ticker,
                    "as_of": vector.as_of,
                    "expected_return": 0.20 if idx == 0 else 0.08,
                    "catalyst_success_prob": 0.70,
                    "confidence": 0.75,
                    "crowding_risk": 0.35,
                    "financing_risk": 0.85,
                    "thesis_horizon": vector.thesis_horizon,
                    "model_name": "test",
                    "model_version": "v1",
                    "metadata": vector.metadata,
                }
            )
        from biopharma_agent.vnext.entities import ModelPrediction

        predictions = [ModelPrediction(**row) for row in prediction_rows]
        signal = aggregate_signal(snapshot.ticker, snapshot.as_of, predictions, ["test"], snapshot.evidence)
        recommendation = PortfolioConstructor().recommend(signal)
        self.assertEqual(recommendation.scenario, "avoid due to financing")
        self.assertEqual(recommendation.target_weight, 0.0)

    def test_walk_forward_leakage_audit_rejects_legacy_feature(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            frame = pd.DataFrame(
                [
                    {
                        "entity_id": "A",
                        "ticker": "TEST",
                        "as_of": "2025-01-01T00:00:00",
                        "thesis_horizon": "90d",
                        "market_flow_return_6mo_legacy": 0.2,
                        "target_return_90d": 0.1,
                        "target_catalyst_success": 1,
                    }
                ]
            )
            frame.to_parquet(store.tables_dir / "feature_vectors.parquet", index=False)
            evaluator = WalkForwardEvaluator(store=store)
            self.assertFalse(evaluator.leakage_audit(frame))

    @patch("biopharma_agent.vnext.facade.AutoResearchAgent.generate_literature_review", return_value="Literature summary.")
    @patch("biopharma_agent.vnext.sources.fetch_legacy_snapshot")
    def test_facade_builds_legacy_report_shape(self, mock_fetch, _mock_lit):
        mock_fetch.return_value = make_raw_company()
        with tempfile.TemporaryDirectory() as tmpdir:
            platform = TatetuckPlatform(store=LocalResearchStore(base_dir=tmpdir))
            report = platform.build_legacy_report("TEST", "Test Therapeutics")
            self.assertIn("valuation", report)
            self.assertIn("signal_artifact", report)
            self.assertIn("portfolio_recommendation", report)
            self.assertEqual(report["ticker"], "TEST")


if __name__ == "__main__":
    unittest.main()
