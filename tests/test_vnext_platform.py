import tempfile
import unittest
from dataclasses import asdict
from unittest.mock import patch

import pandas as pd

from biopharma_agent.vnext import TatetuckPlatform, archive_universe
from biopharma_agent.vnext.entities import SignalArtifact
from biopharma_agent.vnext.evaluation import WalkForwardEvaluator
from biopharma_agent.vnext.features import FeatureEngineer
from biopharma_agent.vnext.graph import build_company_snapshot
from biopharma_agent.vnext.market_profile import build_expectation_lens, classify_company_state
from biopharma_agent.vnext.portfolio import PortfolioConstructor, aggregate_signal
from biopharma_agent.vnext.sources import IngestionService, enrich_snapshot_with_external_data
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
        self.assertFalse(snapshot.approved_products)
        self.assertTrue(snapshot.metadata["commercial_revenue_present"])
        self.assertTrue(any(event.event_type == "commercial_update" for event in snapshot.catalyst_events))

    def test_graph_builder_uses_registry_for_known_products(self):
        raw = make_raw_company()
        raw["ticker"] = "CRSP"
        raw["company_name"] = "CRISPR Therapeutics"
        raw["finance"]["totalRevenue"] = 3_500_000
        snapshot = build_company_snapshot(raw)
        self.assertTrue(snapshot.approved_products)
        self.assertEqual(snapshot.approved_products[0].name, "CASGEVY")

    def test_market_profile_classifies_company_states(self):
        pre = build_company_snapshot(make_raw_company() | {"finance": {**make_raw_company()["finance"], "totalRevenue": 0.0}})
        launch_raw = make_raw_company()
        launch_raw["ticker"] = "CRSP"
        launch_raw["company_name"] = "CRISPR Therapeutics"
        launch_raw["finance"]["totalRevenue"] = 50_000_000
        launch = build_company_snapshot(launch_raw)
        mature_raw = make_raw_company()
        mature_raw["ticker"] = "MRNA"
        mature_raw["company_name"] = "Moderna"
        mature_raw["finance"]["totalRevenue"] = 1_500_000_000
        mature = build_company_snapshot(mature_raw)

        self.assertEqual(classify_company_state(pre), "pre_commercial")
        self.assertEqual(classify_company_state(launch), "commercial_launch")
        self.assertEqual(classify_company_state(mature), "commercialized")

    def test_feature_engineer_outputs_non_leaky_vectors(self):
        snapshot = build_company_snapshot(make_raw_company())
        snapshot.metadata.update(
            {
                "sec_revenue_ttm": 120_000_000,
                "sec_operating_cashflow": 20_000_000,
                "last_10q_date": "2025-01-15",
                "recent_offering_signal": 1.0,
            }
        )
        engineer = FeatureEngineer()
        vectors = engineer.build_all(snapshot)
        self.assertTrue(vectors)
        program_vector = vectors[0]
        self.assertNotIn("market_flow_return_6mo_legacy", program_vector.feature_family)
        self.assertIn("catalyst_timing_probability", program_vector.feature_family)
        self.assertIn("commercial_execution_sec_revenue_scale", program_vector.feature_family)
        self.assertIn("balance_sheet_recent_offering_signal", program_vector.feature_family)
        self.assertGreaterEqual(program_vector.feature_family["catalyst_timing_filing_freshness_days"], 0.0)

    def test_external_enrichment_adds_sec_evidence_calendar_and_financing(self):
        snapshot = build_company_snapshot(make_raw_company())
        sec_payload = {
            "cik": "0000000123",
            "parsed": {
                "revenue_ttm": 150_000_000,
                "cash_latest": 600_000_000,
                "operating_cashflow": 30_000_000,
                "last_10q_date": "2025-02-01",
                "recent_filings": [
                    {
                        "form": "10-Q",
                        "filing_date": "2025-02-01",
                        "accession_number": "0001",
                        "primary_document": "q1.htm",
                        "url": "https://example.com/10q",
                    }
                ],
                "recent_offering_forms": [
                    {
                        "form": "S-3",
                        "filing_date": "2025-02-15",
                    }
                ],
            },
        }
        calendar_payload = {
            "events": [
                {
                    "event_type": "earnings",
                    "title": "TEST estimated quarterly update",
                    "expected_date": "2025-05-01",
                }
            ]
        }

        enriched = enrich_snapshot_with_external_data(snapshot, sec_payload, calendar_payload)
        self.assertEqual(enriched.metadata["sec_cik"], "0000000123")
        self.assertEqual(enriched.metadata["recent_offering_signal"], 1.0)
        self.assertTrue(any(item.source == "sec" for item in enriched.evidence))
        self.assertTrue(any(event.event_type == "earnings" for event in enriched.catalyst_events))
        self.assertTrue(any(event.event_type == "recent_offering_filing" for event in enriched.financing_events))

    def test_store_persists_company_level_catalysts(self):
        snapshot = build_company_snapshot(make_raw_company())
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            store.write_snapshot(snapshot)
            catalysts = store.read_table("catalysts")
            self.assertIn("phase3_readout", catalysts["event_type"].tolist())
            self.assertIn("commercial_update", catalysts["event_type"].tolist())

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

    def test_portfolio_constructor_expands_high_conviction_weights(self):
        constructor = PortfolioConstructor()
        high_signal = SignalArtifact(
            ticker="HIGH",
            as_of="2025-01-01T00:00:00+00:00",
            expected_return=0.24,
            catalyst_success_prob=0.72,
            confidence=0.84,
            crowding_risk=0.22,
            financing_risk=0.18,
            thesis_horizon="90d",
            primary_event_type="phase3_readout",
            primary_event_bucket="clinical",
            rationale=[],
            supporting_evidence=[],
        )
        medium_signal = SignalArtifact(
            ticker="MED",
            as_of="2025-01-01T00:00:00+00:00",
            expected_return=0.09,
            catalyst_success_prob=0.55,
            confidence=0.63,
            crowding_risk=0.35,
            financing_risk=0.30,
            thesis_horizon="90d",
            primary_event_type="phase2_readout",
            primary_event_bucket="clinical",
            rationale=[],
            supporting_evidence=[],
        )

        high_rec = constructor.recommend(high_signal)
        medium_rec = constructor.recommend(medium_signal)

        self.assertEqual(high_rec.scenario, "pre-catalyst long")
        self.assertGreater(high_rec.target_weight, medium_rec.target_weight)
        self.assertGreaterEqual(high_rec.target_weight, 3.0)

    def test_portfolio_constructor_smooths_small_weight_changes(self):
        constructor = PortfolioConstructor()
        previous_signal = SignalArtifact(
            ticker="STBL",
            as_of="2025-01-01T00:00:00+00:00",
            expected_return=0.18,
            catalyst_success_prob=0.68,
            confidence=0.78,
            crowding_risk=0.28,
            financing_risk=0.20,
            thesis_horizon="90d",
            primary_event_type="phase3_readout",
            primary_event_bucket="clinical",
            rationale=[],
            supporting_evidence=[],
        )
        current_signal = SignalArtifact(
            ticker="STBL",
            as_of="2025-02-01T00:00:00+00:00",
            expected_return=0.17,
            catalyst_success_prob=0.65,
            confidence=0.75,
            crowding_risk=0.29,
            financing_risk=0.21,
            thesis_horizon="90d",
            primary_event_type="phase3_readout",
            primary_event_bucket="clinical",
            rationale=[],
            supporting_evidence=[],
        )
        previous_recommendation = constructor.recommend(previous_signal)
        current_recommendation = constructor.recommend(
            current_signal,
            previous_recommendation=previous_recommendation,
            previous_signal=previous_signal,
        )

        self.assertGreater(previous_recommendation.target_weight, 0.0)
        self.assertLess(abs(current_recommendation.target_weight - previous_recommendation.target_weight), 1.0)

    def test_obesity_name_without_firm_event_is_treated_as_asymmetry_setup(self):
        raw = make_raw_company()
        raw["ticker"] = "VKTX"
        raw["company_name"] = "Viking Therapeutics"
        raw["finance"]["totalRevenue"] = 0.0
        raw["finance"]["description"] = "Obesity company with an oral and injectable metabolic pipeline."
        raw["trials"][0]["conditions"] = ["Obesity"]
        raw["trials"][0]["interventions"] = ["VK2735"]
        raw["trials"][1]["conditions"] = ["Obesity"]
        raw["trials"][1]["interventions"] = ["VK2809"]
        snapshot = build_company_snapshot(raw)
        signal = SignalArtifact(
            ticker="VKTX",
            as_of=snapshot.as_of,
            expected_return=0.18,
            catalyst_success_prob=0.60,
            confidence=0.70,
            crowding_risk=0.55,
            financing_risk=0.20,
            thesis_horizon="90d",
            primary_event_type="phase2_readout",
            primary_event_bucket="clinical",
            company_state="pre_commercial",
            rationale=[],
            supporting_evidence=[],
        )
        primary_event = snapshot.catalyst_events[0]
        lens = build_expectation_lens(snapshot, signal, primary_event, {"valuation_posture": "discounted"})

        self.assertEqual(lens["company_state"], "pre_commercial")
        self.assertEqual(lens["setup_type"], "asymmetry_without_near_term_catalyst")
        self.assertGreater(lens["competition_intensity"], 0.8)

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

    @patch("biopharma_agent.vnext.sources.fetch_legacy_snapshot")
    @patch("biopharma_agent.vnext.sources.CorporateCalendarClient.fetch_company_calendar")
    @patch("biopharma_agent.vnext.sources.SECXBRLClient.fetch_company_facts")
    def test_archive_universe_returns_store_summary(self, mock_sec, mock_calendar, mock_fetch):
        mock_fetch.return_value = make_raw_company()
        mock_sec.return_value = {
            "ticker": "TEST",
            "cik": "0000000123",
            "parsed": {
                "revenue_ttm": 120_000_000,
                "operating_cashflow": 25_000_000,
                "recent_filings": [],
                "recent_offering_forms": [],
            },
        }
        mock_calendar.return_value = {"ticker": "TEST", "events": []}
        with tempfile.TemporaryDirectory() as tmpdir:
            platform = TatetuckPlatform(store=LocalResearchStore(base_dir=tmpdir))
            analyses, summary = archive_universe(platform, [("TEST", "Test Therapeutics")])
            self.assertEqual(len(analyses), 1)
            self.assertEqual(summary.archived_companies, 1)
            self.assertEqual(summary.sec_enriched_companies, 1)
            self.assertGreaterEqual(summary.snapshot_rows, 1)
            self.assertGreaterEqual(summary.feature_rows, 1)
            self.assertGreaterEqual(summary.prediction_rows, 1)
            self.assertEqual(summary.top_ideas[0]["ticker"], "TEST")

    @patch("biopharma_agent.vnext.facade.IngestionService.ingest_company", side_effect=RuntimeError("offline"))
    def test_facade_falls_back_to_archived_snapshot(self, _mock_ingest):
        snapshot = build_company_snapshot(make_raw_company())
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            store.write_raw_payload("snapshots", "TEST_2025-01-01T00-00-00+00-00", asdict(snapshot))
            platform = TatetuckPlatform(store=store)
            analysis = platform.analyze_ticker("TEST", "Test Therapeutics", include_literature=False, fallback_to_archive=True)
            self.assertEqual(analysis.snapshot.ticker, "TEST")
            self.assertEqual(analysis.metadata["analysis_source"], "archive_fallback")
            self.assertIn("live_ingestion_error", analysis.snapshot.metadata)

    def test_facade_archive_read_only_does_not_persist(self):
        snapshot = build_company_snapshot(make_raw_company())
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            store.write_raw_payload("snapshots", "TEST_2025-01-01T00-00-00+00-00", asdict(snapshot))
            platform = TatetuckPlatform(store=store)
            before_rows = len(store.read_table("company_snapshots"))
            analysis = platform.analyze_ticker(
                "TEST",
                "Test Therapeutics",
                include_literature=False,
                prefer_archive=True,
                persist=False,
            )
            after_rows = len(store.read_table("company_snapshots"))
            self.assertEqual(analysis.metadata["analysis_source"], "archive")
            self.assertFalse(analysis.metadata["persisted"])
            self.assertIn("why_now", analysis.metadata)
            self.assertIn("valuation_summary", analysis.metadata)
            self.assertIn("kill_points", analysis.metadata)
            self.assertIn("company_state", analysis.metadata)
            self.assertIn("setup_type", analysis.metadata)
            self.assertIn("competitive_summary", analysis.metadata)
            self.assertIn("asymmetry_summary", analysis.metadata)
            self.assertEqual(before_rows, after_rows)

    @patch("biopharma_agent.vnext.sources.fetch_legacy_snapshot")
    @patch("biopharma_agent.vnext.sources.CorporateCalendarClient.fetch_company_calendar")
    @patch("biopharma_agent.vnext.sources.SECXBRLClient.fetch_company_facts")
    def test_ingestion_service_recovers_from_cached_payloads(self, mock_sec, mock_calendar, mock_fetch):
        sparse_raw = {
            "ticker": "TEST",
            "company_name": "Test Therapeutics",
            "finance": {"ticker": "TEST", "trailing_6mo_return": None, "momentum_3mo": None},
            "trials": [],
            "num_trials": 0,
            "num_total_trials": 0,
            "pubmed_papers": [],
            "num_papers": 0,
            "conditions": [],
        }
        mock_fetch.return_value = sparse_raw
        mock_sec.return_value = {"ticker": "TEST", "records": [], "source": "sec_xbrl", "status": "missing_cik"}
        mock_calendar.return_value = {"ticker": "TEST", "events": []}

        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            store.write_raw_payload("legacy_prepare", "TEST_2025-01-01T00-00-00+00-00", make_raw_company())
            store.write_raw_payload(
                "sec_xbrl",
                "TEST_2025-01-01T00-00-00+00-00",
                {
                    "ticker": "TEST",
                    "cik": "0000000123",
                    "parsed": {
                        "revenue_ttm": 120_000_000,
                        "operating_cashflow": 25_000_000,
                        "recent_filings": [],
                        "recent_offering_forms": [],
                    },
                },
            )
            store.write_raw_payload(
                "corp_calendar",
                "TEST_2025-01-01T00-00-00+00-00",
                {
                    "ticker": "TEST",
                    "events": [
                        {
                            "event_type": "earnings",
                            "title": "TEST estimated quarterly update",
                            "expected_date": "2025-05-01",
                        }
                    ],
                },
            )

            snapshot = IngestionService(store=store).ingest_company("TEST", "Test Therapeutics")
            self.assertTrue(snapshot.programs)
            self.assertEqual(snapshot.metadata["sec_cik"], "0000000123")
            self.assertEqual(
                snapshot.metadata["fallback_sources"],
                ["corp_calendar", "legacy_prepare", "sec_xbrl"],
            )


if __name__ == "__main__":
    unittest.main()
