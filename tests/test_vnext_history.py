import json
import tempfile
import unittest
from dataclasses import asdict
from unittest.mock import Mock, patch

import pandas as pd

from biopharma_agent.vnext.entities import ModelPrediction
from biopharma_agent.vnext.evaluation import WalkForwardEvaluator
from biopharma_agent.vnext.graph import build_company_snapshot
from biopharma_agent.vnext.labels import EODHDHistoryProvider, PointInTimeLabeler
from biopharma_agent.vnext.models import EventDrivenEnsemble
from biopharma_agent.vnext.settings import VNextSettings
from biopharma_agent.vnext.replay import HistoricalReplayEngine
from biopharma_agent.vnext.storage import LocalResearchStore


class StubHistoryProvider:
    def __init__(self, close_by_ticker):
        self.close_by_ticker = close_by_ticker

    def load_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        frame = self.close_by_ticker.get(ticker, pd.DataFrame(columns=["close"]))
        if frame.empty:
            return frame
        mask = (frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))
        return frame.loc[mask]


def make_price_frame():
    dates = pd.date_range("2025-01-01", periods=260, freq="D")
    close = pd.Series([100.0 + day for day in range(len(dates))], index=dates)
    return pd.DataFrame({"close": close.values}, index=dates)


class TestVNextHistory(unittest.TestCase):
    def test_eodhd_provider_normalizes_symbol_and_uses_adjusted_close(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = [
            {"date": "2025-01-02", "close": 101.0, "adjusted_close": 100.5},
            {"date": "2025-01-03", "close": 102.0, "adjusted_close": 101.5},
        ]
        session = Mock()
        session.get.return_value = response

        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            provider = EODHDHistoryProvider(
                store=store,
                api_key="test-key",
                session=session,
            )
            frame = provider.load_history("CRSP", "2025-01-01", "2025-01-31")

            self.assertEqual(session.get.call_args.kwargs["params"]["from"], "2025-01-01")
            self.assertIn("CRSP.US", session.get.call_args.args[0])
            self.assertEqual(frame.iloc[0]["close"], 100.5)
            self.assertEqual(len(frame), 2)

    def test_eodhd_provider_reuses_cached_payload(self):
        session = Mock()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            store.write_raw_payload(
                "market_prices_eodhd",
                "CRSP_2025-01-01_2025-01-31",
                [{"date": "2025-01-02", "close": 99.5}],
            )
            provider = EODHDHistoryProvider(
                store=store,
                api_key="test-key",
                session=session,
            )
            frame = provider.load_history("CRSP", "2025-01-01", "2025-01-31")

            session.get.assert_not_called()
            self.assertEqual(frame.iloc[0]["close"], 99.5)

    def test_labeler_builds_forward_and_event_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            snapshots = pd.DataFrame(
                [
                    {"ticker": "TEST", "as_of": "2025-01-15T00:00:00"},
                    {"ticker": "TEST", "as_of": "2025-02-15T00:00:00"},
                ]
            )
            catalysts = pd.DataFrame(
                [
                    {
                        "ticker": "TEST",
                        "as_of": "2025-01-15T00:00:00",
                        "event_id": "TEST:event:1",
                        "event_type": "clinical_readout",
                        "expected_date": "2025-02-01",
                        "horizon_days": 17,
                        "importance": 0.8,
                    },
                    {
                        "ticker": "TEST",
                        "as_of": "2025-02-15T00:00:00",
                        "event_id": "TEST:event:2",
                        "event_type": "earnings",
                        "expected_date": "2025-03-01",
                        "horizon_days": 14,
                        "importance": 0.6,
                    },
                ]
            )
            labeler = PointInTimeLabeler(
                store=store,
                history_provider=StubHistoryProvider({"TEST": make_price_frame()}),
            )
            summary = labeler.materialize_labels(snapshots=snapshots, catalysts=catalysts)
            labels = store.read_table("labels")
            event_labels = store.read_table("event_labels")

            self.assertEqual(summary.snapshot_label_rows, 2)
            self.assertEqual(summary.event_label_rows, 2)
            self.assertIn("target_return_90d", labels.columns)
            self.assertIn("target_event_return_10d", labels.columns)
            self.assertTrue(labels["target_return_90d"].notna().all())
            self.assertTrue(event_labels["target_event_return_10d"].notna().all())

    def test_labeler_prefers_exact_events_over_synthetic_placeholders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            snapshots = pd.DataFrame([{"ticker": "TEST", "as_of": "2025-02-01T00:00:00"}])
            catalysts = pd.DataFrame(
                [
                    {
                        "ticker": "TEST",
                        "as_of": "2025-02-01T00:00:00",
                        "event_id": "TEST:synthetic:1",
                        "event_type": "phase3_readout",
                        "title": "TEST-101 next milestone",
                        "expected_date": "2025-05-01",
                        "horizon_days": 89,
                        "importance": 0.9,
                        "crowdedness": 0.2,
                        "status": "phase_timing_estimate",
                    },
                    {
                        "ticker": "TEST",
                        "as_of": "2025-02-01T00:00:00",
                        "event_id": "TEST:exact:1",
                        "event_type": "earnings",
                        "title": "TEST filed 10-Q financial update",
                        "expected_date": "2025-02-01T21:05:00",
                        "horizon_days": 0,
                        "importance": 0.55,
                        "crowdedness": 0.35,
                        "status": "exact_sec_filing",
                    },
                ]
            )
            labeler = PointInTimeLabeler(
                store=store,
                history_provider=StubHistoryProvider({"TEST": make_price_frame()}),
            )
            labels, event_labels = labeler.build_label_frames(snapshots=snapshots, catalysts=catalysts)

            self.assertEqual(labels.iloc[0]["target_primary_event_type"], "earnings")
            self.assertEqual(event_labels.iloc[0]["event_id"], "TEST:exact:1")

    def test_labeler_prioritizes_clinical_events_over_earnings_noise(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            snapshots = pd.DataFrame([{"ticker": "TEST", "as_of": "2025-01-15T00:00:00"}])
            catalysts = pd.DataFrame(
                [
                    {
                        "ticker": "TEST",
                        "as_of": "2025-01-15T00:00:00",
                        "event_id": "TEST:earnings",
                        "event_type": "earnings",
                        "expected_date": "2025-01-25",
                        "horizon_days": 10,
                        "importance": 0.4,
                    },
                    {
                        "ticker": "TEST",
                        "as_of": "2025-01-15T00:00:00",
                        "event_id": "TEST:phase3",
                        "event_type": "phase3_readout",
                        "expected_date": "2025-02-20",
                        "horizon_days": 36,
                        "importance": 0.9,
                    },
                ]
            )
            labeler = PointInTimeLabeler(
                store=store,
                history_provider=StubHistoryProvider({"TEST": make_price_frame()}),
            )
            labeler.materialize_labels(snapshots=snapshots, catalysts=catalysts)
            labels = store.read_table("labels")
            self.assertEqual(labels.iloc[0]["target_primary_event_type"], "phase3_readout")
            self.assertEqual(labels.iloc[0]["target_primary_event_bucket"], "clinical")

    def test_labeler_anchors_base_price_on_prior_trading_day(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            snapshots = pd.DataFrame(
                [
                    {"ticker": "TEST", "as_of": "2025-01-04T00:00:00"},  # Saturday
                ]
            )
            dates = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-02-03", "2025-04-04", "2025-07-03"])
            frame = pd.DataFrame({"close": [100.0, 110.0, 120.0, 130.0, 140.0]}, index=dates)
            labeler = PointInTimeLabeler(
                store=store,
                history_provider=StubHistoryProvider({"TEST": frame}),
            )
            summary = labeler.materialize_labels(snapshots=snapshots, catalysts=pd.DataFrame())
            labels = store.read_table("labels")
            self.assertEqual(summary.snapshot_label_rows, 1)
            self.assertAlmostEqual(labels.iloc[0]["target_return_30d"], (120.0 / 110.0) - 1.0)

    def test_evaluator_rebuilds_feature_frame_from_archived_snapshots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            snapshot = build_company_snapshot(
                {
                    "ticker": "TEST",
                    "company_name": "Replay Therapeutics",
                    "finance": {
                        "marketCap": 1_000_000_000,
                        "enterpriseValue": 900_000_000,
                        "totalRevenue": 0.0,
                        "cash": 300_000_000,
                        "debt": 50_000_000,
                        "momentum_3mo": 0.12,
                        "trailing_6mo_return": 0.08,
                        "volatility": 0.03,
                        "description": "Rare disease company with a pivotal hematology asset.",
                    },
                    "trials": [
                        {
                            "nct_id": "NCT-1",
                            "title": "RP-101 pivotal",
                            "overall_status": "RECRUITING",
                            "phase": ["Phase 3"],
                            "conditions": ["Beta-Thalassemia"],
                            "interventions": ["RP-101"],
                            "primary_outcomes": ["hemoglobin response"],
                            "enrollment": 250,
                        }
                    ],
                    "num_trials": 1,
                    "best_phase": "PHASE3",
                    "pubmed_papers": [],
                    "num_papers": 0,
                }
            )
            store.write_raw_payload("snapshots", "TEST_2025-01-15T00-00-00+00-00", asdict(snapshot))
            pd.DataFrame([{"ticker": "TEST", "as_of": snapshot.as_of}]).to_parquet(
                store.tables_dir / "company_snapshots.parquet",
                index=False,
            )
            pd.DataFrame(
                [
                    {
                        "ticker": "TEST",
                        "as_of": snapshot.as_of,
                        "target_return_90d": 0.10,
                        "target_catalyst_success": 1,
                    }
                ]
            ).to_parquet(store.tables_dir / "labels.parquet", index=False)

            frame = WalkForwardEvaluator(store=store).build_training_frame()
            self.assertFalse(frame.empty)
            self.assertIn("program_quality_pos_prior", frame.columns)

    def test_replay_engine_rebuilds_tables_from_archived_snapshots(self):
        snapshot_payload = {
            "ticker": "TEST",
            "company_name": "Replay Therapeutics",
            "as_of": "2025-01-15T00:00:00+00:00",
            "market_cap": 1_000_000_000,
            "enterprise_value": 900_000_000,
            "revenue": 75_000_000,
            "cash": 300_000_000,
            "debt": 50_000_000,
            "momentum_3mo": 0.12,
            "trailing_6mo_return": 0.08,
            "volatility": 0.03,
            "programs": [
                {
                    "program_id": "TEST:1",
                    "name": "RP-101",
                    "modality": "small molecule",
                    "phase": "PHASE3",
                    "conditions": ["Renal Cell Carcinoma"],
                    "trials": [
                        {
                            "trial_id": "NCT-1",
                            "title": "RP-101 pivotal",
                            "phase": "PHASE3",
                            "status": "RECRUITING",
                            "conditions": ["Renal Cell Carcinoma"],
                            "interventions": ["RP-101"],
                            "enrollment": 250,
                            "primary_outcomes": ["overall survival"],
                        }
                    ],
                    "pos_prior": 0.62,
                    "tam_estimate": 4_500_000_000,
                    "catalyst_events": [
                        {
                            "event_id": "TEST:1:readout",
                            "program_id": "TEST:1",
                            "event_type": "clinical_readout",
                            "title": "RP-101 readout",
                            "expected_date": "2025-03-01",
                            "horizon_days": 45,
                            "probability": 0.7,
                            "importance": 0.8,
                            "crowdedness": 0.2,
                            "status": "anticipated",
                        }
                    ],
                    "evidence": [],
                }
            ],
            "approved_products": [],
            "catalyst_events": [
                {
                    "event_id": "TEST:1:readout",
                    "program_id": "TEST:1",
                    "event_type": "clinical_readout",
                    "title": "RP-101 readout",
                    "expected_date": "2025-03-01",
                    "horizon_days": 45,
                    "probability": 0.7,
                    "importance": 0.8,
                    "crowdedness": 0.2,
                    "status": "anticipated",
                }
            ],
            "financing_events": [],
            "evidence": [],
            "metadata": {"runway_months": 24.0},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            raw_path = store.raw_dir / "snapshots" / "TEST_2025-01-15T00-00-00+00-00.json"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(snapshot_payload, f)

            summary = HistoricalReplayEngine(store=store).rebuild_from_archived_snapshots()
            self.assertEqual(summary.replayed_snapshots, 1)
            self.assertEqual(len(store.read_table("company_snapshots")), 1)
            self.assertFalse(store.read_table("feature_vectors").empty)
            self.assertFalse(store.read_table("predictions").empty)

    def test_evaluator_collapses_same_day_archives(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            features = pd.DataFrame(
                [
                    {
                        "entity_id": "TEST:1",
                        "ticker": "TEST",
                        "as_of": "2025-01-15T10:00:00",
                        "thesis_horizon": "90d",
                        "program_quality_pos_prior": 0.4,
                    },
                    {
                        "entity_id": "TEST:1",
                        "ticker": "TEST",
                        "as_of": "2025-01-15T15:00:00",
                        "thesis_horizon": "90d",
                        "program_quality_pos_prior": 0.5,
                    },
                    {
                        "entity_id": "TEST:1",
                        "ticker": "TEST",
                        "as_of": "2025-02-15T15:00:00",
                        "thesis_horizon": "90d",
                        "program_quality_pos_prior": 0.6,
                    },
                ]
            )
            labels = pd.DataFrame(
                [
                    {
                        "ticker": "TEST",
                        "as_of": "2025-01-15T10:00:00",
                        "target_return_90d": 0.10,
                        "target_catalyst_success": 1,
                    },
                    {
                        "ticker": "TEST",
                        "as_of": "2025-01-15T15:00:00",
                        "target_return_90d": 0.11,
                        "target_catalyst_success": 1,
                    },
                    {
                        "ticker": "TEST",
                        "as_of": "2025-02-15T15:00:00",
                        "target_return_90d": 0.05,
                        "target_catalyst_success": 0,
                    },
                ]
            )
            snapshots = pd.DataFrame(
                [
                    {"ticker": "TEST", "as_of": "2025-01-15T10:00:00"},
                    {"ticker": "TEST", "as_of": "2025-01-15T15:00:00"},
                    {"ticker": "TEST", "as_of": "2025-02-15T15:00:00"},
                ]
            )
            features.to_parquet(store.tables_dir / "feature_vectors.parquet", index=False)
            labels.to_parquet(store.tables_dir / "labels.parquet", index=False)
            snapshots.to_parquet(store.tables_dir / "company_snapshots.parquet", index=False)

            frame = WalkForwardEvaluator(store=store).build_training_frame()
            self.assertEqual(len(frame), 2)
            self.assertTrue((frame["evaluation_date"].nunique()) == 2)

    @patch("biopharma_agent.vnext.evaluation.EventDrivenEnsemble.fit", return_value=None)
    @patch("biopharma_agent.vnext.evaluation.EventDrivenEnsemble.score")
    def test_evaluator_normalizes_prediction_timestamps_before_merge(self, mock_score, _mock_fit):
        def fake_score(vectors):
            return [
                ModelPrediction(
                    entity_id=vector.entity_id,
                    ticker=vector.ticker,
                    as_of=str(vector.as_of),
                    expected_return=0.10,
                    catalyst_success_prob=0.60,
                    confidence=0.70,
                    crowding_risk=0.30,
                    financing_risk=0.20,
                    thesis_horizon=vector.thesis_horizon,
                    model_name="test",
                    model_version="v1",
                    metadata={},
                )
                for vector in vectors
            ]

        mock_score.side_effect = fake_score

        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            rows = []
            label_rows = []
            snapshot_rows = []
            for ticker in ["AAA", "BBB"]:
                for as_of, target in [
                    ("2025-01-15T00:00:00+00:00", 0.10),
                    ("2025-02-15T00:00:00+00:00", 0.05),
                    ("2025-03-15T00:00:00+00:00", -0.02),
                ]:
                    rows.append(
                        {
                            "entity_id": f"{ticker}:1",
                            "ticker": ticker,
                            "as_of": as_of,
                            "thesis_horizon": "90d",
                            "program_quality_pos_prior": 0.4 if ticker == "AAA" else 0.3,
                            "catalyst_timing_probability": 0.6,
                        }
                    )
                    label_rows.append(
                        {
                            "ticker": ticker,
                            "as_of": as_of,
                            "target_return_90d": target,
                            "target_catalyst_success": int(target > 0),
                        }
                    )
                    snapshot_rows.append({"ticker": ticker, "as_of": as_of})

            pd.DataFrame(rows).to_parquet(store.tables_dir / "feature_vectors.parquet", index=False)
            pd.DataFrame(label_rows).to_parquet(store.tables_dir / "labels.parquet", index=False)
            pd.DataFrame(snapshot_rows).to_parquet(store.tables_dir / "company_snapshots.parquet", index=False)

            settings = VNextSettings(
                evaluation_min_names_per_window=2,
                evaluation_rebalance_spacing_days=1,
                evaluation_turnover_book_weight_floor=1.0,
            )
            summary = WalkForwardEvaluator(store=store, settings=settings).evaluate(min_train_rows=2)
            self.assertGreaterEqual(summary.num_windows, 1)
            self.assertIn("none", summary.event_type_scorecards)

    def test_ensemble_fit_can_skip_artifact_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            rows = []
            for idx in range(24):
                rows.append(
                    {
                        "entity_id": f"T{idx}:1",
                        "ticker": f"T{idx}",
                        "as_of": f"2025-01-{(idx % 9) + 1:02d}T00:00:00+00:00",
                        "thesis_horizon": "90d",
                        "program_quality_pos_prior": 0.2 + (idx * 0.01),
                        "program_quality_phase_score": 0.3 + ((idx % 3) * 0.1),
                        "catalyst_timing_probability": 0.4 + ((idx % 4) * 0.05),
                        "catalyst_timing_expected_value": 0.1 + ((idx % 5) * 0.02),
                        "target_return_90d": -0.05 + (idx * 0.01),
                        "target_catalyst_success": idx % 2,
                    }
                )
            frame = pd.DataFrame(rows)
            record = EventDrivenEnsemble(store=store).fit(frame, persist_artifact=False)
            self.assertIsNotNone(record)
            self.assertIsNone(record.artifact_path)

    def test_evaluator_coarsens_rebalance_dates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = VNextSettings(
                evaluation_rebalance_spacing_days=14,
                evaluation_min_names_per_window=1,
                evaluation_max_snapshot_staleness_days=120,
                evaluation_turnover_book_weight_floor=1.0,
            )
            evaluator = WalkForwardEvaluator(store=LocalResearchStore(base_dir=tmpdir), settings=settings)
            unique_dates = list(pd.to_datetime(["2025-01-01", "2025-01-08", "2025-01-22", "2025-02-10"]))
            selected = evaluator._rebalance_dates(unique_dates)
            self.assertEqual([item.strftime("%Y-%m-%d") for item in selected], ["2025-01-01", "2025-01-22", "2025-02-10"])

    def test_evaluator_carries_forward_latest_snapshot_within_staleness_window(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            frame = pd.DataFrame(
                [
                    {
                        "entity_id": "AAA:1",
                        "ticker": "AAA",
                        "as_of": "2025-01-01T00:00:00+00:00",
                        "evaluation_date": pd.Timestamp("2025-01-01"),
                    },
                    {
                        "entity_id": "BBB:1",
                        "ticker": "BBB",
                        "as_of": "2025-01-15T00:00:00+00:00",
                        "evaluation_date": pd.Timestamp("2025-01-15"),
                    },
                ]
            )
            evaluator = WalkForwardEvaluator(
                store=store,
                settings=VNextSettings(
                    evaluation_rebalance_spacing_days=1,
                    evaluation_min_names_per_window=1,
                    evaluation_max_snapshot_staleness_days=30,
                ),
            )
            test = evaluator._latest_test_frame(frame, pd.Timestamp("2025-01-20"))
            self.assertEqual(sorted(test["ticker"].tolist()), ["AAA", "BBB"])

    def test_evaluator_reports_pm_context_and_synthetic_event_usage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            dates = ["2025-01-01T00:00:00+00:00", "2025-01-08T00:00:00+00:00", "2025-01-15T00:00:00+00:00"]
            snapshot_rows = []
            feature_rows = []
            label_rows = []
            for idx, as_of in enumerate(dates):
                snapshot_rows.append({"ticker": "AAA", "as_of": as_of})
                store.write_raw_payload(
                    "snapshots",
                    f"AAA_{as_of.replace(':', '-')}",
                    {
                        "ticker": "AAA",
                        "company_name": "AAA Bio",
                        "as_of": as_of,
                        "market_cap": 1_000_000_000,
                        "enterprise_value": 900_000_000,
                        "revenue": 0.0,
                        "cash": 250_000_000,
                        "debt": 0.0,
                        "momentum_3mo": 0.05,
                        "trailing_6mo_return": 0.0,
                        "volatility": 0.10,
                        "programs": [
                            {
                                "program_id": "AAA:1",
                                "name": "AAA-101",
                                "modality": "small molecule",
                                "phase": "PHASE3",
                                "conditions": ["oncology"],
                                "trials": [],
                                "pos_prior": 0.55,
                                "tam_estimate": 3_000_000_000,
                                "catalyst_events": [
                                    {
                                        "event_id": "AAA:1:phase3:120",
                                        "program_id": "AAA:1",
                                        "event_type": "phase3_readout",
                                        "title": "AAA-101 next milestone",
                                        "expected_date": "2025-05-01",
                                        "horizon_days": 120,
                                        "probability": 0.7,
                                        "importance": 0.8,
                                        "crowdedness": 0.2,
                                        "status": "phase_timing_estimate",
                                    }
                                ],
                                "evidence": [],
                            }
                        ],
                        "approved_products": [],
                        "catalyst_events": [
                            {
                                "event_id": "AAA:1:phase3:120",
                                "program_id": "AAA:1",
                                "event_type": "phase3_readout",
                                "title": "AAA-101 next milestone",
                                "expected_date": "2025-05-01",
                                "horizon_days": 120,
                                "probability": 0.7,
                                "importance": 0.8,
                                "crowdedness": 0.2,
                                "status": "phase_timing_estimate",
                            }
                        ],
                        "financing_events": [],
                        "evidence": [],
                        "metadata": {"price_now": 10.0},
                    },
                )
                feature_rows.append(
                    {
                        "entity_id": "AAA:1",
                        "ticker": "AAA",
                        "as_of": as_of,
                        "thesis_horizon": "90d",
                        "program_quality_pos_prior": 0.55,
                        "program_quality_phase_score": 0.75,
                        "program_quality_endpoint_score": 0.50,
                        "program_quality_tam_to_cap": 0.40,
                        "program_quality_modality_risk": 0.25,
                        "program_quality_trial_count": 1.0,
                        "catalyst_timing_probability": 0.70,
                        "catalyst_timing_expected_value": 0.45,
                        "catalyst_timing_crowdedness": 0.20,
                        "catalyst_timing_company_event_earnings": 0.0,
                        "commercial_execution_revenue_to_cap": -2.0,
                        "balance_sheet_cash_to_cap": 0.25,
                        "balance_sheet_financing_pressure": 0.0,
                        "market_flow_momentum_3mo": 0.05,
                        "market_flow_volatility": 0.10,
                        "state_profile_pre_commercial": 1.0,
                        "state_profile_commercial_launch": 0.0,
                        "state_profile_commercialized": 0.0,
                        "state_profile_competition_intensity": 0.82,
                        "state_profile_floor_support_pct": 0.25,
                        "state_profile_launch_progress_pct": 0.0,
                        "state_profile_lifecycle_management_score": 0.1,
                        "state_profile_pipeline_optionality_score": 0.2,
                        "state_profile_capital_deployment_score": 0.1,
                        "state_profile_hard_catalyst_presence": 1.0,
                        "state_profile_precommercial_value_gap": 0.40,
                        "meta_event_type": "phase3_readout",
                        "meta_event_status": "phase_timing_estimate",
                        "meta_company_state": "pre_commercial",
                    }
                )
                label_rows.append(
                    {
                        "ticker": "AAA",
                        "as_of": as_of,
                        "target_return_90d": 0.10 + (idx * 0.02),
                        "target_catalyst_success": 1,
                    }
                )

            pd.DataFrame(feature_rows).to_parquet(store.tables_dir / "feature_vectors.parquet", index=False)
            pd.DataFrame(label_rows).to_parquet(store.tables_dir / "labels.parquet", index=False)
            pd.DataFrame(snapshot_rows).to_parquet(store.tables_dir / "company_snapshots.parquet", index=False)

            settings = VNextSettings(
                evaluation_min_names_per_window=1,
                evaluation_rebalance_spacing_days=1,
                evaluation_max_snapshot_staleness_days=30,
                evaluation_turnover_book_weight_floor=1.0,
            )
            summary = WalkForwardEvaluator(store=store, settings=settings).evaluate(min_train_rows=2)
            self.assertGreaterEqual(summary.pm_context_coverage, 1.0)
            self.assertGreater(summary.synthetic_primary_event_rate, 0.0)
            self.assertTrue(summary.latest_window_top_trades)
            self.assertTrue(summary.institutional_blockers)


if __name__ == "__main__":
    unittest.main()
