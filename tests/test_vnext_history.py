import json
import tempfile
import unittest

import pandas as pd

from biopharma_agent.vnext.evaluation import WalkForwardEvaluator
from biopharma_agent.vnext.labels import PointInTimeLabeler
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


if __name__ == "__main__":
    unittest.main()
