import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from biopharma_agent.vnext.evaluation import WalkForwardSummary
from biopharma_agent.vnext.ops import build_readiness_report, record_pipeline_run
from biopharma_agent.vnext.settings import VNextSettings
from biopharma_agent.vnext.storage import LocalResearchStore


class TestVNextOps(unittest.TestCase):
    def test_settings_load_from_env(self):
        with patch.dict(
            os.environ,
            {
                "TATETUCK_STORE_DIR": "/tmp/tatetuck_store",
                "EODHD_API_KEY": "secret",
                "SEC_USER_AGENT": "TatetuckBot/1.0 ops@example.com",
                "TATETUCK_MIN_SNAPSHOT_DATES": "12",
            },
            clear=False,
        ):
            settings = VNextSettings.from_env()
            self.assertEqual(settings.store_dir, "/tmp/tatetuck_store")
            self.assertEqual(settings.min_snapshot_dates, 12)
            self.assertTrue(settings.public_metadata()["eodhd_api_key"])

    def test_record_pipeline_run_persists_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            record_pipeline_run(
                store=store,
                job_name="archive_vnext",
                status="success",
                started_at="2025-01-01T00:00:00+00:00",
                finished_at="2025-01-01T00:01:30+00:00",
                metrics={"archived_companies": 2},
                config={"limit": 2},
            )
            runs = store.read_table("pipeline_runs")
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs.iloc[0]["job_name"], "archive_vnext")

    @patch("biopharma_agent.vnext.ops.WalkForwardEvaluator.evaluate")
    def test_readiness_report_flags_missing_history(self, mock_evaluate):
        mock_evaluate.return_value = WalkForwardSummary(
            num_rows=0,
            num_windows=0,
            rank_ic=0.0,
            hit_rate=0.0,
            top_bottom_spread=0.0,
            turnover=0.0,
            max_drawdown=0.0,
            beta_adjusted_return=0.0,
            calibrated_brier=1.0,
            leakage_passed=False,
            message="Need more history.",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            settings = VNextSettings(
                store_dir=tmpdir,
                eodhd_api_key=None,
                sec_user_agent="TatetuckBot/1.0 support@tatetuck.local",
                include_literature=False,
                min_snapshot_dates=5,
                min_matured_return_rows=10,
                min_walkforward_windows=2,
                max_snapshot_age_hours=36,
                min_archive_runs=1,
            )
            report = build_readiness_report(store=store, settings=settings)
            self.assertEqual(report.status, "needs_attention")
            self.assertTrue(report.blockers)
            self.assertIn("Leakage audit is not passing.", report.blockers)

    @patch("biopharma_agent.vnext.ops.WalkForwardEvaluator.evaluate")
    def test_readiness_report_can_pass_when_thresholds_met(self, mock_evaluate):
        mock_evaluate.return_value = WalkForwardSummary(
            num_rows=120,
            num_windows=4,
            rank_ic=0.12,
            hit_rate=0.58,
            top_bottom_spread=0.09,
            turnover=0.45,
            max_drawdown=0.08,
            beta_adjusted_return=0.06,
            calibrated_brier=0.21,
            leakage_passed=True,
            message="ok",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            snapshot_rows = pd.DataFrame(
                [
                    {"ticker": "TEST", "as_of": "2025-01-01T00:00:00+00:00"},
                    {"ticker": "TEST", "as_of": "2025-01-02T00:00:00+00:00"},
                    {"ticker": "TEST", "as_of": "2025-01-03T00:00:00+00:00"},
                ]
            )
            labels = pd.DataFrame(
                [{"ticker": f"T{i}", "as_of": "2025-01-01T00:00:00+00:00", "target_return_90d": 0.1} for i in range(12)]
            )
            snapshot_rows.to_parquet(store.tables_dir / "company_snapshots.parquet", index=False)
            labels.to_parquet(store.tables_dir / "labels.parquet", index=False)
            pd.DataFrame([{"x": 1}]).to_parquet(store.tables_dir / "event_labels.parquet", index=False)
            store.write_raw_payload("market_prices_eodhd", "TEST_2025-01-01_2025-04-01", [{"date": "2025-01-02", "close": 10.0}])
            record_pipeline_run(
                store=store,
                job_name="archive_vnext",
                status="success",
                started_at="2025-01-01T00:00:00+00:00",
                finished_at="2025-01-01T00:00:10+00:00",
                metrics={},
                config={},
            )
            settings = VNextSettings(
                store_dir=tmpdir,
                eodhd_api_key="secret",
                sec_user_agent="TatetuckBot/1.0 ops@example.com",
                include_literature=False,
                min_snapshot_dates=3,
                min_matured_return_rows=10,
                min_walkforward_windows=3,
                max_snapshot_age_hours=100000,
                min_archive_runs=1,
            )
            report = build_readiness_report(store=store, settings=settings)
            self.assertEqual(report.status, "production_ready")
            self.assertFalse(report.blockers)

    @patch("biopharma_agent.vnext.ops.WalkForwardEvaluator.evaluate")
    def test_readiness_report_can_use_cached_validation_payload(self, mock_evaluate):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            snapshots = pd.DataFrame(
                [
                    {"ticker": "TEST", "as_of": "2025-01-01T00:00:00+00:00"},
                    {"ticker": "TEST", "as_of": "2025-01-02T00:00:00+00:00"},
                    {"ticker": "TEST", "as_of": "2025-01-03T00:00:00+00:00"},
                ]
            )
            labels = pd.DataFrame(
                [{"ticker": f"T{i}", "as_of": "2025-01-01T00:00:00+00:00", "target_return_90d": 0.1} for i in range(12)]
            )
            snapshots.to_parquet(store.tables_dir / "company_snapshots.parquet", index=False)
            labels.to_parquet(store.tables_dir / "labels.parquet", index=False)
            pd.DataFrame([{"x": 1, "target_event_return_10d": 0.02}]).to_parquet(store.tables_dir / "event_labels.parquet", index=False)
            store.write_raw_payload(
                "validation_audits",
                "latest_walkforward_audit",
                {
                    "generated_at": "2025-01-03T00:00:00+00:00",
                    "rows": 120,
                    "windows": 4,
                    "rank_ic": 0.12,
                    "hit_rate": 0.58,
                    "cost_adjusted_top_bottom_spread": 0.09,
                    "turnover": 0.45,
                    "max_drawdown": 0.08,
                    "beta_adjusted_return": 0.06,
                    "calibrated_brier": 0.21,
                    "leakage_passed": True,
                    "institutional_blockers": [],
                },
            )
            store.write_raw_payload("market_prices_eodhd", "TEST_2025-01-01_2025-04-01", [{"date": "2025-01-02", "close": 10.0}])
            record_pipeline_run(
                store=store,
                job_name="archive_vnext",
                status="success",
                started_at="2025-01-01T00:00:00+00:00",
                finished_at="2025-01-01T00:00:10+00:00",
                metrics={},
                config={},
            )
            settings = VNextSettings(
                store_dir=tmpdir,
                eodhd_api_key="secret",
                sec_user_agent="TatetuckBot/1.0 ops@example.com",
                include_literature=False,
                min_snapshot_dates=3,
                min_matured_return_rows=10,
                min_walkforward_windows=3,
                max_snapshot_age_hours=100000,
                min_archive_runs=1,
            )
            report = build_readiness_report(store=store, settings=settings, prefer_cached_validation=True)
            self.assertEqual(report.status, "production_ready")
            mock_evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
