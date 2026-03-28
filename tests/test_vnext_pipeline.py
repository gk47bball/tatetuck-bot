import tempfile
import unittest
from unittest.mock import patch

from biopharma_agent.vnext.archive import ArchiveSummary
from biopharma_agent.vnext.evaluation import WalkForwardSummary
from biopharma_agent.vnext.labels import LabelSummary
from biopharma_agent.vnext.pipeline import run_vnext_pipeline
from biopharma_agent.vnext.replay import ReplaySummary
from biopharma_agent.vnext.settings import VNextSettings
from biopharma_agent.vnext.storage import LocalResearchStore


class TestVNextPipeline(unittest.TestCase):
    @patch("biopharma_agent.vnext.pipeline.build_readiness_report")
    @patch("biopharma_agent.vnext.pipeline.WalkForwardEvaluator.evaluate")
    @patch("biopharma_agent.vnext.pipeline.PointInTimeLabeler.materialize_labels")
    @patch("biopharma_agent.vnext.pipeline.HistoricalReplayEngine.rebuild_from_archived_snapshots")
    @patch("biopharma_agent.vnext.pipeline.archive_universe")
    def test_run_pipeline_records_success(
        self,
        mock_archive,
        mock_replay,
        mock_labels,
        mock_evaluate,
        mock_readiness,
    ):
        mock_archive.return_value = (
            [],
            ArchiveSummary(
                archived_companies=2,
                archived_at="2025-01-01T00:00:00+00:00",
                sec_enriched_companies=2,
                financing_flagged_companies=0,
                snapshot_rows=2,
                feature_rows=4,
                prediction_rows=4,
                store_dir="tmp",
                top_ideas=[],
            ),
        )
        mock_replay.return_value = ReplaySummary(
            replayed_snapshots=2,
            replayed_tickers=2,
            feature_rows_written=4,
            prediction_rows_written=4,
            earliest_as_of="2025-01-01T00:00:00+00:00",
            latest_as_of="2025-01-02T00:00:00+00:00",
            store_dir="tmp",
        )
        mock_labels.return_value = LabelSummary(
            snapshot_label_rows=10,
            event_label_rows=2,
            matured_return_90d_rows=8,
            matured_event_rows=1,
            num_tickers=2,
        )
        mock_evaluate.return_value = WalkForwardSummary(
            num_rows=10,
            num_windows=2,
            rank_ic=0.1,
            hit_rate=0.6,
            top_bottom_spread=0.05,
            turnover=0.4,
            max_drawdown=0.1,
            beta_adjusted_return=0.03,
            calibrated_brier=0.2,
            leakage_passed=True,
            message="ok",
        )
        mock_readiness.return_value = type(
            "Readiness",
            (),
            {"status": "needs_attention", "blockers": ["need more history"]},
        )()

        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            settings = VNextSettings(
                store_dir=tmpdir,
                eodhd_api_key="secret",
                sec_user_agent="TatetuckBot/1.0 ops@example.com",
                include_literature=False,
                min_snapshot_dates=3,
                min_matured_return_rows=10,
                min_walkforward_windows=2,
                max_snapshot_age_hours=24,
                min_archive_runs=1,
            )
            summary = run_vnext_pipeline(
                universe=[("TEST", "Test Therapeutics")],
                settings=settings,
                store=store,
                run_evaluation=True,
            )
            self.assertEqual(summary.archive.archived_companies, 2)
            runs = store.read_table("pipeline_runs")
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs.iloc[0]["job_name"], "operate_vnext")


if __name__ == "__main__":
    unittest.main()
