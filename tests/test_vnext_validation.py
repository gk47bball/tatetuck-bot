import tempfile
import unittest

import pandas as pd

from biopharma_agent.vnext.entities import SignalArtifact
from biopharma_agent.vnext.evaluation import WalkForwardEvaluator
from biopharma_agent.vnext.failure_universe import failure_label_rows
from biopharma_agent.vnext.ops import record_pipeline_run
from biopharma_agent.vnext.settings import VNextSettings
from biopharma_agent.vnext.storage import LocalResearchStore
from biopharma_agent.vnext.validation import derive_promotion_record, load_best_validation_payload


class TestVNextValidation(unittest.TestCase):
    def test_failure_universe_row_can_restore_pm_context(self):
        row = pd.Series(
            failure_label_rows(
                {
                    "ticker": "FAIL",
                    "company": "Failure Therapeutics",
                    "failure_date": "2026-01-15",
                    "failure_type": "phase3_failure",
                    "indication": "oncology",
                    "drug_name": "FTX-101",
                    "phase_at_failure": "PHASE3",
                    "peak_market_cap_est": 900_000_000,
                    "post_failure_return": -0.62,
                }
            )[0]
        )
        signal = SignalArtifact(
            ticker="FAIL",
            as_of="2025-10-17",
            expected_return=0.12,
            catalyst_success_prob=0.61,
            confidence=0.72,
            crowding_risk=0.28,
            financing_risk=0.22,
            thesis_horizon="90d",
        )

        signal = WalkForwardEvaluator._apply_row_context_fallback(signal, row, predictions=[])

        self.assertEqual(signal.company_state, "pre_commercial")
        self.assertEqual(signal.setup_type, "hard_catalyst")
        self.assertEqual(signal.primary_event_type, "phase3_readout")
        self.assertGreater(signal.internal_upside_pct, 0.0)
        self.assertGreater(signal.floor_support_pct, 0.0)

    def test_load_best_validation_payload_merges_newer_run_with_richer_raw_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            store.write_raw_payload(
                "validation_audits",
                "latest_walkforward_audit",
                {
                    "generated_at": "2026-04-05T10:00:00+00:00",
                    "rank_ic": 0.20,
                    "a_grade_gates": {"paper_trading": {"passed": True}},
                },
            )
            record_pipeline_run(
                store=store,
                job_name="evaluate_vnext",
                status="success",
                started_at="2026-04-05T10:00:30+00:00",
                finished_at="2026-04-05T10:01:00+00:00",
                metrics={"rank_ic": 0.21, "windows": 7},
                config={},
            )

            payload = load_best_validation_payload(store)

            self.assertEqual(payload["rank_ic"], 0.21)
            self.assertEqual(payload["windows"], 7)
            self.assertIn("a_grade_gates", payload)
            self.assertTrue(payload["a_grade_gates"]["paper_trading"]["passed"])

    def test_derive_promotion_record_can_clear_paper_trading_without_full_a_grade(self):
        settings = VNextSettings(
            min_walkforward_windows=3,
            min_matured_return_rows=100,
        )
        record = derive_promotion_record(
            {
                "generated_at": "2026-04-05T11:00:00+00:00",
                "rows": 320,
                "windows": 12,
                "rank_ic": 0.18,
                "strict_rank_ic": 0.03,
                "hit_rate": 0.57,
                "cost_adjusted_top_bottom_spread": 0.14,
                "pm_context_coverage": 0.99,
                "exact_primary_event_rate": 0.91,
                "synthetic_primary_event_rate": 0.05,
                "institutional_blockers": [],
                "leakage_passed": True,
            },
            settings=settings,
        )

        self.assertEqual(record["decision"], "paper_trade_ready")
        self.assertTrue(record["a_grade_gates"]["paper_trading"]["passed"])
        self.assertFalse(record["a_grade_gates"]["a_grade_ready"]["passed"])
        self.assertTrue(any("A-grade" in blocker or "confidence band" in blocker or "strict exact-event" in blocker for blocker in record["blockers"]))


if __name__ == "__main__":
    unittest.main()
