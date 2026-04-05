import tempfile
import unittest

import pandas as pd

from biopharma_agent.vnext.catalyst import CatalystEventEvaluator, CatalystEventStackBuilder
from biopharma_agent.vnext.settings import VNextSettings
from biopharma_agent.vnext.storage import LocalResearchStore


class StubHistoryProvider:
    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames

    def load_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        frame = self.frames.get(ticker, pd.DataFrame(columns=["close"]))
        if frame.empty:
            return frame
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        subset = frame[(frame.index >= start_ts) & (frame.index <= end_ts)].copy()
        return subset[["close"]] if "close" in subset.columns else pd.DataFrame(columns=["close"])


def price_frame(rows: list[tuple[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["date", "close"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date").sort_index()
    return frame


class TestCatalystEventStack(unittest.TestCase):
    def test_event_master_freezes_source_and_context_at_first_seen(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            store.replace_table(
                "company_snapshots",
                [
                    {
                        "ticker": "TEST",
                        "as_of": "2026-03-01T00:00:00+00:00",
                        "market_cap": 2_000_000_000,
                        "revenue": 250_000_000,
                        "cash": 700_000_000,
                        "debt": 100_000_000,
                        "momentum_3mo": 0.10,
                        "volatility": 0.25,
                        "num_approved_products": 1,
                    },
                    {
                        "ticker": "TEST",
                        "as_of": "2026-03-15T00:00:00+00:00",
                        "market_cap": 8_000_000_000,
                        "revenue": 1_500_000_000,
                        "cash": 2_000_000_000,
                        "debt": 50_000_000,
                        "momentum_3mo": 0.35,
                        "volatility": 0.12,
                        "num_approved_products": 4,
                    }
                ],
            )
            store.replace_table(
                "catalysts",
                [
                    {
                        "ticker": "TEST",
                        "as_of": "2026-03-01T00:00:00+00:00",
                        "event_id": "internal-1",
                        "program_id": "drug-a",
                        "event_type": "phase3_readout",
                        "title": "Anticipated phase 3 readout",
                        "expected_date": "2026-04-10T00:00:00+00:00",
                        "status": "phase_timing_estimate",
                        "source": "internal_inference",
                        "timing_exact": False,
                        "timing_synthetic": True,
                        "probability": 0.60,
                        "importance": 0.50,
                        "crowdedness": 0.30,
                        "provenance_confidence": 0.30,
                    },
                    {
                        "ticker": "TEST",
                        "as_of": "2026-03-15T00:00:00+00:00",
                        "event_id": "sec-1",
                        "program_id": "drug-a",
                        "event_type": "phase3_readout",
                        "title": "Company announces exact phase 3 topline timing",
                        "expected_date": "2026-04-10T00:00:00+00:00",
                        "status": "exact_sec_filing",
                        "source": "sec",
                        "timing_exact": True,
                        "timing_synthetic": False,
                        "probability": 0.85,
                        "importance": 0.80,
                        "crowdedness": 0.20,
                        "provenance_confidence": 0.90,
                    },
                ],
            )
            builder = CatalystEventStackBuilder(
                store=store,
                settings=VNextSettings(store_dir=tmpdir),
                history_provider=StubHistoryProvider({}),
            )

            event_master = builder.build_event_master(
                snapshots=store.read_table("company_snapshots"),
                catalysts=store.read_table("catalysts"),
            )

            self.assertEqual(len(event_master), 1)
            self.assertEqual(event_master.iloc[0]["source"], "internal_inference")
            self.assertFalse(bool(event_master.iloc[0]["strict_schedule_eligible"]))
            self.assertFalse(bool(event_master.iloc[0]["corroborated"]))
            self.assertEqual(event_master.iloc[0]["company_state"], "commercial_launch")
            self.assertEqual(int(event_master.iloc[0]["days_until_first_seen"]), 40)
            self.assertEqual(event_master.iloc[0]["latest_source"], "sec")
            self.assertTrue(bool(event_master.iloc[0]["latest_strict_schedule_eligible"]))
            self.assertTrue(bool(event_master.iloc[0]["latest_corroborated"]))

    def test_trade_labels_cluster_overlapping_same_ticker_family_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            store.replace_table(
                "company_snapshots",
                [
                    {
                        "ticker": "TEST",
                        "as_of": "2026-03-01T00:00:00+00:00",
                        "market_cap": 2_000_000_000,
                        "revenue": 200_000_000,
                        "cash": 500_000_000,
                        "debt": 100_000_000,
                        "momentum_3mo": 0.05,
                        "volatility": 0.20,
                        "num_approved_products": 1,
                    },
                    {
                        "ticker": "TEST",
                        "as_of": "2026-03-10T00:00:00+00:00",
                        "market_cap": 2_100_000_000,
                        "revenue": 220_000_000,
                        "cash": 520_000_000,
                        "debt": 100_000_000,
                        "momentum_3mo": 0.04,
                        "volatility": 0.22,
                        "num_approved_products": 1,
                    },
                ],
            )
            store.replace_table(
                "catalysts",
                [
                    {
                        "ticker": "TEST",
                        "as_of": "2026-03-01T00:00:00+00:00",
                        "event_id": "evt-a",
                        "program_id": "drug-a",
                        "event_type": "phase3_readout",
                        "title": "First exact phase 3 date",
                        "expected_date": "2026-04-10T00:00:00+00:00",
                        "status": "exact_press_release",
                        "source": "eodhd_news",
                        "timing_exact": True,
                        "timing_synthetic": False,
                        "probability": 0.80,
                        "importance": 0.80,
                        "crowdedness": 0.20,
                        "provenance_confidence": 0.90,
                    },
                    {
                        "ticker": "TEST",
                        "as_of": "2026-03-10T00:00:00+00:00",
                        "event_id": "evt-b",
                        "program_id": "drug-a",
                        "event_type": "phase3_readout",
                        "title": "Revised exact phase 3 date",
                        "expected_date": "2026-04-18T00:00:00+00:00",
                        "status": "exact_press_release",
                        "source": "eodhd_news",
                        "timing_exact": True,
                        "timing_synthetic": False,
                        "probability": 0.82,
                        "importance": 0.84,
                        "crowdedness": 0.18,
                        "provenance_confidence": 0.91,
                    },
                ],
            )
            store.replace_table("event_tape", [])
            provider = StubHistoryProvider(
                {
                    "TEST": price_frame(
                        [
                            ("2026-03-21", 100.0),
                            ("2026-04-11", 104.0),
                            ("2026-04-19", 108.0),
                            ("2026-04-24", 111.0),
                        ]
                    )
                }
            )
            builder = CatalystEventStackBuilder(
                store=store,
                settings=VNextSettings(store_dir=tmpdir),
                history_provider=provider,
            )

            summary = builder.materialize()
            event_master = store.read_table("catalyst_event_master")
            trade_labels = store.read_table("catalyst_trade_labels")
            pre_event_trades = trade_labels[trade_labels["sleeve"] == "pre_event_long"].copy()

            self.assertEqual(summary.event_master_rows, 2)
            self.assertEqual(event_master["cluster_active"].fillna(False).astype(bool).sum(), 1)
            self.assertEqual(pre_event_trades["event_instance_id"].nunique(), 1)
            self.assertEqual(
                pre_event_trades.iloc[0]["event_instance_id"],
                event_master[event_master["cluster_active"].fillna(False)].iloc[0]["event_instance_id"],
            )

    def test_materialize_builds_pre_post_and_short_trade_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            snapshots = [
                {
                    "ticker": "POS",
                    "as_of": "2026-03-20T00:00:00+00:00",
                    "market_cap": 3_000_000_000,
                    "revenue": 350_000_000,
                    "cash": 900_000_000,
                    "debt": 120_000_000,
                    "momentum_3mo": 0.08,
                    "volatility": 0.22,
                    "num_approved_products": 1,
                },
                {
                    "ticker": "NEG",
                    "as_of": "2026-03-20T00:00:00+00:00",
                    "market_cap": 1_400_000_000,
                    "revenue": 40_000_000,
                    "cash": 250_000_000,
                    "debt": 150_000_000,
                    "momentum_3mo": -0.04,
                    "volatility": 0.30,
                    "num_approved_products": 0,
                },
            ]
            catalysts = [
                {
                    "ticker": "POS",
                    "as_of": "2026-03-20T00:00:00+00:00",
                    "event_id": "pos-evt",
                    "program_id": "drug-pos",
                    "event_type": "pdufa",
                    "title": "Positive exact PDUFA date announced",
                    "expected_date": "2026-04-10T00:00:00+00:00",
                    "status": "exact_press_release",
                    "source": "eodhd_news",
                    "timing_exact": True,
                    "timing_synthetic": False,
                    "probability": 0.90,
                    "importance": 0.85,
                    "crowdedness": 0.18,
                    "provenance_confidence": 0.88,
                },
                {
                    "ticker": "NEG",
                    "as_of": "2026-03-20T00:00:00+00:00",
                    "event_id": "neg-evt",
                    "program_id": "drug-neg",
                    "event_type": "phase3_readout",
                    "title": "Negative exact phase 3 date announced",
                    "expected_date": "2026-04-10T00:00:00+00:00",
                    "status": "exact_press_release",
                    "source": "eodhd_news",
                    "timing_exact": True,
                    "timing_synthetic": False,
                    "probability": 0.82,
                    "importance": 0.82,
                    "crowdedness": 0.22,
                    "provenance_confidence": 0.86,
                },
            ]
            event_tape = [
                {
                    "ticker": "POS",
                    "event_id": "pos-outcome",
                    "event_type": "pdufa",
                    "title": "FDA approved the therapy",
                    "details": "Company received FDA approval and positive label.",
                    "event_timestamp": "2026-04-11T00:00:00+00:00",
                    "status": "exact_press_release",
                    "source": "eodhd_news",
                    "source_url": "https://example.com/pos",
                    "timing_exact": True,
                },
                {
                    "ticker": "NEG",
                    "event_id": "neg-outcome",
                    "event_type": "phase3_readout",
                    "title": "Company failed to meet the primary endpoint",
                    "details": "Topline phase 3 data missed the primary endpoint.",
                    "event_timestamp": "2026-04-11T00:00:00+00:00",
                    "status": "exact_press_release",
                    "source": "eodhd_news",
                    "source_url": "https://example.com/neg",
                    "timing_exact": True,
                },
            ]
            store.replace_table("company_snapshots", snapshots)
            store.replace_table("catalysts", catalysts)
            store.replace_table("event_tape", event_tape)
            provider = StubHistoryProvider(
                {
                    "POS": price_frame(
                        [
                            ("2026-03-31", 100.0),
                            ("2026-04-11", 120.0),
                            ("2026-04-12", 121.0),
                            ("2026-04-17", 130.0),
                        ]
                    ),
                    "NEG": price_frame(
                        [
                            ("2026-03-31", 100.0),
                            ("2026-04-11", 80.0),
                            ("2026-04-12", 78.0),
                            ("2026-04-17", 60.0),
                        ]
                    ),
                }
            )
            builder = CatalystEventStackBuilder(
                store=store,
                settings=VNextSettings(store_dir=tmpdir),
                history_provider=provider,
            )

            summary = builder.materialize()
            trade_labels = store.read_table("catalyst_trade_labels")

            self.assertEqual(summary.event_master_rows, 2)
            self.assertEqual(summary.outcome_master_rows, 2)
            self.assertIn("pre_event_long", set(trade_labels["sleeve"]))
            self.assertIn("post_event_reaction_long", set(trade_labels["sleeve"]))
            self.assertIn("event_short_or_pairs", set(trade_labels["sleeve"]))
            self.assertTrue(
                (
                    (trade_labels["ticker"] == "POS")
                    & (trade_labels["sleeve"] == "post_event_reaction_long")
                ).any()
            )
            self.assertTrue(
                (
                    (trade_labels["ticker"] == "NEG")
                    & (trade_labels["sleeve"] == "event_short_or_pairs")
                ).any()
            )

    def test_catalyst_evaluator_reports_separate_sleeve_scorecards(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            trade_labels = []
            for idx in range(24):
                entry_date = pd.Timestamp("2025-01-01") + pd.Timedelta(days=idx * 14)
                family = "phase3_readout" if idx % 2 == 0 else "pdufa"
                trade_labels.append(
                    {
                        "sleeve": "pre_event_long",
                        "variant": "pre_10d_plus1d",
                        "event_instance_id": f"pre-{idx}",
                        "ticker": f"P{idx:02d}",
                        "event_family": family,
                        "event_type": family,
                        "event_bucket": "clinical" if family == "phase3_readout" else "regulatory",
                        "entry_date": entry_date.isoformat(),
                        "exit_date": (entry_date + pd.Timedelta(days=11)).isoformat(),
                        "gross_return": 0.12 + (idx % 3) * 0.01,
                        "cost_adjusted_return": 0.10 + (idx % 3) * 0.01,
                        "score": 0.70 + (idx % 5) * 0.03,
                        "regime": "neutral_momentum",
                        "horizon_bucket": "31_60d",
                        "knowledge_bucket": "known_22_45d",
                        "exact_timing": True,
                        "exact_outcome": True,
                    }
                )
                trade_labels.append(
                    {
                        "sleeve": "post_event_reaction_long",
                        "variant": "post_next_bar_plus5d",
                        "event_instance_id": f"post-{idx}",
                        "ticker": f"Q{idx:02d}",
                        "event_family": family,
                        "event_type": family,
                        "event_bucket": "clinical" if family == "phase3_readout" else "regulatory",
                        "entry_date": entry_date.isoformat(),
                        "exit_date": (entry_date + pd.Timedelta(days=6)).isoformat(),
                        "gross_return": 0.09 + (idx % 2) * 0.01,
                        "cost_adjusted_return": 0.08 + (idx % 2) * 0.01,
                        "score": 0.68 + (idx % 4) * 0.02,
                        "regime": "neutral_momentum",
                        "horizon_bucket": "07_30d",
                        "knowledge_bucket": "known_8_21d",
                        "exact_timing": True,
                        "exact_outcome": True,
                    }
                )
            store.replace_table("catalyst_trade_labels", trade_labels)
            store.replace_table(
                "catalyst_event_master",
                [
                    {"event_instance_id": f"evt-{idx}", "ticker": f"P{idx:02d}"}
                    for idx in range(24)
                ],
            )
            settings = VNextSettings(
                store_dir=tmpdir,
                catalyst_pre_event_min_rows=20,
                catalyst_pre_event_min_windows=8,
                catalyst_pre_event_min_rank_ic=-1.0,
                catalyst_pre_event_min_net_spread=-1.0,
                catalyst_pre_event_min_hit_rate=0.0,
                catalyst_pre_event_min_non_negative_trailing_windows=0,
                catalyst_post_event_min_rows=20,
                catalyst_post_event_min_windows=8,
                catalyst_post_event_min_rank_ic=-1.0,
                catalyst_post_event_min_net_spread=-1.0,
                catalyst_exact_timing_rate_min=0.5,
                catalyst_exact_outcome_rate_min=0.5,
            )

            audit = CatalystEventEvaluator(store=store, settings=settings).evaluate(source_job="unit_test")

            self.assertIn("pre_event_long", audit["sleeve_scorecards"])
            self.assertIn("post_event_reaction_long", audit["sleeve_scorecards"])
            self.assertIn("phase3_readout", audit["family_scorecards"])
            self.assertIn("pdufa", audit["family_scorecards"])
            self.assertIn("overall_catalyst_bot", audit["gates"])

    def test_catalyst_evaluator_exact_rates_use_non_empty_evaluated_sleeves(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            trade_labels = []
            for idx in range(10):
                entry_date = pd.Timestamp("2025-01-01") + pd.Timedelta(days=idx * 21)
                trade_labels.append(
                    {
                        "sleeve": "post_event_reaction_long",
                        "variant": "post_next_bar_plus5d",
                        "event_instance_id": f"post-{idx}",
                        "ticker": f"Q{idx:02d}",
                        "event_family": "pdufa",
                        "event_type": "pdufa",
                        "event_bucket": "regulatory",
                        "entry_date": entry_date.isoformat(),
                        "exit_date": (entry_date + pd.Timedelta(days=6)).isoformat(),
                        "gross_return": 0.04,
                        "cost_adjusted_return": 0.03,
                        "score": 0.60 + (idx * 0.01),
                        "regime": "neutral_momentum",
                        "horizon_bucket": "07_30d",
                        "knowledge_bucket": "known_8_21d",
                        "exact_timing": True,
                        "exact_outcome": True,
                    }
                )
            store.replace_table("catalyst_trade_labels", trade_labels)
            store.replace_table("catalyst_event_master", [{"event_instance_id": "evt-1", "ticker": "Q00"}])

            audit = CatalystEventEvaluator(store=store, settings=VNextSettings(store_dir=tmpdir)).evaluate(source_job="unit_test")

            self.assertAlmostEqual(audit["exact_timing_rate"], 1.0, places=6)
            self.assertAlmostEqual(audit["exact_outcome_rate"], 1.0, places=6)
            self.assertAlmostEqual(
                audit["sleeve_exactness"]["post_event_reaction_long"]["exact_timing_rate"],
                1.0,
                places=6,
            )


if __name__ == "__main__":
    unittest.main()
