from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import tempfile
import unittest

import pandas as pd

from biopharma_agent.vnext.graph import build_company_snapshot
from biopharma_agent.vnext.history import HistoricalSnapshotBootstrapper
from biopharma_agent.vnext.labels import PointInTimeLabeler
from biopharma_agent.vnext.storage import LocalResearchStore


class StubHistoryProvider:
    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames

    def load_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        frame = self.frames.get(ticker, pd.DataFrame(columns=["close"]))
        if frame.empty:
            return frame
        mask = (frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))
        return frame.loc[mask]


def make_price_frame() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=1200, freq="D")
    close = pd.Series([20.0 + (day * 0.05) for day in range(len(dates))], index=dates)
    return pd.DataFrame({"close": close.values}, index=dates)


def make_legacy_raw() -> dict:
    return {
        "ticker": "TEST",
        "company_name": "Test Therapeutics",
        "finance": {
            "marketCap": 2_000_000_000,
            "enterpriseValue": 1_700_000_000,
            "totalRevenue": 150_000_000,
            "cash": 450_000_000,
            "debt": 150_000_000,
            "momentum_3mo": 0.15,
            "trailing_6mo_return": 0.10,
            "volatility": 0.04,
            "price_now": 50.0,
            "description": "Late-stage oncology company with commercial transition and catalyst density.",
        },
        "trials": [
            {
                "nct_id": "NCT-TEST-1",
                "title": "Lead oncology trial",
                "overall_status": "RECRUITING",
                "phase": ["Phase 3"],
                "conditions": ["Renal Cell Carcinoma"],
                "interventions": ["TTX-101"],
                "primary_outcomes": ["overall survival"],
                "enrollment": 280,
            }
        ],
        "num_trials": 1,
        "best_phase": "PHASE3",
        "num_papers": 2,
        "pubmed_papers": [{"pmid": "1", "title": "Oncology data", "abstract": "Durable response signal."}],
        "conditions": ["Renal Cell Carcinoma"],
    }


def make_sec_payload() -> dict:
    return {
        "ticker": "TEST",
        "cik": "0000000123",
        "submissions": {
            "filings": {
                "recent": {
                    "form": ["10-Q", "10-Q", "10-K", "10-Q", "10-Q", "10-K"],
                    "filingDate": ["2024-05-10", "2024-08-08", "2025-02-27", "2025-05-08", "2025-08-07", "2026-02-26"],
                    "accessionNumber": ["0001", "0002", "0003", "0004", "0005", "0006"],
                    "primaryDocument": ["q1.htm", "q2.htm", "10k.htm", "q1_25.htm", "q2_25.htm", "10k_25.htm"],
                }
            }
        },
        "company_facts": {
            "facts": {
                "us-gaap": {
                    "RevenueFromContractWithCustomerExcludingAssessedTax": {
                        "units": {
                            "USD": [
                                {"start": "2023-01-01", "end": "2023-12-31", "filed": "2024-02-28", "val": 80_000_000},
                                {"start": "2024-01-01", "end": "2024-12-31", "filed": "2025-02-27", "val": 120_000_000},
                                {"start": "2025-01-01", "end": "2025-12-31", "filed": "2026-02-26", "val": 160_000_000},
                            ]
                        }
                    },
                    "CashAndCashEquivalentsAtCarryingValue": {
                        "units": {
                            "USD": [
                                {"end": "2024-03-31", "filed": "2024-05-10", "val": 250_000_000},
                                {"end": "2024-06-30", "filed": "2024-08-08", "val": 270_000_000},
                                {"end": "2024-12-31", "filed": "2025-02-27", "val": 320_000_000},
                                {"end": "2025-06-30", "filed": "2025-08-07", "val": 360_000_000},
                            ]
                        }
                    },
                    "LongTermDebt": {
                        "units": {
                            "USD": [
                                {"end": "2024-03-31", "filed": "2024-05-10", "val": 100_000_000},
                                {"end": "2024-12-31", "filed": "2025-02-27", "val": 150_000_000},
                                {"end": "2025-06-30", "filed": "2025-08-07", "val": 180_000_000},
                            ]
                        }
                    },
                    "NetCashProvidedByUsedInOperatingActivities": {
                        "units": {
                            "USD": [
                                {"start": "2024-01-01", "end": "2024-12-31", "filed": "2025-02-27", "val": -60_000_000},
                                {"start": "2025-01-01", "end": "2025-12-31", "filed": "2026-02-26", "val": -40_000_000},
                            ]
                        }
                    },
                },
                "dei": {
                    "EntityCommonStockSharesOutstanding": {
                        "units": {
                            "shares": [
                                {"end": "2024-05-01", "filed": "2024-05-10", "val": 40_000_000},
                                {"end": "2024-08-01", "filed": "2024-08-08", "val": 42_000_000},
                                {"end": "2025-02-20", "filed": "2025-02-27", "val": 44_000_000},
                                {"end": "2025-08-01", "filed": "2025-08-07", "val": 47_000_000},
                            ]
                        }
                    }
                },
            }
        },
    }


class TestVNextBootstrapHistory(unittest.TestCase):
    def test_bootstrapper_generates_multiple_historical_snapshots(self):
        latest_as_of = datetime(2026, 3, 28, 15, 22, 28, tzinfo=timezone.utc)
        raw = make_legacy_raw()
        snapshot = build_company_snapshot(raw, as_of=latest_as_of)
        snapshot.metadata["sec_cik"] = "0000000123"

        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            store.write_raw_payload("legacy_prepare", f"TEST_{snapshot.as_of.replace(':', '-')}", raw)
            store.write_raw_payload("sec_xbrl", f"TEST_{snapshot.as_of.replace(':', '-')}", make_sec_payload())
            store.write_raw_payload("snapshots", f"TEST_{snapshot.as_of.replace(':', '-')}", asdict(snapshot))
            store.write_snapshot(snapshot)

            bootstrapper = HistoricalSnapshotBootstrapper(
                store=store,
                history_provider=StubHistoryProvider({"TEST": make_price_frame()}),
            )
            summary = bootstrapper.materialize(ticker="TEST", max_anchors_per_ticker=4)

            snapshots = store.read_table("company_snapshots")
            self.assertEqual(summary.generated_snapshots, 4)
            self.assertEqual(summary.tickers_with_history, 1)
            self.assertGreaterEqual(snapshots["as_of"].nunique(), 5)
            older = snapshots[snapshots["as_of"] != snapshot.as_of].sort_values("as_of").iloc[0]
            metadata = json.loads(older["metadata"]) if isinstance(older["metadata"], str) else older["metadata"]
            self.assertTrue(bool(metadata["history_reconstruction"]))
            self.assertGreater(float(older["revenue"]), 0.0)
            self.assertGreater(float(older["market_cap"]), 0.0)

            labeler = PointInTimeLabeler(
                store=store,
                history_provider=StubHistoryProvider({"TEST": make_price_frame()}),
            )
            label_summary = labeler.materialize_labels(
                snapshots=store.read_table("company_snapshots"),
                catalysts=store.read_table("catalysts"),
            )
            labels = store.read_table("labels").sort_values("as_of")
            self.assertGreaterEqual(label_summary.snapshot_label_rows, 5)
            self.assertGreater(labels["target_return_90d"].notna().sum(), 0)


if __name__ == "__main__":
    unittest.main()
