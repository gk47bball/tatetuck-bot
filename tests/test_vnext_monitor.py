import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from biopharma_agent.vnext.entities import CompanyAnalysis, CompanySnapshot, PortfolioRecommendation, SignalArtifact
from biopharma_agent.vnext.execution import BrokerAccount, BrokerPosition, PMExecutionPlanner
from biopharma_agent.vnext.monitor import AutonomyMonitor
from biopharma_agent.vnext.settings import VNextSettings
from biopharma_agent.vnext.storage import LocalResearchStore
from biopharma_agent.vnext.trigger_ingestion import TriggerIngestionSummary


def make_analysis(ticker: str) -> CompanyAnalysis:
    snapshot = CompanySnapshot(
        ticker=ticker,
        company_name=f"{ticker} Therapeutics",
        as_of="2026-04-05T13:00:00+00:00",
        market_cap=1_000_000_000,
        enterprise_value=900_000_000,
        revenue=0.0,
        cash=250_000_000,
        debt=0.0,
        momentum_3mo=0.1,
        trailing_6mo_return=0.0,
        volatility=0.05,
        programs=[],
        approved_products=[],
        catalyst_events=[],
        financing_events=[],
    )
    signal = SignalArtifact(
        ticker=ticker,
        as_of=snapshot.as_of,
        expected_return=0.20,
        catalyst_success_prob=0.72,
        confidence=0.74,
        crowding_risk=0.2,
        financing_risk=0.2,
        thesis_horizon="90d",
        primary_event_type="phase3_readout",
        primary_event_bucket="clinical",
        primary_event_status="exact_press_release",
        primary_event_date="2026-04-05T12:30:00+00:00",
        primary_event_exact=True,
        primary_event_synthetic=False,
        company_state="pre_commercial",
        setup_type="hard_catalyst",
        internal_upside_pct=0.24,
        floor_support_pct=0.14,
        rationale=["Topline event now requires immediate refresh."],
        supporting_evidence=[],
    )
    rec = PortfolioRecommendation(
        ticker=ticker,
        as_of=snapshot.as_of,
        stance="long",
        target_weight=4.0,
        max_weight=6.0,
        confidence=0.74,
        scenario="pre-catalyst long",
        thesis_horizon="90d",
        primary_event_type="phase3_readout",
        company_state="pre_commercial",
        setup_type="hard_catalyst",
        risk_flags=[],
    )
    return CompanyAnalysis(
        snapshot=snapshot,
        signal=signal,
        portfolio=rec,
        feature_vectors=[],
        program_predictions=[],
    )


class FakePlatform:
    def __init__(self):
        self.calls: list[str] = []

    def analyze_ticker(self, ticker: str, **kwargs) -> CompanyAnalysis:
        self.calls.append(ticker)
        return make_analysis(ticker)


class NoopBroker:
    def is_configured(self) -> bool:
        return False

    def simulated_account(self) -> BrokerAccount:
        return BrokerAccount(
            account_id="SIM-1",
            status="SIMULATED",
            equity=100000.0,
            buying_power=100000.0,
            cash=100000.0,
            paper=True,
            trading_blocked=False,
            account_blocked=False,
            pattern_day_trader=False,
        )


class NoopTriggerIngestor:
    def ingest_symbols(self, symbols, *, as_of, **kwargs) -> TriggerIngestionSummary:
        tickers = [str(symbol).upper() for symbol in symbols]
        return TriggerIngestionSummary(
            captured_at=as_of.isoformat(),
            symbols_requested=len(tickers),
            symbols_polled=len(tickers),
            symbols_with_new_events=0,
            eodhd_event_rows=0,
            sec_event_rows=0,
            triggered_symbols=[],
            warnings=[],
        )


class WritingTriggerIngestor:
    def __init__(self, store: LocalResearchStore):
        self.store = store
        self.calls: list[list[str]] = []

    def ingest_symbols(self, symbols, *, as_of, **kwargs) -> TriggerIngestionSummary:
        tickers = [str(symbol).upper() for symbol in symbols]
        self.calls.append(tickers)
        self.store.append_records(
            "event_tape",
            [
                {
                    "ticker": "CRSP",
                    "as_of": as_of.isoformat(),
                    "event_id": "CRSP:live:event",
                    "event_type": "phase3_readout",
                    "title": "CRSP exact topline press release",
                    "event_timestamp": (as_of - timedelta(minutes=5)).isoformat(),
                    "status": "exact_press_release",
                    "timing_exact": True,
                    "timing_synthetic": False,
                    "source": "test_live_event",
                }
            ],
        )
        return TriggerIngestionSummary(
            captured_at=as_of.isoformat(),
            symbols_requested=len(tickers),
            symbols_polled=len(tickers),
            symbols_with_new_events=1,
            eodhd_event_rows=1,
            sec_event_rows=0,
            triggered_symbols=["CRSP"],
            warnings=[],
        )


class FakeConfiguredBroker:
    def is_configured(self) -> bool:
        return True

    def ensure_expected_account(self, account: BrokerAccount) -> None:
        return None

    def account(self) -> BrokerAccount:
        return BrokerAccount(
            account_id="PA-1",
            status="ACTIVE",
            equity=100000.0,
            buying_power=100000.0,
            cash=100000.0,
            paper=True,
            trading_blocked=False,
            account_blocked=False,
            pattern_day_trader=False,
        )

    def positions(self) -> list[BrokerPosition]:
        return [
            BrokerPosition(
                symbol="CRSP",
                qty=25.0,
                market_value=750.0,
                current_price=30.0,
                side="long",
            )
        ]

    def recent_orders(self, limit: int = 200):
        return []

    def simulated_account(self):
        raise AssertionError("configured broker should not use simulated account")

    def ensure_paper_only(self) -> None:
        return None

    def submit_market_notional_buy(self, symbol: str, notional: float):
        raise AssertionError("submit should not be called in dry-run test")

    def submit_market_qty_sell(self, symbol: str, qty: float):
        raise AssertionError("submit should not be called in dry-run test")


class TestVNextMonitor(unittest.TestCase):
    def make_settings(self, tmpdir: str) -> VNextSettings:
        return VNextSettings(
            store_dir=tmpdir,
            eodhd_api_key="secret",
            sec_user_agent="TatetuckBot/1.0 ops@example.com",
            include_literature=False,
            min_snapshot_dates=3,
            min_matured_return_rows=10,
            min_walkforward_windows=2,
            max_snapshot_age_hours=36,
            min_archive_runs=1,
            allow_blocked_paper_trading=True,
            monitor_snapshot_stale_hours=6,
            monitor_max_symbols_per_cycle=5,
        )

    def test_detect_triggers_picks_recent_exact_event_and_stale_active_symbol(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            now = datetime(2026, 4, 5, 14, 0, tzinfo=timezone.utc)
            store.append_records(
                "event_tape",
                [
                    {
                        "ticker": "ARVN",
                        "as_of": now.isoformat(),
                        "event_id": "ARVN:event",
                        "event_type": "pdufa",
                        "title": "ARVN PDUFA decision",
                        "event_timestamp": (now - timedelta(hours=1)).isoformat(),
                        "status": "exact_press_release",
                        "timing_exact": True,
                        "timing_synthetic": False,
                    }
                ],
            )
            store.append_records(
                "company_snapshots",
                [
                    {
                        "ticker": "CRSP",
                        "company_name": "CRSP Therapeutics",
                        "as_of": (now - timedelta(hours=18)).isoformat(),
                        "market_cap": 1_000_000_000,
                    }
                ],
            )
            store.append_records(
                "trade_decisions",
                [
                    {
                        "generated_at": now.isoformat(),
                        "symbol": "CRSP",
                        "action": "hold",
                    }
                ],
            )

            monitor = AutonomyMonitor(
                settings=self.make_settings(tmpdir),
                store=store,
                platform=FakePlatform(),
                broker=NoopBroker(),
                planner=PMExecutionPlanner(settings=self.make_settings(tmpdir), store=store),
                trigger_ingestor=NoopTriggerIngestor(),
            )

            triggers = monitor.detect_triggers(now=now)
            symbols = [item.symbol for item in triggers]

            self.assertIn("ARVN", symbols)
            self.assertIn("CRSP", symbols)
            self.assertEqual(symbols[0], "ARVN")

    def test_run_once_persists_monitor_heartbeat_and_plan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            settings = self.make_settings(tmpdir)
            now = datetime(2026, 4, 5, 14, 0, tzinfo=timezone.utc)
            store.write_raw_payload(
                "validation_audits",
                "latest_walkforward_audit",
                {
                    "generated_at": now.isoformat(),
                    "rows": 180,
                    "windows": 6,
                    "rank_ic": 0.18,
                    "hit_rate": 0.56,
                    "cost_adjusted_top_bottom_spread": 0.11,
                    "leakage_passed": True,
                    "institutional_blockers": [],
                    "promotion_decision": "paper_trade_ready",
                },
            )
            store.append_records(
                "company_snapshots",
                [
                    {
                        "ticker": "CRSP",
                        "company_name": "CRSP Therapeutics",
                        "as_of": now.isoformat(),
                        "market_cap": 1_000_000_000,
                    }
                ],
            )
            store.append_records(
                "labels",
                [{"ticker": "CRSP", "as_of": now.isoformat(), "target_return_90d": 0.1}],
            )
            store.append_records(
                "event_labels",
                [{"ticker": "CRSP", "as_of": now.isoformat(), "target_event_return_10d": 0.04}],
            )
            store.append_records(
                "event_tape",
                [
                    {
                        "ticker": "CRSP",
                        "as_of": now.isoformat(),
                        "event_id": "CRSP:event",
                        "event_type": "phase3_readout",
                        "title": "CRSP topline",
                        "event_timestamp": (now - timedelta(minutes=30)).isoformat(),
                        "status": "exact_press_release",
                        "timing_exact": True,
                        "timing_synthetic": False,
                    }
                ],
            )
            store.write_pipeline_run(
                {
                    "job_name": "archive_vnext",
                    "status": "success",
                    "started_at": now.isoformat(),
                    "finished_at": now.isoformat(),
                    "duration_seconds": 0.0,
                    "metrics": {},
                    "config": {},
                    "notes": "",
                }
            )
            platform = FakePlatform()
            monitor = AutonomyMonitor(
                settings=settings,
                store=store,
                platform=platform,
                broker=NoopBroker(),
                planner=PMExecutionPlanner(settings=settings, store=store),
                trigger_ingestor=NoopTriggerIngestor(),
            )

            with patch("biopharma_agent.vnext.execution._fetch_dollar_adv", return_value=None):
                result = monitor.run_once(submit=False)

            self.assertEqual(result.status, "success")
            self.assertEqual(result.trigger_count, 1)
            self.assertEqual(platform.calls, ["CRSP"])
            self.assertEqual(len(store.read_table("monitor_heartbeats")), 1)
            self.assertEqual(len(store.read_table("trade_decisions")), 1)
            self.assertEqual(
                store.read_table("pipeline_runs").sort_values("finished_at").iloc[-1]["job_name"],
                "monitor_vnext",
            )
            payload = store.read_latest_raw_payload("monitor_runs", "monitor_run_")
            self.assertIsInstance(payload, dict)

    def test_run_once_records_broker_reconciliation_when_configured(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            settings = self.make_settings(tmpdir)
            now = datetime(2026, 4, 5, 14, 0, tzinfo=timezone.utc)
            store.write_raw_payload(
                "validation_audits",
                "latest_walkforward_audit",
                {
                    "generated_at": now.isoformat(),
                    "rows": 180,
                    "windows": 6,
                    "rank_ic": 0.18,
                    "hit_rate": 0.56,
                    "cost_adjusted_top_bottom_spread": 0.11,
                    "leakage_passed": True,
                    "institutional_blockers": [],
                    "promotion_decision": "paper_trade_ready",
                },
            )
            store.append_records(
                "company_snapshots",
                [
                    {
                        "ticker": "CRSP",
                        "company_name": "CRSP Therapeutics",
                        "as_of": now.isoformat(),
                        "market_cap": 1_000_000_000,
                    }
                ],
            )
            store.append_records(
                "labels",
                [{"ticker": "CRSP", "as_of": now.isoformat(), "target_return_90d": 0.1}],
            )
            store.append_records(
                "event_labels",
                [{"ticker": "CRSP", "as_of": now.isoformat(), "target_event_return_10d": 0.04}],
            )
            store.append_records(
                "event_tape",
                [
                    {
                        "ticker": "CRSP",
                        "as_of": now.isoformat(),
                        "event_id": "CRSP:event",
                        "event_type": "phase3_readout",
                        "title": "CRSP topline",
                        "event_timestamp": (now - timedelta(minutes=30)).isoformat(),
                        "status": "exact_press_release",
                        "timing_exact": True,
                        "timing_synthetic": False,
                    }
                ],
            )
            store.write_pipeline_run(
                {
                    "job_name": "archive_vnext",
                    "status": "success",
                    "started_at": now.isoformat(),
                    "finished_at": now.isoformat(),
                    "duration_seconds": 0.0,
                    "metrics": {},
                    "config": {},
                    "notes": "",
                }
            )
            monitor = AutonomyMonitor(
                settings=settings,
                store=store,
                platform=FakePlatform(),
                broker=FakeConfiguredBroker(),
                planner=PMExecutionPlanner(settings=settings, store=store),
                trigger_ingestor=NoopTriggerIngestor(),
            )

            with patch("biopharma_agent.vnext.execution._fetch_dollar_adv", return_value=None):
                result = monitor.run_once(submit=False)

            self.assertEqual(result.status, "success")
            self.assertEqual(len(store.read_table("broker_reconciliations")), 1)
            self.assertEqual(len(store.read_table("portfolio_nav")), 2)

    def test_detect_triggers_uses_live_ingestion_before_scanning_event_tape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            now = datetime(2026, 4, 5, 14, 0, tzinfo=timezone.utc)
            store.append_records(
                "trade_decisions",
                [
                    {
                        "generated_at": now.isoformat(),
                        "symbol": "CRSP",
                        "action": "hold",
                    }
                ],
            )
            ingestor = WritingTriggerIngestor(store)
            monitor = AutonomyMonitor(
                settings=self.make_settings(tmpdir),
                store=store,
                platform=FakePlatform(),
                broker=NoopBroker(),
                planner=PMExecutionPlanner(settings=self.make_settings(tmpdir), store=store),
                trigger_ingestor=ingestor,
            )

            triggers = monitor.detect_triggers(now=now)

            self.assertEqual(ingestor.calls, [["CRSP"]])
            self.assertEqual([item.symbol for item in triggers], ["CRSP"])
            self.assertEqual(triggers[0].source, "event_tape")


if __name__ == "__main__":
    unittest.main()
