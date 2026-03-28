import tempfile
import unittest
from unittest.mock import Mock

from biopharma_agent.vnext.entities import CompanyAnalysis, CompanySnapshot, PortfolioRecommendation, SignalArtifact
from biopharma_agent.vnext.execution import (
    AlpacaPaperBroker,
    DiscordTradeNotifier,
    ExecutionInstruction,
    ExecutionPlan,
    OrderSubmission,
    PMExecutionPlanner,
    execute_plan,
)
from biopharma_agent.vnext.settings import VNextSettings
from biopharma_agent.vnext.storage import LocalResearchStore


def make_analysis(
    ticker: str,
    scenario: str,
    target_weight: float,
    confidence: float,
    expected_return: float = 0.2,
) -> CompanyAnalysis:
    snapshot = CompanySnapshot(
        ticker=ticker,
        company_name=f"{ticker} Therapeutics",
        as_of="2025-01-01T00:00:00+00:00",
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
        expected_return=expected_return,
        catalyst_success_prob=0.7,
        confidence=confidence,
        crowding_risk=0.2,
        financing_risk=0.2,
        thesis_horizon="90d",
        primary_event_type="phase3_readout",
        primary_event_bucket="clinical",
        rationale=[],
        supporting_evidence=[],
    )
    rec = PortfolioRecommendation(
        ticker=ticker,
        as_of=snapshot.as_of,
        stance="long" if target_weight > 0 else "avoid",
        target_weight=target_weight,
        max_weight=6.0,
        confidence=confidence,
        scenario=scenario,
        thesis_horizon="90d",
        primary_event_type="phase3_readout",
        risk_flags=[],
    )
    return CompanyAnalysis(
        snapshot=snapshot,
        signal=signal,
        portfolio=rec,
        feature_vectors=[],
        program_predictions=[],
    )


class DummyAccount:
    def __init__(self):
        self.account_id = "PA-1"
        self.status = "ACTIVE"
        self.equity = 100000.0
        self.buying_power = 100000.0
        self.cash = 100000.0
        self.paper = True
        self.trading_blocked = False
        self.account_blocked = False
        self.pattern_day_trader = False


class DummyPosition:
    def __init__(self, symbol: str, qty: float, market_value: float, current_price: float):
        self.symbol = symbol
        self.qty = qty
        self.market_value = market_value
        self.current_price = current_price
        self.side = "long"


class DummyReadiness:
    def __init__(self, blockers=None, status="needs_attention"):
        self.blockers = blockers or []
        self.status = status


class TestVNextExecution(unittest.TestCase):
    def make_settings(self) -> VNextSettings:
        return VNextSettings(
            store_dir=".tatetuck_store",
            eodhd_api_key="secret",
            sec_user_agent="TatetuckBot/1.0 ops@example.com",
            discord_token="discord-token",
            discord_channel_id="general-channel",
            discord_trade_log_channel_id="trade-log-channel",
            alpaca_api_key_id="paper-key",
            alpaca_api_secret_key="paper-secret",
            alpaca_api_base_url="https://paper-api.alpaca.markets",
            alpaca_paper_account_id="PA3ZUXE6OCWI",
            include_literature=False,
            min_snapshot_dates=10,
            min_matured_return_rows=100,
            min_walkforward_windows=3,
            max_snapshot_age_hours=36,
            min_archive_runs=1,
            max_gross_exposure_pct=18.0,
            max_single_position_pct=4.0,
            min_execution_weight_pct=1.0,
            min_execution_confidence=0.60,
            min_order_notional=150.0,
            max_new_positions=6,
            execution_hold_weight_pct=0.75,
            execution_hold_confidence=0.50,
            execution_rebalance_band_pct=0.75,
            evaluation_rebalance_spacing_days=21,
            evaluation_min_names_per_window=3,
            evaluation_turnover_book_weight_floor=1.0,
            allow_blocked_paper_trading=True,
        )

    def test_planner_creates_buy_and_exit_instructions(self):
        settings = self.make_settings()
        planner = PMExecutionPlanner(settings)
        analyses = [
            make_analysis("CRSP", "pre-catalyst long", 4.0, 0.72),
            make_analysis("BEAM", "watchlist only", 0.5, 0.55),
        ]
        account = DummyAccount()
        positions = [DummyPosition("BEAM", qty=10.0, market_value=500.0, current_price=50.0)]
        plan = planner.build_plan(analyses, account, positions, DummyReadiness(blockers=[]))

        buy = next(item for item in plan.instructions if item.symbol == "CRSP")
        exit_beam = next(item for item in plan.instructions if item.symbol == "BEAM")
        self.assertEqual(buy.action, "buy")
        self.assertEqual(exit_beam.action, "sell")

    def test_execute_plan_dry_run_does_not_call_broker(self):
        settings = self.make_settings()
        planner = PMExecutionPlanner(settings)
        analyses = [make_analysis("CRSP", "pre-catalyst long", 4.0, 0.72)]
        plan = planner.build_plan(analyses, DummyAccount(), [], DummyReadiness(blockers=[]))
        broker = Mock()
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            submissions = execute_plan(plan=plan, broker=broker, store=store, submit=False)
            broker.submit_market_notional_buy.assert_not_called()
            self.assertTrue(submissions)
            self.assertEqual(submissions[0].status, "planned")

    def test_planner_keeps_existing_position_as_holdover(self):
        settings = self.make_settings()
        planner = PMExecutionPlanner(settings)
        analyses = [make_analysis("CRSP", "pairs candidate", 1.2, 0.56, expected_return=0.09)]
        account = DummyAccount()
        positions = [DummyPosition("CRSP", qty=40.0, market_value=1200.0, current_price=30.0)]
        plan = planner.build_plan(analyses, account, positions, DummyReadiness(blockers=[]))

        instruction = next(item for item in plan.instructions if item.symbol == "CRSP")
        self.assertEqual(instruction.action, "hold")
        self.assertIn("CRSP", plan.selected_symbols)

    def test_planner_uses_rebalance_band_for_small_deltas(self):
        settings = self.make_settings()
        planner = PMExecutionPlanner(settings)
        analyses = [make_analysis("CRSP", "pre-catalyst long", 4.0, 0.72)]
        account = DummyAccount()
        positions = [DummyPosition("CRSP", qty=126.6667, market_value=3800.0, current_price=30.0)]
        plan = planner.build_plan(analyses, account, positions, DummyReadiness(blockers=[]))

        instruction = next(item for item in plan.instructions if item.symbol == "CRSP")
        self.assertEqual(instruction.action, "hold")

    def test_alpaca_broker_uses_paper_headers(self):
        settings = self.make_settings()
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"id": "order-1", "status": "accepted"}
        session = Mock()
        session.request.return_value = response
        broker = AlpacaPaperBroker(settings=settings, session=session)
        broker.submit_market_notional_buy("CRSP", 500.0)
        call = session.request.call_args
        self.assertEqual(call.kwargs["headers"]["APCA-API-KEY-ID"], "paper-key")
        self.assertIn("/v2/orders", call.args[1])

    def test_simulated_account_uses_expected_paper_id(self):
        settings = self.make_settings()
        settings.alpaca_api_key_id = None
        settings.alpaca_api_secret_key = None
        broker = AlpacaPaperBroker(settings=settings)
        account = broker.simulated_account()
        self.assertEqual(account.account_id, "PA3ZUXE6OCWI")
        self.assertEqual(account.status, "SIMULATED")
        self.assertTrue(account.paper)

    def test_discord_trade_notifier_posts_submitted_orders(self):
        settings = self.make_settings()
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"id": "discord-message-1"}
        session = Mock()
        session.request.return_value = response
        notifier = DiscordTradeNotifier(settings=settings, session=session)
        plan = ExecutionPlan(
            generated_at="2025-01-01T00:00:00+00:00",
            account_id="PA3ZUXE6OCWI",
            equity=100000.0,
            buying_power=100000.0,
            deployable_notional=18000.0,
            selected_symbols=["CRSP"],
            instructions=[],
            blockers=[],
            warnings=[],
            readiness_status="production_ready",
        )
        submissions = [
            OrderSubmission(
                symbol="CRSP",
                action="buy_notional",
                status="submitted",
                client_order_id="coid-1",
                order_id="order-1",
                submitted_notional=1500.0,
                submitted_qty=None,
                raw_status="accepted",
            )
        ]
        instructions = [
            ExecutionInstruction(
                symbol="CRSP",
                company_name="CRSP Therapeutics",
                action="buy",
                side="buy",
                scenario="pre-catalyst long",
                confidence=0.72,
                target_weight=4.0,
                scaled_target_weight=4.0,
                target_notional=4000.0,
                current_notional=0.0,
                delta_notional=1500.0,
                qty=None,
                notional=1500.0,
                rationale=[],
            )
        ]

        result = notifier.post_trade_alert(plan=plan, submissions=submissions, instructions=instructions)

        self.assertIsNotNone(result)
        self.assertEqual(result.channel_id, "trade-log-channel")
        self.assertFalse(result.fallback_used)
        call = session.request.call_args
        self.assertIn("/channels/trade-log-channel/messages", call.args[1])
        self.assertIn("TATETUCK PAPER TRADE ALERT", call.kwargs["json"]["content"])
        self.assertIn("CRSP", call.kwargs["json"]["content"])

    def test_discord_trade_notifier_falls_back_to_primary_channel(self):
        settings = self.make_settings()
        def request_side_effect(method, url, headers=None, json=None, timeout=None):
            response = Mock()
            if "trade-log-channel" in url:
                error = requests.HTTPError("forbidden")
                error.response = Mock(status_code=403)
                raise error
            response.raise_for_status.return_value = None
            response.json.return_value = {"id": "discord-message-2"}
            return response

        import requests

        session = Mock()
        session.request.side_effect = request_side_effect
        notifier = DiscordTradeNotifier(settings=settings, session=session)
        plan = ExecutionPlan(
            generated_at="2025-01-01T00:00:00+00:00",
            account_id="PA3ZUXE6OCWI",
            equity=100000.0,
            buying_power=100000.0,
            deployable_notional=18000.0,
            selected_symbols=["CRSP"],
            instructions=[],
            blockers=[],
            warnings=[],
            readiness_status="production_ready",
        )
        submissions = [
            OrderSubmission(
                symbol="CRSP",
                action="buy_notional",
                status="submitted",
                client_order_id="coid-1",
                order_id="order-1",
                submitted_notional=1500.0,
                submitted_qty=None,
                raw_status="accepted",
            )
        ]

        result = notifier.post_trade_alert(plan=plan, submissions=submissions, instructions=[])

        self.assertIsNotNone(result)
        self.assertEqual(result.channel_id, "general-channel")
        self.assertTrue(result.fallback_used)


if __name__ == "__main__":
    unittest.main()
