import tempfile
import unittest
from unittest.mock import Mock

import pandas as pd

from biopharma_agent.vnext.autonomy import (
    record_trade_decision_run,
    reconcile_broker_state,
    write_autonomy_health_snapshot,
)
from biopharma_agent.vnext.entities import CompanyAnalysis, CompanySnapshot, PortfolioRecommendation, SignalArtifact
from biopharma_agent.vnext.execution import (
    AlpacaPaperBroker,
    DiscordTradeNotifier,
    ExecutionInstruction,
    ExecutionPlan,
    OrderSubmission,
    PMExecutionPlanner,
    execute_plan,
    materialize_execution_feedback,
)
from biopharma_agent.vnext.ops import ReadinessReport
from biopharma_agent.vnext.settings import VNextSettings
from biopharma_agent.vnext.storage import LocalResearchStore


def make_analysis(
    ticker: str,
    scenario: str,
    target_weight: float,
    confidence: float,
    expected_return: float = 0.2,
    company_state: str = "pre_commercial",
    setup_type: str = "hard_catalyst",
    internal_upside_pct: float = 0.25,
    floor_support_pct: float = 0.14,
    catalyst_success_prob: float = 0.7,
    primary_event_type: str = "phase3_readout",
    primary_event_bucket: str = "clinical",
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
        catalyst_success_prob=catalyst_success_prob,
        confidence=confidence,
        crowding_risk=0.2,
        financing_risk=0.2,
        thesis_horizon="90d",
        primary_event_type=primary_event_type,
        primary_event_bucket=primary_event_bucket,
        company_state=company_state,
        setup_type=setup_type,
        internal_upside_pct=internal_upside_pct,
        floor_support_pct=floor_support_pct,
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
        primary_event_type=primary_event_type,
        company_state=company_state,
        setup_type=setup_type,
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
    def __init__(self, blockers=None, status="needs_attention", warnings=None):
        self.blockers = blockers or []
        self.status = status
        self.warnings = warnings or []


class StubHistoryProvider:
    def load_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        dates = pd.to_datetime(["2025-01-01", "2025-01-11", "2025-01-31", "2025-04-01"])
        return pd.DataFrame({"close": [100.0, 105.0, 115.0, 125.0]}, index=dates)


class PlannedAtPrecedenceHistoryProvider:
    def load_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        dates = pd.to_datetime(["2024-01-01", "2025-01-01", "2025-01-11", "2025-01-31", "2025-04-01"])
        return pd.DataFrame({"close": [80.0, 100.0, 105.0, 115.0, 125.0]}, index=dates)


class RiskOffBenchmarkProvider:
    def load_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        dates = pd.to_datetime(["2026-01-01", "2026-02-15", "2026-04-01"])
        return pd.DataFrame({"close": [100.0, 88.0, 80.0]}, index=dates)


class FakeReconcileBroker:
    def account(self) -> DummyAccount:
        return DummyAccount()

    def positions(self) -> list[DummyPosition]:
        return [DummyPosition("CRSP", qty=50.0, market_value=1500.0, current_price=30.0)]

    def recent_orders(self, limit: int = 200) -> list[dict[str, object]]:
        return [
            {
                "id": "order-1",
                "client_order_id": "coid-1",
                "symbol": "CRSP",
                "status": "filled",
                "filled_qty": "50",
                "filled_avg_price": "30",
                "submitted_at": "2025-01-01T10:00:00+00:00",
                "updated_at": "2025-01-01T10:01:00+00:00",
                "filled_at": "2025-01-01T10:01:00+00:00",
            }
        ]


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

    def test_store_recovers_from_corrupt_predictions_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            corrupt_path = store.tables_dir / "predictions.parquet"
            corrupt_path.write_bytes(b"not-a-parquet-file")

            store.append_records("predictions", [{"ticker": "CRSP", "expected_return": 0.12}])

            recovered = store.read_table("predictions")
            self.assertEqual(len(recovered), 1)
            self.assertTrue((store.tables_dir / "predictions.corrupt.parquet").exists())

    def test_planner_keeps_existing_position_as_holdover(self):
        settings = self.make_settings()
        planner = PMExecutionPlanner(settings)
        analyses = [
            make_analysis(
                "CRSP",
                "pairs candidate",
                1.2,
                0.56,
                expected_return=0.09,
                setup_type="asymmetry_without_near_term_catalyst",
                internal_upside_pct=0.18,
                floor_support_pct=0.22,
            )
        ]
        account = DummyAccount()
        positions = [DummyPosition("CRSP", qty=40.0, market_value=1200.0, current_price=30.0)]
        plan = planner.build_plan(analyses, account, positions, DummyReadiness(blockers=[]))

        instruction = next(item for item in plan.instructions if item.symbol == "CRSP")
        self.assertEqual(instruction.action, "hold")
        self.assertIn("CRSP", plan.selected_symbols)
        self.assertEqual(instruction.execution_profile, "precommercial_asymmetry_hold")

    def test_planner_uses_rebalance_band_for_small_deltas(self):
        settings = self.make_settings()
        planner = PMExecutionPlanner(settings)
        analyses = [make_analysis("CRSP", "pre-catalyst long", 4.0, 0.72)]
        account = DummyAccount()
        positions = [DummyPosition("CRSP", qty=126.6667, market_value=3800.0, current_price=30.0)]
        plan = planner.build_plan(analyses, account, positions, DummyReadiness(blockers=[]))

        instruction = next(item for item in plan.instructions if item.symbol == "CRSP")
        self.assertEqual(instruction.action, "hold")

    def test_precommercial_asymmetry_without_catalyst_is_not_auto_deployed(self):
        settings = self.make_settings()
        planner = PMExecutionPlanner(settings)
        analyses = [
            make_analysis(
                "VKTX",
                "watchlist only",
                1.0,
                0.66,
                expected_return=0.14,
                setup_type="asymmetry_without_near_term_catalyst",
                internal_upside_pct=0.20,
                floor_support_pct=0.18,
                primary_event_type="thematic_repricing",
                primary_event_bucket="commercial",
            )
        ]

        plan = planner.build_plan(analyses, DummyAccount(), [], DummyReadiness(blockers=[]))

        self.assertFalse(plan.selected_symbols)
        self.assertIn("No recommendations cleared the PM execution thresholds.", plan.blockers)

    def test_commercialized_capital_allocation_is_size_capped(self):
        settings = self.make_settings()
        planner = PMExecutionPlanner(settings)
        analyses = [
            make_analysis(
                "REGN",
                "commercial compounder",
                5.0,
                0.63,
                expected_return=0.12,
                company_state="commercialized",
                setup_type="capital_allocation",
                internal_upside_pct=0.14,
                floor_support_pct=0.18,
                primary_event_type="capital_allocation",
                primary_event_bucket="commercial",
            )
        ]

        plan = planner.build_plan(analyses, DummyAccount(), [], DummyReadiness(blockers=[]))

        instruction = next(item for item in plan.instructions if item.symbol == "REGN")
        self.assertEqual(instruction.action, "buy")
        self.assertEqual(instruction.execution_profile, "capital_allocation")
        self.assertLessEqual(instruction.scaled_target_weight, settings.execution_max_franchise_weight_pct)

    def test_negative_internal_upside_blocks_entry(self):
        settings = self.make_settings()
        planner = PMExecutionPlanner(settings)
        analyses = [
            make_analysis(
                "BEAM",
                "pre-catalyst long",
                3.0,
                0.71,
                expected_return=0.11,
                internal_upside_pct=-0.12,
            )
        ]

        plan = planner.build_plan(analyses, DummyAccount(), [], DummyReadiness(blockers=[]))

        self.assertFalse(plan.selected_symbols)
        self.assertIn("No recommendations cleared the PM execution thresholds.", plan.blockers)

    def test_pending_transaction_is_not_auto_deployed(self):
        settings = self.make_settings()
        planner = PMExecutionPlanner(settings)
        analysis = make_analysis(
            "APLS",
            "commercial compounder",
            3.0,
            0.82,
            expected_return=0.18,
            company_state="commercialized",
            setup_type="pipeline_optionality",
            internal_upside_pct=0.16,
            floor_support_pct=0.15,
            primary_event_type="strategic_transaction",
            primary_event_bucket="strategic",
        )
        analysis.snapshot.metadata["special_situation"] = "pending_transaction"
        analysis.snapshot.metadata["special_situation_reason"] = (
            "Announced Biogen acquisition caps standalone upside largely to the cash consideration, CVR, and deal spread."
        )

        plan = planner.build_plan([analysis], DummyAccount(), [], DummyReadiness(blockers=[]))

        self.assertFalse(plan.selected_symbols)
        self.assertIn("No recommendations cleared the PM execution thresholds.", plan.blockers)

    def test_validation_overlay_blocks_negative_setup_regime_for_new_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            store.write_raw_payload(
                "validation_audits",
                "latest_walkforward_audit",
                {
                    "rolling_setup_regime_scorecards": {
                        "launch_asymmetry|negative_momentum": {
                            "rows": 41.0,
                            "windows": 6.0,
                            "rank_ic": -0.31,
                            "hit_rate": 0.50,
                            "beta_adjusted_return": -0.17,
                            "cost_adjusted_top_bottom_spread": -0.18,
                        }
                    }
                },
            )
            planner = PMExecutionPlanner(self.make_settings(), store=store)
            analysis = make_analysis(
                "ARVN",
                "commercial compounder",
                4.0,
                0.72,
                expected_return=0.14,
                company_state="commercial_launch",
                setup_type="launch_asymmetry",
                internal_upside_pct=0.18,
                floor_support_pct=0.16,
                primary_event_type="commercial_update",
                primary_event_bucket="commercial",
            )
            analysis.snapshot.momentum_3mo = -0.20

            plan = planner.build_plan([analysis], DummyAccount(), [], DummyReadiness(blockers=[]))

            self.assertFalse(plan.selected_symbols)
            self.assertIn("No recommendations cleared the PM execution thresholds.", plan.blockers)

    def test_validation_overlay_scales_down_weak_but_positive_combo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            store.write_raw_payload(
                "validation_audits",
                "latest_walkforward_audit",
                {
                    "rolling_setup_regime_scorecards": {
                        "capital_allocation|neutral_momentum": {
                            "rows": 16.0,
                            "windows": 6.0,
                            "rank_ic": 0.13,
                            "hit_rate": 0.4167,
                            "beta_adjusted_return": 0.05,
                            "cost_adjusted_top_bottom_spread": 0.045,
                        }
                    }
                },
            )
            planner = PMExecutionPlanner(self.make_settings(), store=store)
            analysis = make_analysis(
                "REGN",
                "commercial compounder",
                4.0,
                0.72,
                expected_return=0.12,
                company_state="commercialized",
                setup_type="capital_allocation",
                internal_upside_pct=0.16,
                floor_support_pct=0.18,
                primary_event_type="capital_allocation",
                primary_event_bucket="commercial",
            )
            analysis.snapshot.momentum_3mo = 0.0

            plan = planner.build_plan([analysis], DummyAccount(), [], DummyReadiness(blockers=[]))

            instruction = next(item for item in plan.instructions if item.symbol == "REGN")
            self.assertEqual(instruction.action, "buy")
            self.assertLessEqual(instruction.scaled_target_weight, self.make_settings().execution_max_floor_weight_pct)

    def test_stale_validation_blocks_new_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            store.write_raw_payload(
                "validation_audits",
                "latest_walkforward_audit",
                {
                    "generated_at": "2024-01-01T00:00:00+00:00",
                    "rolling_setup_regime_scorecards": {
                        "hard_catalyst|positive_momentum": {
                            "rows": 48.0,
                            "windows": 8.0,
                            "rank_ic": 0.24,
                            "hit_rate": 0.64,
                            "beta_adjusted_return": 0.11,
                            "cost_adjusted_top_bottom_spread": 0.18,
                        }
                    },
                },
            )
            planner = PMExecutionPlanner(self.make_settings(), store=store)
            analysis = make_analysis("ARVN", "pre-catalyst long", 4.0, 0.82)

            plan = planner.build_plan([analysis], DummyAccount(), [], DummyReadiness(blockers=[]))

            self.assertFalse(plan.selected_symbols)
            self.assertIn("No recommendations cleared the PM execution thresholds.", plan.blockers)
            self.assertTrue(any("Validation audit is" in warning for warning in plan.warnings))

    def test_exposure_governor_scales_book_in_risk_off_biotech_tape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            store.write_raw_payload(
                "validation_audits",
                "latest_walkforward_audit",
                {"generated_at": "2026-04-05T00:00:00+00:00", "promotion_decision": "paper_trade_ready"},
            )
            planner = PMExecutionPlanner(
                self.make_settings(),
                store=store,
                market_history_provider=RiskOffBenchmarkProvider(),
            )
            analysis = make_analysis("CRSP", "pre-catalyst long", 4.0, 0.82)
            analysis.snapshot.momentum_3mo = -0.18
            analysis.snapshot.volatility = 0.65

            plan = planner.build_plan([analysis], DummyAccount(), [], DummyReadiness(blockers=[]))

            instruction = next(item for item in plan.instructions if item.symbol == "CRSP")
            self.assertLess(plan.gross_cap_multiplier, 1.0)
            self.assertLess(instruction.scaled_target_weight, instruction.target_weight)
            self.assertTrue(any("Exposure governor trimmed max gross" in warning for warning in plan.warnings))
            self.assertTrue(any("regime_governor" in item for item in instruction.rationale))

    def test_execution_feedback_materializes_profile_scorecards(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            store.append_records(
                "order_plans",
                [
                    {
                        "symbol": "CRSP",
                        "company_name": "CRSP Therapeutics",
                        "action": "buy",
                        "side": "buy",
                        "scenario": "pre-catalyst long",
                        "company_state": "pre_commercial",
                        "setup_type": "hard_catalyst",
                        "execution_profile": "hard_catalyst",
                        "confidence": 0.72,
                        "target_weight": 4.0,
                        "scaled_target_weight": 4.0,
                        "target_notional": 1_000.0,
                        "current_notional": 0.0,
                        "delta_notional": 1_000.0,
                        "as_of": "2025-01-01T00:00:00",
                        "planned_at": "2025-01-01T00:00:00",
                    }
                ],
            )
            summary = materialize_execution_feedback(store=store, history_provider=StubHistoryProvider())
            feedback = store.read_table("execution_feedback")
            scorecards = store.read_table("execution_profile_scorecards")

            self.assertEqual(summary.feedback_rows, 1)
            self.assertEqual(len(feedback), 1)
            self.assertEqual(scorecards.iloc[0]["execution_profile"], "hard_catalyst")
            self.assertGreater(float(scorecards.iloc[0]["avg_return_30d"]), 0.0)

    def test_execution_feedback_prefers_planned_at_and_records_net_returns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            store.append_records(
                "order_plans",
                [
                    {
                        "symbol": "CRSP",
                        "company_name": "CRSP Therapeutics",
                        "action": "buy",
                        "side": "buy",
                        "scenario": "pre-catalyst long",
                        "company_state": "pre_commercial",
                        "setup_type": "hard_catalyst",
                        "execution_profile": "hard_catalyst",
                        "confidence": 0.72,
                        "target_weight": 4.0,
                        "scaled_target_weight": 4.0,
                        "target_notional": 1_000.0,
                        "current_notional": 0.0,
                        "delta_notional": 1_000.0,
                        "as_of": "2024-01-01T00:00:00",
                        "planned_at": "2025-01-01T00:00:00",
                    }
                ],
            )

            materialize_execution_feedback(store=store, history_provider=PlannedAtPrecedenceHistoryProvider())
            feedback = store.read_table("execution_feedback")
            scorecards = store.read_table("execution_profile_scorecards")

            self.assertEqual(feedback.iloc[0]["entry_anchor_source"], "planned_at")
            self.assertAlmostEqual(float(feedback.iloc[0]["entry_price"]), 100.0)
            self.assertLess(
                float(feedback.iloc[0]["return_10d_net"]),
                float(feedback.iloc[0]["return_10d"]),
            )
            self.assertLess(
                float(scorecards.iloc[0]["avg_net_return_30d"]),
                float(scorecards.iloc[0]["avg_return_30d"]),
            )

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
                company_state="pre_commercial",
                setup_type="hard_catalyst",
                execution_profile="hard_catalyst",
                confidence=0.72,
                target_weight=4.0,
                scaled_target_weight=4.0,
                target_notional=4000.0,
                current_notional=0.0,
                delta_notional=1500.0,
                internal_upside_pct=0.25,
                floor_support_pct=0.14,
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

    def test_reconcile_broker_state_persists_summary_and_nav(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            plan = ExecutionPlan(
                generated_at="2025-01-01T00:00:00+00:00",
                account_id="PA3ZUXE6OCWI",
                equity=100000.0,
                buying_power=100000.0,
                deployable_notional=18000.0,
                selected_symbols=["CRSP"],
                instructions=[
                    ExecutionInstruction(
                        symbol="CRSP",
                        company_name="CRSP Therapeutics",
                        action="buy",
                        side="buy",
                        scenario="pre-catalyst long",
                        company_state="pre_commercial",
                        setup_type="hard_catalyst",
                        execution_profile="hard_catalyst",
                        confidence=0.72,
                        target_weight=4.0,
                        scaled_target_weight=4.0,
                        target_notional=4000.0,
                        current_notional=0.0,
                        delta_notional=1500.0,
                        internal_upside_pct=0.25,
                        floor_support_pct=0.14,
                        notional=1500.0,
                        rationale=[],
                    )
                ],
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

            summary = reconcile_broker_state(
                store=store,
                broker=FakeReconcileBroker(),
                plan=plan,
                submissions=submissions,
            )

            reconciliations = store.read_table("broker_reconciliations")
            nav = store.read_table("portfolio_nav")
            order_updates = store.read_table("broker_order_updates")

            self.assertEqual(summary.filled_order_count, 1)
            self.assertEqual(len(reconciliations), 1)
            self.assertEqual(len(nav), 1)
            self.assertEqual(len(order_updates), 1)

    def test_record_trade_decision_run_persists_audit_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            store.write_raw_payload(
                "validation_audits",
                "latest_walkforward_audit",
                {"generated_at": "2026-04-05T00:00:00+00:00", "promotion_decision": "paper_trade_ready"},
            )
            analysis = make_analysis("CRSP", "pre-catalyst long", 4.0, 0.72)
            plan = ExecutionPlan(
                generated_at="2025-01-01T00:00:00+00:00",
                account_id="PA3ZUXE6OCWI",
                equity=100000.0,
                buying_power=100000.0,
                deployable_notional=18000.0,
                selected_symbols=["CRSP"],
                instructions=[
                    ExecutionInstruction(
                        symbol="CRSP",
                        company_name="CRSP Therapeutics",
                        action="buy",
                        side="buy",
                        scenario="pre-catalyst long",
                        company_state="pre_commercial",
                        setup_type="hard_catalyst",
                        execution_profile="hard_catalyst",
                        confidence=0.72,
                        target_weight=4.0,
                        scaled_target_weight=3.0,
                        target_notional=3000.0,
                        current_notional=0.0,
                        delta_notional=3000.0,
                        as_of="2025-01-01T00:00:00+00:00",
                        internal_upside_pct=0.25,
                        floor_support_pct=0.14,
                        notional=3000.0,
                        rationale=["alpha thesis"],
                    )
                ],
                blockers=[],
                warnings=[],
                readiness_status="production_ready",
                gross_cap_pct=12.0,
                gross_cap_multiplier=0.75,
                exposure_governor={"validation_decision": "paper_trade_ready"},
            )

            record_trade_decision_run(
                store=store,
                plan=plan,
                analyses=[analysis],
                readiness=DummyReadiness(blockers=[], status="production_ready"),
                settings=self.make_settings(),
                account=DummyAccount(),
                submit_requested=False,
                submit_attempted=False,
                submissions=[],
                reconciliation=None,
            )

            decisions = store.read_table("trade_decisions")
            payload = store.read_latest_raw_payload("trade_decision_runs", "trade_decision_run_")

            self.assertEqual(len(decisions), 1)
            self.assertEqual(decisions.iloc[0]["validation_decision"], "paper_trade_ready")
            self.assertIsInstance(payload, dict)
            self.assertEqual(payload["plan"]["gross_cap_multiplier"], 0.75)

    def test_write_autonomy_health_snapshot_flags_failed_trade_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalResearchStore(base_dir=tmpdir)
            store.append_records(
                "company_snapshots",
                [{"ticker": "BROKEN", "as_of": "2026-04-05T00:00:00+00:00", "market_cap": 0.0}],
            )
            store.write_pipeline_run(
                {
                    "job_name": "trade_vnext",
                    "status": "failed",
                    "started_at": "2026-04-05T09:00:00+00:00",
                    "finished_at": "2026-04-05T09:01:00+00:00",
                    "duration_seconds": 60.0,
                    "metrics": {},
                    "config": {},
                    "notes": "simulated failure",
                }
            )
            readiness = ReadinessReport(
                status="production_ready",
                generated_at="2026-04-05T10:00:00+00:00",
                store_dir=tmpdir,
                eodhd_configured=True,
                sec_user_agent_configured=True,
                snapshot_rows=1,
                distinct_snapshot_dates=12,
                latest_snapshot_age_hours=1.0,
                label_rows=120,
                event_label_rows=60,
                matured_return_90d_rows=120,
                matured_event_rows=20,
                archive_run_count=5,
                successful_archive_runs=5,
                backfill_run_count=2,
                successful_backfill_runs=2,
                evaluate_run_count=3,
                successful_evaluate_runs=3,
                eodhd_cache_files=10,
                walkforward_rows=120,
                walkforward_windows=6,
                leakage_passed=True,
                blockers=[],
                warnings=[],
                evaluation_message="ok",
            )

            payload = write_autonomy_health_snapshot(store=store, settings=self.make_settings(), readiness=readiness)

            self.assertEqual(payload["status"], "blocked")
            self.assertEqual(payload["zero_market_cap_active"], 1)
            self.assertEqual(payload["latest_trade_run"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
