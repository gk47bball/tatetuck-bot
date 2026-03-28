from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import os
from typing import Iterable
from uuid import uuid4

import pandas as pd
import requests

from .entities import CompanyAnalysis, PortfolioRecommendation
from .labels import CompositeHistoryProvider, EODHDHistoryProvider, PriceHistoryProvider, YFinanceHistoryProvider
from .ops import ReadinessReport, record_pipeline_run, utc_now_iso
from .settings import VNextSettings
from .storage import LocalResearchStore


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class BrokerAccount:
    account_id: str
    status: str
    equity: float
    buying_power: float
    cash: float
    paper: bool
    trading_blocked: bool
    account_blocked: bool
    pattern_day_trader: bool


@dataclass(slots=True)
class BrokerPosition:
    symbol: str
    qty: float
    market_value: float
    current_price: float
    side: str


@dataclass(slots=True)
class ExecutionInstruction:
    symbol: str
    company_name: str
    action: str
    side: str
    scenario: str
    company_state: str | None
    setup_type: str | None
    execution_profile: str
    confidence: float
    target_weight: float
    scaled_target_weight: float
    target_notional: float
    current_notional: float
    delta_notional: float
    as_of: str | None = None
    internal_upside_pct: float | None = None
    floor_support_pct: float | None = None
    qty: float | None = None
    notional: float | None = None
    rationale: list[str] = field(default_factory=list)

    def to_row(self) -> dict[str, object]:
        payload = asdict(self)
        payload["planned_at"] = datetime.now(timezone.utc).isoformat()
        return payload


@dataclass(slots=True)
class ExecutionProfile:
    name: str
    mode: str
    weight_cap_pct: float
    score: float
    rationale: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExecutionPlan:
    generated_at: str
    account_id: str | None
    equity: float
    buying_power: float
    deployable_notional: float
    selected_symbols: list[str]
    instructions: list[ExecutionInstruction]
    blockers: list[str]
    warnings: list[str]
    readiness_status: str


@dataclass(slots=True)
class OrderSubmission:
    symbol: str
    action: str
    status: str
    client_order_id: str
    order_id: str | None
    submitted_notional: float | None
    submitted_qty: float | None
    raw_status: str | None = None
    notes: str = ""

    def to_row(self) -> dict[str, object]:
        payload = asdict(self)
        payload["submitted_at"] = datetime.now(timezone.utc).isoformat()
        return payload


@dataclass(slots=True)
class DiscordNotificationResult:
    channel_id: str
    message_id: str | None
    order_count: int
    fallback_used: bool = False


@dataclass(slots=True)
class ExecutionFeedbackSummary:
    feedback_rows: int
    matured_30d_rows: int
    matured_90d_rows: int
    scorecard_rows: int


class AlpacaPaperBroker:
    def __init__(
        self,
        settings: VNextSettings,
        session: requests.Session | None = None,
    ):
        self.settings = settings
        self.session = session or requests.Session()
        self.base_url = settings.alpaca_api_base_url.rstrip("/")

    def is_configured(self) -> bool:
        return bool(self.settings.alpaca_api_key_id and self.settings.alpaca_api_secret_key)

    def ensure_paper_only(self) -> None:
        if "paper-api.alpaca.markets" not in self.base_url:
            raise RuntimeError(f"Refusing to trade against non-paper Alpaca endpoint: {self.base_url}")

    def simulated_account(self) -> BrokerAccount:
        equity = max(float(self.settings.simulated_paper_equity), 0.0)
        account_id = self.settings.alpaca_paper_account_id or "SIMULATED-PAPER"
        return BrokerAccount(
            account_id=account_id,
            status="SIMULATED",
            equity=equity,
            buying_power=equity,
            cash=equity,
            paper=True,
            trading_blocked=False,
            account_blocked=False,
            pattern_day_trader=False,
        )

    def ensure_expected_account(self, account: BrokerAccount) -> None:
        expected = (self.settings.alpaca_paper_account_id or "").strip()
        actual = (account.account_id or "").strip()
        if expected and actual and expected != actual:
            raise RuntimeError(
                f"Configured APCA_PAPER_ACCOUNT_ID={expected} does not match broker account {actual}."
            )

    def account(self) -> BrokerAccount:
        data = self._request("GET", "/v2/account")
        return BrokerAccount(
            account_id=str(data.get("account_number") or data.get("id") or ""),
            status=str(data.get("status") or "UNKNOWN"),
            equity=_to_float(data.get("equity")),
            buying_power=_to_float(data.get("buying_power")),
            cash=_to_float(data.get("cash")),
            paper="paper" in self.base_url,
            trading_blocked=bool(data.get("trading_blocked", False)),
            account_blocked=bool(data.get("account_blocked", False)),
            pattern_day_trader=bool(data.get("pattern_day_trader", False)),
        )

    def positions(self) -> list[BrokerPosition]:
        data = self._request("GET", "/v2/positions")
        positions: list[BrokerPosition] = []
        for item in data:
            positions.append(
                BrokerPosition(
                    symbol=str(item.get("symbol")),
                    qty=_to_float(item.get("qty")),
                    market_value=abs(_to_float(item.get("market_value"))),
                    current_price=_to_float(item.get("current_price")),
                    side=str(item.get("side") or "long"),
                )
            )
        return positions

    def submit_market_notional_buy(self, symbol: str, notional: float) -> OrderSubmission:
        client_order_id = self._client_order_id(symbol, "buy")
        data = self._request(
            "POST",
            "/v2/orders",
            json={
                "symbol": symbol,
                "notional": round(notional, 2),
                "side": "buy",
                "type": "market",
                "time_in_force": "day",
                "client_order_id": client_order_id,
            },
        )
        return OrderSubmission(
            symbol=symbol,
            action="buy_notional",
            status="submitted",
            client_order_id=client_order_id,
            order_id=data.get("id"),
            submitted_notional=round(notional, 2),
            submitted_qty=None,
            raw_status=data.get("status"),
        )

    def submit_market_qty_sell(self, symbol: str, qty: float) -> OrderSubmission:
        client_order_id = self._client_order_id(symbol, "sell")
        data = self._request(
            "POST",
            "/v2/orders",
            json={
                "symbol": symbol,
                "qty": round(qty, 6),
                "side": "sell",
                "type": "market",
                "time_in_force": "day",
                "client_order_id": client_order_id,
            },
        )
        return OrderSubmission(
            symbol=symbol,
            action="sell_qty",
            status="submitted",
            client_order_id=client_order_id,
            order_id=data.get("id"),
            submitted_notional=None,
            submitted_qty=round(qty, 6),
            raw_status=data.get("status"),
        )

    def _request(self, method: str, path: str, json: dict | None = None):
        if not self.is_configured():
            raise RuntimeError("Alpaca paper trading is not configured. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY.")
        self.ensure_paper_only()
        response = self.session.request(
            method,
            f"{self.base_url}{path}",
            headers={
                "APCA-API-KEY-ID": self.settings.alpaca_api_key_id or "",
                "APCA-API-SECRET-KEY": self.settings.alpaca_api_secret_key or "",
            },
            json=json,
            timeout=20,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _client_order_id(symbol: str, side: str) -> str:
        suffix = uuid4().hex[:12]
        return f"tatetuck-{side}-{symbol.lower()}-{suffix}"[:48]


class DiscordTradeNotifier:
    def __init__(
        self,
        settings: VNextSettings,
        session: requests.Session | None = None,
    ):
        self.settings = settings
        self.session = session or requests.Session()
        self.base_url = "https://discord.com/api/v10"

    def is_configured(self) -> bool:
        return bool(self.settings.discord_token and self._target_channels())

    def post_trade_alert(
        self,
        plan: ExecutionPlan,
        submissions: list[OrderSubmission],
        instructions: list[ExecutionInstruction],
    ) -> DiscordNotificationResult | None:
        submitted = [item for item in submissions if item.status == "submitted"]
        if not submitted or not self.is_configured():
            return None

        channel_ids = self._target_channels()
        if not channel_ids:
            return None

        content = self._build_message(plan=plan, submissions=submitted, instructions=instructions)
        first_channel = channel_ids[0]
        last_exc: Exception | None = None
        for channel_id in channel_ids:
            try:
                data = self._request(
                    "POST",
                    f"/channels/{channel_id}/messages",
                    json={"content": content},
                )
                return DiscordNotificationResult(
                    channel_id=channel_id,
                    message_id=str(data.get("id") or "") or None,
                    order_count=len(submitted),
                    fallback_used=channel_id != first_channel,
                )
            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                if status_code not in {403, 404}:
                    raise
                last_exc = exc
                continue

        if last_exc is not None:
            raise last_exc
        return None

    def _target_channels(self) -> list[str]:
        channels: list[str] = []
        for channel_id in (
            self.settings.discord_trade_log_channel_id,
            self.settings.discord_channel_id,
        ):
            if channel_id and channel_id not in channels:
                channels.append(channel_id)
        return channels

    def _build_message(
        self,
        plan: ExecutionPlan,
        submissions: list[OrderSubmission],
        instructions: list[ExecutionInstruction],
    ) -> str:
        instructions_by_symbol = {item.symbol: item for item in instructions}
        lines = [
            "TATETUCK PAPER TRADE ALERT",
            f"Account: {plan.account_id or 'unknown'}",
            f"Readiness: {plan.readiness_status}",
            f"Orders submitted: {len(submissions)}",
            "",
        ]
        for submission in submissions[:8]:
            instruction = instructions_by_symbol.get(submission.symbol)
            side = submission.action.replace("_", " ").upper()
            detail = f"{side} {submission.symbol}"
            if submission.submitted_notional is not None:
                detail += f" | notional ${submission.submitted_notional:,.2f}"
            if submission.submitted_qty is not None:
                detail += f" | qty {submission.submitted_qty:,.4f}"
            if instruction is not None:
                detail += f" | scenario {instruction.scenario}"
                if instruction.company_state or instruction.setup_type:
                    detail += " | "
                    detail += "/".join(
                        [
                            value
                            for value in (instruction.company_state, instruction.setup_type)
                            if value
                        ]
                    )
                detail += f" | confidence {instruction.confidence * 100:.1f}%"
            if submission.raw_status:
                detail += f" | alpaca {submission.raw_status}"
            lines.append(detail)
        if len(submissions) > 8:
            lines.append(f"... plus {len(submissions) - 8} more submitted orders")
        message = "\n".join(lines)
        return message[:1900]

    def _request(self, method: str, path: str, json: dict[str, object]):
        token = self.settings.discord_token
        if not token:
            raise RuntimeError("Discord trade alerts are not configured. Set DISCORD_TOKEN.")
        response = self.session.request(
            method,
            f"{self.base_url}{path}",
            headers={
                "Authorization": f"Bot {token}",
                "User-Agent": "TatetuckBot/1.0",
            },
            json=json,
            timeout=20,
        )
        response.raise_for_status()
        return response.json()


class PMExecutionPlanner:
    def __init__(self, settings: VNextSettings):
        self.settings = settings

    def build_plan(
        self,
        analyses: Iterable[CompanyAnalysis],
        account: BrokerAccount,
        positions: Iterable[BrokerPosition],
        readiness: ReadinessReport,
    ) -> ExecutionPlan:
        analyses = list(analyses)
        positions_by_symbol = {position.symbol: position for position in positions}
        profiles = {
            analysis.snapshot.ticker: self._execution_profile(
                analysis,
                has_position=analysis.snapshot.ticker in positions_by_symbol,
            )
            for analysis in analyses
        }
        blockers: list[str] = []
        warnings: list[str] = list(readiness.blockers)
        selected = self._select_recommendations(analyses, positions_by_symbol, profiles)
        deployable_notional = min(
            account.equity * (self.settings.max_gross_exposure_pct / 100.0),
            account.buying_power,
        )

        if account.trading_blocked or account.account_blocked:
            blockers.append("Broker account is blocked from trading.")
        if account.status.upper() not in {"ACTIVE", "ACCOUNT_UPDATED", "SIMULATED"}:
            warnings.append(f"Broker account status is {account.status}.")
        if readiness.blockers and not self.settings.allow_blocked_paper_trading:
            blockers.append("Readiness blockers prevent paper deployment. Set TATETUCK_ALLOW_BLOCKED_PAPER_TRADING=true to override.")
        if not selected:
            blockers.append("No recommendations cleared the PM execution thresholds.")

        capped_weights = {
            analysis.snapshot.ticker: min(
                analysis.portfolio.target_weight,
                self.settings.max_single_position_pct,
                profiles[analysis.snapshot.ticker].weight_cap_pct,
            )
            for analysis in selected
        }
        total_weight = sum(capped_weights.values())
        gross_cap = self.settings.max_gross_exposure_pct
        scale = 1.0 if total_weight <= gross_cap or total_weight <= 0 else gross_cap / total_weight

        instructions: list[ExecutionInstruction] = []
        selected_symbols: list[str] = []
        for analysis in selected:
            rec = analysis.portfolio
            signal = analysis.signal
            ticker = analysis.snapshot.ticker
            profile = profiles[ticker]
            selected_symbols.append(ticker)
            scaled_target_weight = capped_weights[ticker] * scale
            target_notional = deployable_notional * (scaled_target_weight / max(gross_cap, 1e-9))
            position = positions_by_symbol.get(ticker)
            current_notional = position.market_value if position else 0.0
            delta_notional = target_notional - current_notional
            current_weight_pct = (current_notional / max(account.equity, 1e-9)) * 100.0 if current_notional else 0.0
            weight_gap_pct = abs(scaled_target_weight - current_weight_pct)
            action = "hold"
            side = "none"
            qty = None
            notional = None
            rationale = [
                f"execution_profile={profile.name}",
                f"execution_mode={profile.mode}",
                f"scenario={rec.scenario}",
                f"confidence={rec.confidence:.3f}",
                f"target_weight={rec.target_weight:.2f}%",
                f"scaled_target_weight={scaled_target_weight:.2f}%",
            ]
            rationale.extend(profile.rationale)
            if position and weight_gap_pct < self.settings.execution_rebalance_band_pct:
                rationale.append(
                    f"inside_rebalance_band={self.settings.execution_rebalance_band_pct:.2f}%"
                )
            elif delta_notional >= self.settings.min_order_notional:
                action = "buy"
                side = "buy"
                notional = round(delta_notional, 2)
            elif delta_notional <= -self.settings.min_order_notional and position and position.current_price > 0:
                action = "sell"
                side = "sell"
                qty = min(position.qty, abs(delta_notional) / position.current_price)
                if qty * position.current_price < self.settings.min_order_notional:
                    action = "hold"
                    side = "none"
                    qty = None

            instructions.append(
                ExecutionInstruction(
                    symbol=ticker,
                    company_name=analysis.snapshot.company_name,
                    as_of=analysis.snapshot.as_of,
                    action=action,
                    side=side,
                    scenario=rec.scenario,
                    company_state=signal.company_state or rec.company_state,
                    setup_type=signal.setup_type or rec.setup_type,
                    execution_profile=profile.name,
                    confidence=rec.confidence,
                    target_weight=rec.target_weight,
                    scaled_target_weight=round(scaled_target_weight, 2),
                    target_notional=round(target_notional, 2),
                    current_notional=round(current_notional, 2),
                    delta_notional=round(delta_notional, 2),
                    internal_upside_pct=signal.internal_upside_pct,
                    floor_support_pct=signal.floor_support_pct,
                    qty=round(qty, 6) if qty is not None else None,
                    notional=notional,
                    rationale=rationale,
                )
            )

        managed_symbols = {analysis.snapshot.ticker for analysis in analyses}
        analyses_by_symbol = {analysis.snapshot.ticker: analysis for analysis in analyses}
        for symbol, position in positions_by_symbol.items():
            if symbol in selected_symbols or symbol not in managed_symbols:
                continue
            if position.market_value < self.settings.min_order_notional:
                continue
            analysis = analyses_by_symbol.get(symbol)
            profile = profiles.get(symbol)
            if analysis is not None and self._qualifies_hold(analysis, profile):
                instructions.append(
                    ExecutionInstruction(
                        symbol=symbol,
                        company_name=analysis.snapshot.company_name,
                        as_of=analysis.snapshot.as_of,
                        action="hold",
                        side="none",
                        scenario=f"hold {analysis.portfolio.scenario}",
                        company_state=analysis.signal.company_state or analysis.portfolio.company_state,
                        setup_type=analysis.signal.setup_type or analysis.portfolio.setup_type,
                        execution_profile=profile.name if profile is not None else "holdover",
                        confidence=analysis.portfolio.confidence,
                        target_weight=analysis.portfolio.target_weight,
                        scaled_target_weight=analysis.portfolio.target_weight,
                        target_notional=round(position.market_value, 2),
                        current_notional=round(position.market_value, 2),
                        delta_notional=0.0,
                        internal_upside_pct=analysis.signal.internal_upside_pct,
                        floor_support_pct=analysis.signal.floor_support_pct,
                        qty=None,
                        notional=None,
                        rationale=[
                            "Existing position retained under PM holdover thresholds."
                        ]
                        + ([] if profile is None else [f"execution_profile={profile.name}"] + profile.rationale),
                    )
                )
                continue
            instructions.append(
                ExecutionInstruction(
                    symbol=symbol,
                    company_name=symbol,
                    as_of=analysis.snapshot.as_of if analysis is not None else None,
                    action="sell",
                    side="sell",
                    scenario="exit unmanaged",
                    company_state=analysis.signal.company_state if analysis is not None else None,
                    setup_type=analysis.signal.setup_type if analysis is not None else None,
                    execution_profile="exit_unmanaged",
                    confidence=0.0,
                    target_weight=0.0,
                    scaled_target_weight=0.0,
                    target_notional=0.0,
                    current_notional=round(position.market_value, 2),
                    delta_notional=round(-position.market_value, 2),
                    internal_upside_pct=analysis.signal.internal_upside_pct if analysis is not None else None,
                    floor_support_pct=analysis.signal.floor_support_pct if analysis is not None else None,
                    qty=round(position.qty, 6),
                    notional=None,
                    rationale=["Position is in managed universe but no longer selected for PM deployment."],
                )
            )

        return ExecutionPlan(
            generated_at=utc_now_iso(),
            account_id=account.account_id,
            equity=account.equity,
            buying_power=account.buying_power,
            deployable_notional=round(deployable_notional, 2),
            selected_symbols=selected_symbols,
            instructions=instructions,
            blockers=blockers,
            warnings=warnings,
            readiness_status=readiness.status,
        )

    def _select_recommendations(
        self,
        analyses: list[CompanyAnalysis],
        positions_by_symbol: dict[str, BrokerPosition],
        profiles: dict[str, ExecutionProfile],
    ) -> list[CompanyAnalysis]:
        auto_eligible = [
            analysis
            for analysis in analyses
            if self._qualifies_auto(analysis, profiles.get(analysis.snapshot.ticker))
        ]
        auto_eligible = sorted(
            auto_eligible,
            key=lambda analysis: (
                profiles[analysis.snapshot.ticker].score,
                analysis.portfolio.target_weight * analysis.portfolio.confidence,
                analysis.signal.expected_return,
            ),
            reverse=True,
        )
        selected: list[CompanyAnalysis] = []
        selected_symbols: set[str] = set()

        for analysis in auto_eligible:
            if analysis.snapshot.ticker in positions_by_symbol:
                selected.append(analysis)
                selected_symbols.add(analysis.snapshot.ticker)

        new_slots = max(self.settings.max_new_positions, 0)
        for analysis in auto_eligible:
            if analysis.snapshot.ticker in selected_symbols:
                continue
            if new_slots <= 0:
                break
            selected.append(analysis)
            selected_symbols.add(analysis.snapshot.ticker)
            new_slots -= 1

        holdovers = [
            analysis
            for analysis in analyses
            if analysis.snapshot.ticker in positions_by_symbol
            and analysis.snapshot.ticker not in selected_symbols
            and self._qualifies_hold(analysis, profiles.get(analysis.snapshot.ticker))
        ]
        holdovers = sorted(
            holdovers,
            key=lambda analysis: (
                profiles[analysis.snapshot.ticker].score,
                analysis.portfolio.target_weight,
                analysis.portfolio.confidence,
                analysis.signal.expected_return,
            ),
            reverse=True,
        )
        selected.extend(holdovers)
        return selected

    @staticmethod
    def _qualifies_auto(analysis: CompanyAnalysis, profile: ExecutionProfile | None) -> bool:
        return analysis.portfolio.stance == "long" and profile is not None and profile.mode == "auto"

    @staticmethod
    def _qualifies_hold(analysis: CompanyAnalysis, profile: ExecutionProfile | None) -> bool:
        return (
            analysis.portfolio.stance == "long"
            and profile is not None
            and profile.mode in {"auto", "hold"}
        )

    def _execution_profile(
        self,
        analysis: CompanyAnalysis,
        has_position: bool,
    ) -> ExecutionProfile:
        rec = analysis.portfolio
        signal = analysis.signal
        state = signal.company_state or rec.company_state or "pre_commercial"
        setup = signal.setup_type or rec.setup_type or "watchful"
        expected_return = signal.expected_return
        catalyst_success = signal.catalyst_success_prob
        confidence = rec.confidence
        target_weight = rec.target_weight
        financing_risk = signal.financing_risk
        crowding_risk = signal.crowding_risk
        upside = signal.internal_upside_pct if signal.internal_upside_pct is not None else 0.0
        floor = signal.floor_support_pct if signal.floor_support_pct is not None else 0.0

        if rec.stance != "long":
            return self._blocked_profile("not_long", analysis, "Recommendation stance is not long.")
        if rec.scenario == "avoid due to financing" or financing_risk > 0.82:
            return self._blocked_profile("financing_blocked", analysis, "Financing overhang is too high for deployment.")
        if rec.scenario == "watchlist only" and setup not in {"asymmetry_without_near_term_catalyst", "sentiment_floor"}:
            return self._blocked_profile("watchlist_only", analysis, "Watchlist-only setup is not eligible for PM deployment.")
        if upside <= -0.12:
            return self._blocked_profile("negative_asymmetry", analysis, "Peer-anchored value view is too far below the market setup.")
        if expected_return <= 0.0 and upside <= 0.0:
            return self._blocked_profile("no_edge", analysis, "Expected return and asymmetry do not justify deployment.")
        if target_weight < self.settings.execution_hold_weight_pct and not has_position:
            return self._blocked_profile("subscale", analysis, "Research target weight is below the minimum entry size.")
        if confidence < self.settings.execution_hold_confidence and not has_position:
            return self._blocked_profile("low_confidence", analysis, "Confidence is below the minimum hold threshold.")

        if setup == "hard_catalyst":
            min_confidence = max(
                self.settings.execution_min_hard_catalyst_confidence,
                self.settings.min_execution_confidence,
            )
            if (
                confidence >= min_confidence
                and catalyst_success >= 0.50
                and upside >= -0.02
                and financing_risk <= 0.72
            ):
                return self._auto_profile(
                    "hard_catalyst",
                    analysis,
                    self.settings.execution_max_hard_catalyst_weight_pct,
                    "High-conviction hard catalyst cleared the execution bar.",
                    score_bonus=0.45,
                )
            if has_position and confidence >= self.settings.execution_hold_confidence and catalyst_success >= 0.45:
                return self._hold_profile(
                    "hard_catalyst_hold",
                    analysis,
                    self.settings.execution_max_soft_catalyst_weight_pct,
                    "Hold existing exposure while the hard-catalyst setup matures.",
                    score_bonus=0.15,
                )
            return self._blocked_profile("hard_catalyst_blocked", analysis, "Hard catalyst exists, but the execution bar was not met.")

        if setup == "soft_catalyst":
            min_floor = max(self.settings.execution_min_floor_support_pct - 0.02, 0.0)
            if (
                confidence >= self.settings.execution_min_soft_catalyst_confidence
                and catalyst_success >= 0.55
                and upside >= self.settings.execution_min_internal_upside_pct - 0.02
                and floor >= min_floor
            ):
                return self._auto_profile(
                    "soft_catalyst",
                    analysis,
                    self.settings.execution_max_soft_catalyst_weight_pct,
                    "Soft catalyst has enough support and asymmetry for a measured entry.",
                    score_bonus=0.20,
                )
            if has_position and confidence >= self.settings.execution_hold_confidence and upside >= self.settings.execution_min_internal_upside_pct:
                return self._hold_profile(
                    "soft_catalyst_hold",
                    analysis,
                    self.settings.execution_max_soft_catalyst_weight_pct,
                    "Keep exposure on, but do not add until the setup firms up.",
                )
            return self._blocked_profile("soft_catalyst_blocked", analysis, "Soft catalyst lacks enough support or upside for auto deployment.")

        if setup == "launch_asymmetry":
            if (
                state == "commercial_launch"
                and confidence >= self.settings.execution_min_launch_confidence
                and expected_return > 0.08
                and upside >= self.settings.execution_min_internal_upside_pct - 0.02
                and floor >= self.settings.execution_min_floor_support_pct
                and financing_risk <= 0.68
            ):
                return self._auto_profile(
                    "launch_asymmetry",
                    analysis,
                    self.settings.execution_max_launch_weight_pct,
                    "Launch setup offers enough asymmetry and balance-sheet support for deployment.",
                    score_bonus=0.30,
                )
            if has_position and confidence >= self.settings.execution_hold_confidence and floor >= self.settings.execution_min_floor_support_pct:
                return self._hold_profile(
                    "launch_hold",
                    analysis,
                    self.settings.execution_max_launch_weight_pct,
                    "Retain launch exposure while waiting for cleaner commercial confirmation.",
                )
            return self._blocked_profile("launch_blocked", analysis, "Launch asymmetry is not yet strong enough for fresh capital.")

        if setup in {"pipeline_optionality", "capital_allocation"}:
            if (
                state in {"commercial_launch", "commercialized"}
                and confidence >= self.settings.execution_min_franchise_confidence
                and expected_return > 0.05
                and upside >= self.settings.execution_min_internal_upside_pct - 0.04
                and floor >= self.settings.execution_min_floor_support_pct
            ):
                return self._auto_profile(
                    setup,
                    analysis,
                    self.settings.execution_max_franchise_weight_pct,
                    "Franchise optionality is strong enough for a measured PM deployment.",
                    score_bonus=0.18,
                )
            if has_position and confidence >= self.settings.execution_hold_confidence and floor >= self.settings.execution_min_floor_support_pct:
                return self._hold_profile(
                    f"{setup}_hold",
                    analysis,
                    self.settings.execution_max_franchise_weight_pct,
                    "Keep franchise exposure on, but wait for more concrete validation before adding.",
                )
            return self._blocked_profile(f"{setup}_blocked", analysis, "Franchise setup does not yet clear the execution thresholds.")

        if setup == "sentiment_floor":
            if (
                state in {"commercial_launch", "commercialized"}
                and confidence >= self.settings.execution_min_franchise_confidence
                and expected_return > 0.04
                and upside >= self.settings.execution_min_internal_upside_pct - 0.05
                and floor >= self.settings.execution_min_floor_support_pct + 0.05
            ):
                return self._auto_profile(
                    "sentiment_floor",
                    analysis,
                    self.settings.execution_max_floor_weight_pct,
                    "Sentiment/floor setup is investable, but only at a smaller size cap.",
                    score_bonus=0.08,
                )
            if has_position and confidence >= self.settings.execution_hold_confidence and floor >= self.settings.execution_min_floor_support_pct:
                return self._hold_profile(
                    "sentiment_floor_hold",
                    analysis,
                    self.settings.execution_max_floor_weight_pct,
                    "Floor support justifies holding existing exposure without adding.",
                )
            return self._blocked_profile("sentiment_floor_blocked", analysis, "Floor setup is not strong enough for fresh capital.")

        if setup == "asymmetry_without_near_term_catalyst":
            if (
                state == "pre_commercial"
                and has_position
                and confidence >= max(self.settings.execution_hold_confidence, 0.56)
                and upside >= self.settings.execution_min_internal_upside_pct
                and floor >= self.settings.execution_min_floor_support_pct + 0.06
                and financing_risk <= 0.60
            ):
                return self._hold_profile(
                    "precommercial_asymmetry_hold",
                    analysis,
                    self.settings.execution_max_floor_weight_pct,
                    "Interesting pre-commercial asymmetry, but without a clean catalyst it stays hold-only.",
                    score_bonus=0.05,
                )
            if (
                state in {"commercial_launch", "commercialized"}
                and confidence >= self.settings.execution_min_franchise_confidence
                and expected_return > 0.07
                and upside >= self.settings.execution_min_internal_upside_pct
                and floor >= self.settings.execution_min_floor_support_pct + 0.02
            ):
                return self._auto_profile(
                    "franchise_asymmetry",
                    analysis,
                    self.settings.execution_max_floor_weight_pct,
                    "Asymmetry without a dated catalyst is only deployable at a smaller size.",
                )
            return self._blocked_profile(
                "no_near_term_catalyst",
                analysis,
                "No hard catalyst yet, so the setup stays on the watchlist until the asymmetry firms up.",
            )

        if crowding_risk > 0.88 and not has_position:
            return self._blocked_profile("crowded_setup", analysis, "Crowding is too elevated for a fresh entry.")
        if has_position and confidence >= self.settings.execution_hold_confidence and upside > 0.0 and floor >= self.settings.execution_min_floor_support_pct:
            return self._hold_profile(
                "default_hold",
                analysis,
                self.settings.execution_max_floor_weight_pct,
                "Setup remains constructive enough to keep the position on.",
            )
        return self._blocked_profile("watchful", analysis, "Setup is interesting, but not yet deployable under PM rules.")

    def _auto_profile(
        self,
        name: str,
        analysis: CompanyAnalysis,
        weight_cap_pct: float,
        rationale: str,
        score_bonus: float = 0.0,
    ) -> ExecutionProfile:
        score = self._execution_score(analysis, score_bonus=score_bonus)
        if analysis.signal.crowding_risk > 0.80 and name != "hard_catalyst":
            weight_cap_pct = min(weight_cap_pct, self.settings.execution_max_floor_weight_pct)
            rationale += " Crowding trims the live size cap."
        return ExecutionProfile(
            name=name,
            mode="auto",
            weight_cap_pct=weight_cap_pct,
            score=score,
            rationale=[rationale],
        )

    def _hold_profile(
        self,
        name: str,
        analysis: CompanyAnalysis,
        weight_cap_pct: float,
        rationale: str,
        score_bonus: float = 0.0,
    ) -> ExecutionProfile:
        return ExecutionProfile(
            name=name,
            mode="hold",
            weight_cap_pct=weight_cap_pct,
            score=self._execution_score(analysis, score_bonus=score_bonus - 0.25),
            rationale=[rationale],
        )

    def _blocked_profile(
        self,
        name: str,
        analysis: CompanyAnalysis,
        rationale: str,
    ) -> ExecutionProfile:
        return ExecutionProfile(
            name=name,
            mode="block",
            weight_cap_pct=0.0,
            score=self._execution_score(analysis, score_bonus=-2.0),
            rationale=[rationale],
        )

    @staticmethod
    def _execution_score(
        analysis: CompanyAnalysis,
        score_bonus: float = 0.0,
    ) -> float:
        signal = analysis.signal
        upside = signal.internal_upside_pct if signal.internal_upside_pct is not None else 0.0
        floor = signal.floor_support_pct if signal.floor_support_pct is not None else 0.0
        return (
            max(signal.expected_return, 0.0) * 4.0
            + (analysis.portfolio.confidence * 1.5)
            + (max(signal.catalyst_success_prob - 0.50, 0.0) * 1.2)
            + (max(upside, 0.0) * 0.02)
            + (max(floor, 0.0) * 0.05)
            - (signal.crowding_risk * 0.8)
            - (signal.financing_risk * 1.2)
            + score_bonus
        )


def execute_plan(
    plan: ExecutionPlan,
    broker: AlpacaPaperBroker,
    store: LocalResearchStore,
    submit: bool = False,
) -> list[OrderSubmission]:
    rows = [instruction.to_row() for instruction in plan.instructions]
    if rows:
        store.append_records("order_plans", rows)

    submissions: list[OrderSubmission] = []
    actionable = [instruction for instruction in plan.instructions if instruction.action in {"buy", "sell"}]
    if not submit:
        return [
            OrderSubmission(
                symbol=instruction.symbol,
                action=instruction.action,
                status="planned",
                client_order_id="dry-run",
                order_id=None,
                submitted_notional=instruction.notional,
                submitted_qty=instruction.qty,
                notes="dry run only",
            )
            for instruction in actionable
        ]

    broker.ensure_paper_only()
    for instruction in actionable:
        if instruction.action == "buy" and instruction.notional:
            submissions.append(broker.submit_market_notional_buy(instruction.symbol, instruction.notional))
        elif instruction.action == "sell" and instruction.qty:
            submissions.append(broker.submit_market_qty_sell(instruction.symbol, instruction.qty))

    if submissions:
        store.append_records("order_submissions", [submission.to_row() for submission in submissions])
    return submissions


def materialize_execution_feedback(
    store: LocalResearchStore,
    history_provider: PriceHistoryProvider | None = None,
) -> ExecutionFeedbackSummary:
    plans = store.read_table("order_plans")
    if plans.empty:
        store.replace_table("execution_feedback", [])
        store.replace_table("execution_profile_scorecards", [])
        return ExecutionFeedbackSummary(
            feedback_rows=0,
            matured_30d_rows=0,
            matured_90d_rows=0,
            scorecard_rows=0,
        )

    eodhd_key = os.environ.get("EODHD_API_KEY")
    provider = history_provider or CompositeHistoryProvider(
        [
            EODHDHistoryProvider(store=store),
            YFinanceHistoryProvider(store=store, allow_live=not bool(eodhd_key)),
        ]
    )

    actionable = plans[plans["action"].isin(["buy", "sell"])].copy()
    if actionable.empty:
        store.replace_table("execution_feedback", [])
        store.replace_table("execution_profile_scorecards", [])
        return ExecutionFeedbackSummary(
            feedback_rows=0,
            matured_30d_rows=0,
            matured_90d_rows=0,
            scorecard_rows=0,
        )
    for column in ("as_of", "planned_at", "company_state", "setup_type", "execution_profile", "scenario", "company_name"):
        if column not in actionable.columns:
            actionable[column] = None
    entry_anchor = actionable["as_of"].fillna(actionable["planned_at"])
    actionable["entry_ts"] = pd.to_datetime(
        entry_anchor,
        errors="coerce",
        utc=True,
        format="mixed",
    ).dt.tz_convert(None)
    actionable = actionable.dropna(subset=["entry_ts"])
    if actionable.empty:
        return ExecutionFeedbackSummary(
            feedback_rows=0,
            matured_30d_rows=0,
            matured_90d_rows=0,
            scorecard_rows=0,
        )

    feedback_rows: list[dict[str, object]] = []
    for symbol in sorted(actionable["symbol"].dropna().unique().tolist()):
        symbol_rows = actionable[actionable["symbol"] == symbol].copy()
        start = (symbol_rows["entry_ts"].min() - pd.Timedelta(days=14)).date().isoformat()
        end = (symbol_rows["entry_ts"].max() + pd.Timedelta(days=220)).date().isoformat()
        history = provider.load_history(symbol, start=start, end=end)
        if history.empty or "close" not in history.columns:
            continue
        close = history["close"].astype(float)
        close.index = pd.to_datetime(close.index).astype("datetime64[ns]")
        for row in symbol_rows.itertuples(index=False):
            entry_ts = pd.Timestamp(row.entry_ts)
            entry_price = _price_at_or_after(close, entry_ts)
            if entry_price is None or entry_price <= 0:
                continue
            direction = 1.0 if row.action == "buy" else -1.0
            record = {
                "symbol": row.symbol,
                "company_name": row.company_name,
                "as_of": row.as_of,
                "planned_at": row.planned_at,
                "entry_ts": entry_ts.isoformat(),
                "entry_price": float(entry_price),
                "direction": direction,
                "action": row.action,
                "scenario": row.scenario,
                "company_state": row.company_state,
                "setup_type": row.setup_type,
                "execution_profile": row.execution_profile,
            }
            latest_price = _price_at_or_before(close, close.index.max())
            record["mark_to_market_return"] = (
                float(direction * ((latest_price / entry_price) - 1.0))
                if latest_price is not None
                else None
            )
            for horizon in (10, 30, 90):
                exit_price = _price_at_or_after(close, entry_ts + pd.Timedelta(days=horizon))
                realized = None if exit_price is None else float(direction * ((exit_price / entry_price) - 1.0))
                record[f"return_{horizon}d"] = realized
                record[f"matured_{horizon}d"] = realized is not None
            feedback_rows.append(record)

    store.replace_table("execution_feedback", feedback_rows)

    feedback = pd.DataFrame(feedback_rows)
    if feedback.empty:
        store.replace_table("execution_profile_scorecards", [])
        return ExecutionFeedbackSummary(
            feedback_rows=0,
            matured_30d_rows=0,
            matured_90d_rows=0,
            scorecard_rows=0,
        )

    scorecards: list[dict[str, object]] = []
    for profile, group in feedback.groupby("execution_profile", dropna=False):
        row: dict[str, object] = {
            "execution_profile": profile,
            "trades": int(len(group)),
            "company_states": ",".join(sorted(set(group["company_state"].dropna().astype(str).tolist()))),
            "setup_types": ",".join(sorted(set(group["setup_type"].dropna().astype(str).tolist()))),
        }
        for horizon in (10, 30, 90):
            matured = group[group[f"matured_{horizon}d"].fillna(False)].copy()
            row[f"matured_{horizon}d_trades"] = int(len(matured))
            row[f"hit_rate_{horizon}d"] = (
                float((matured[f"return_{horizon}d"] > 0.0).mean())
                if not matured.empty
                else None
            )
            row[f"avg_return_{horizon}d"] = (
                float(matured[f"return_{horizon}d"].mean())
                if not matured.empty
                else None
            )
        row["avg_mark_to_market_return"] = float(group["mark_to_market_return"].mean())
        scorecards.append(row)
    store.replace_table("execution_profile_scorecards", scorecards)
    return ExecutionFeedbackSummary(
        feedback_rows=len(feedback_rows),
        matured_30d_rows=int(feedback["matured_30d"].fillna(False).sum()),
        matured_90d_rows=int(feedback["matured_90d"].fillna(False).sum()),
        scorecard_rows=len(scorecards),
    )


def record_trade_run(
    store: LocalResearchStore,
    settings: VNextSettings,
    plan: ExecutionPlan,
    submissions: list[OrderSubmission],
    submit: bool,
    started_at: str,
    status: str,
    notes: str = "",
) -> None:
    finished_at = utc_now_iso()
    record_pipeline_run(
        store=store,
        job_name="trade_vnext",
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        metrics={
            "instructions": len(plan.instructions),
            "actionable_instructions": len([item for item in plan.instructions if item.action in {"buy", "sell"}]),
            "submitted_orders": len([item for item in submissions if item.status == "submitted"]),
            "blocked_orders": len(plan.blockers),
        },
        config={
            "submit": submit,
            "store_dir": settings.store_dir,
            "alpaca_api_base_url": settings.alpaca_api_base_url,
            "alpaca_paper_account_id": settings.alpaca_paper_account_id,
            "max_gross_exposure_pct": settings.max_gross_exposure_pct,
            "max_single_position_pct": settings.max_single_position_pct,
        },
        notes=notes,
    )


def _price_at_or_after(close: pd.Series, timestamp: pd.Timestamp) -> float | None:
    idx = close.index.values.searchsorted(timestamp.to_datetime64().astype("datetime64[ns]"), side="left")
    if idx >= len(close):
        return None
    return float(close.iloc[idx])


def _price_at_or_before(close: pd.Series, timestamp: pd.Timestamp) -> float | None:
    idx = close.index.values.searchsorted(timestamp.to_datetime64().astype("datetime64[ns]"), side="right") - 1
    if idx < 0:
        return None
    return float(close.iloc[idx])
