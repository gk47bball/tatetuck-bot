from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Iterable
from uuid import uuid4

import requests

from .entities import CompanyAnalysis, PortfolioRecommendation
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
    confidence: float
    target_weight: float
    scaled_target_weight: float
    target_notional: float
    current_notional: float
    delta_notional: float
    qty: float | None = None
    notional: float | None = None
    rationale: list[str] = field(default_factory=list)

    def to_row(self) -> dict[str, object]:
        payload = asdict(self)
        payload["planned_at"] = datetime.now(timezone.utc).isoformat()
        return payload


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


class PMExecutionPlanner:
    AUTO_SCENARIOS = {"pre-catalyst long", "commercial compounder"}

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
        blockers: list[str] = []
        warnings: list[str] = list(readiness.blockers)
        selected = self._select_recommendations(analyses)
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
            analysis.snapshot.ticker: min(analysis.portfolio.target_weight, self.settings.max_single_position_pct)
            for analysis in selected
        }
        total_weight = sum(capped_weights.values())
        gross_cap = self.settings.max_gross_exposure_pct
        scale = 1.0 if total_weight <= gross_cap or total_weight <= 0 else gross_cap / total_weight

        instructions: list[ExecutionInstruction] = []
        selected_symbols: list[str] = []
        for analysis in selected:
            rec = analysis.portfolio
            ticker = analysis.snapshot.ticker
            selected_symbols.append(ticker)
            scaled_target_weight = capped_weights[ticker] * scale
            target_notional = deployable_notional * (scaled_target_weight / max(gross_cap, 1e-9))
            position = positions_by_symbol.get(ticker)
            current_notional = position.market_value if position else 0.0
            delta_notional = target_notional - current_notional
            action = "hold"
            side = "none"
            qty = None
            notional = None
            rationale = [
                f"scenario={rec.scenario}",
                f"confidence={rec.confidence:.3f}",
                f"target_weight={rec.target_weight:.2f}%",
                f"scaled_target_weight={scaled_target_weight:.2f}%",
            ]
            if delta_notional >= self.settings.min_order_notional:
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
                    action=action,
                    side=side,
                    scenario=rec.scenario,
                    confidence=rec.confidence,
                    target_weight=rec.target_weight,
                    scaled_target_weight=round(scaled_target_weight, 2),
                    target_notional=round(target_notional, 2),
                    current_notional=round(current_notional, 2),
                    delta_notional=round(delta_notional, 2),
                    qty=round(qty, 6) if qty is not None else None,
                    notional=notional,
                    rationale=rationale,
                )
            )

        managed_symbols = {analysis.snapshot.ticker for analysis in analyses}
        for symbol, position in positions_by_symbol.items():
            if symbol in selected_symbols or symbol not in managed_symbols:
                continue
            if position.market_value < self.settings.min_order_notional:
                continue
            instructions.append(
                ExecutionInstruction(
                    symbol=symbol,
                    company_name=symbol,
                    action="sell",
                    side="sell",
                    scenario="exit unmanaged",
                    confidence=0.0,
                    target_weight=0.0,
                    scaled_target_weight=0.0,
                    target_notional=0.0,
                    current_notional=round(position.market_value, 2),
                    delta_notional=round(-position.market_value, 2),
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

    def _select_recommendations(self, analyses: list[CompanyAnalysis]) -> list[CompanyAnalysis]:
        eligible = [
            analysis
            for analysis in analyses
            if analysis.portfolio.scenario in self.AUTO_SCENARIOS
            and analysis.portfolio.target_weight >= self.settings.min_execution_weight_pct
            and analysis.portfolio.confidence >= self.settings.min_execution_confidence
            and analysis.portfolio.stance == "long"
        ]
        eligible = sorted(
            eligible,
            key=lambda analysis: (
                analysis.portfolio.target_weight * analysis.portfolio.confidence,
                analysis.signal.expected_return,
            ),
            reverse=True,
        )
        return eligible[: self.settings.max_new_positions]


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
