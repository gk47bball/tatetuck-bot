from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def requested_notional_from_target_weight(
    target_weight_pct: float | None,
    *,
    reference_book_notional: float,
    min_order_notional: float = 0.0,
) -> float | None:
    weight = _to_float(target_weight_pct, default=0.0)
    if weight <= 0.0:
        return None
    notional = (weight / 100.0) * max(_to_float(reference_book_notional, 0.0), 0.0)
    if notional <= 0.0:
        return None
    return float(max(notional, _to_float(min_order_notional, 0.0)))


@dataclass(slots=True)
class SimulatedFill:
    symbol: str
    action: str
    requested_notional: float | None
    requested_qty: float | None
    executable_notional: float | None
    executable_qty: float | None
    liquidity_fill_ratio: float
    expected_slippage_bps: float
    expected_round_trip_cost_bps: float
    rejection_reason: str | None = None
    partial_fill: bool = False

    def to_row(self) -> dict[str, object]:
        row = asdict(self)
        row["simulated_at"] = pd.Timestamp.now(tz="UTC").isoformat()
        return row


def snapshot_microstructure(
    snapshots: pd.DataFrame,
    symbol: str,
    entry_ts: pd.Timestamp,
) -> tuple[float | None, float | None]:
    ticker_rows = snapshots[snapshots["ticker"] == symbol]
    if ticker_rows.empty:
        return None, None
    ticker_rows = ticker_rows[ticker_rows["as_of_ts"] <= entry_ts]
    if ticker_rows.empty:
        return None, None
    latest = ticker_rows.iloc[-1]
    market_cap = _to_float(latest.get("market_cap"), default=0.0)
    volatility = _to_float(latest.get("volatility"), default=0.0)
    return (market_cap if market_cap > 0 else None, volatility if volatility > 0 else 0.0)


def estimated_round_trip_cost_bps(
    market_cap: float | None,
    volatility: float | None,
    setup_type: str | None,
) -> float:
    if market_cap is None or market_cap <= 0:
        base_bps = 275.0
    elif market_cap < 250_000_000:
        base_bps = 425.0
    elif market_cap < 1_000_000_000:
        base_bps = 220.0
    elif market_cap < 5_000_000_000:
        base_bps = 110.0
    else:
        base_bps = 45.0
    vol_bps = min(max(_to_float(volatility, 0.0), 0.0) * 1000.0, 125.0)
    setup_bps = 35.0 if setup_type in {"hard_catalyst", "soft_catalyst"} else 0.0
    return float(base_bps + vol_bps + setup_bps)


class ExecutionSimulator:
    def simulate_instruction(
        self,
        symbol: str,
        action: str,
        requested_notional: float | None,
        requested_qty: float | None,
        price: float | None,
        market_cap: float | None,
        volatility: float | None,
        setup_type: str | None,
    ) -> SimulatedFill:
        requested_notional = None if requested_notional is None else max(float(requested_notional), 0.0)
        requested_qty = None if requested_qty is None else max(float(requested_qty), 0.0)
        price = None if price is None or price <= 0 else float(price)
        if price is not None and requested_notional is None and requested_qty is not None:
            requested_notional = requested_qty * price
        if price is not None and requested_qty is None and requested_notional is not None:
            requested_qty = requested_notional / price

        round_trip_cost_bps = estimated_round_trip_cost_bps(market_cap, volatility, setup_type)
        entry_slippage_bps = round_trip_cost_bps / 2.0
        liquidity_cap = self._liquidity_cap_notional(market_cap=market_cap, setup_type=setup_type)
        if requested_notional is None or requested_notional <= 0:
            return SimulatedFill(
                symbol=symbol,
                action=action,
                requested_notional=requested_notional,
                requested_qty=requested_qty,
                executable_notional=None,
                executable_qty=None,
                liquidity_fill_ratio=0.0,
                expected_slippage_bps=entry_slippage_bps,
                expected_round_trip_cost_bps=round_trip_cost_bps,
                rejection_reason="missing_order_size",
            )

        fill_ratio = min(1.0, liquidity_cap / max(requested_notional, 1.0))
        if fill_ratio < 0.25:
            return SimulatedFill(
                symbol=symbol,
                action=action,
                requested_notional=requested_notional,
                requested_qty=requested_qty,
                executable_notional=0.0,
                executable_qty=0.0,
                liquidity_fill_ratio=fill_ratio,
                expected_slippage_bps=entry_slippage_bps,
                expected_round_trip_cost_bps=round_trip_cost_bps,
                rejection_reason="liquidity_cap",
            )

        executable_notional = requested_notional * fill_ratio
        executable_qty = None if requested_qty is None else requested_qty * fill_ratio
        return SimulatedFill(
            symbol=symbol,
            action=action,
            requested_notional=requested_notional,
            requested_qty=requested_qty,
            executable_notional=executable_notional,
            executable_qty=executable_qty,
            liquidity_fill_ratio=fill_ratio,
            expected_slippage_bps=entry_slippage_bps,
            expected_round_trip_cost_bps=round_trip_cost_bps,
            partial_fill=fill_ratio < 0.999,
        )

    @staticmethod
    def _liquidity_cap_notional(market_cap: float | None, setup_type: str | None) -> float:
        if market_cap is None or market_cap <= 0:
            cap = 1_000.0
        elif market_cap < 250_000_000:
            cap = 1_250.0
        elif market_cap < 1_000_000_000:
            cap = 4_000.0
        elif market_cap < 5_000_000_000:
            cap = 10_000.0
        else:
            cap = 25_000.0
        if setup_type in {"hard_catalyst", "soft_catalyst"}:
            cap *= 0.7
        return float(cap)
