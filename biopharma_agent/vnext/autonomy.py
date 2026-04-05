from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Iterable, Protocol

import pandas as pd

from .entities import CompanyAnalysis
from .execution import BrokerAccount, BrokerPosition, ExecutionPlan, OrderSubmission
from .ops import ReadinessReport, build_readiness_report, utc_now_iso
from .settings import VNextSettings
from .storage import LocalResearchStore
from .validation import load_best_validation_payload, validation_payload_age_days


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _ts(value: Any) -> pd.Timestamp | pd.NaT:
    ts = pd.to_datetime(value, errors="coerce", utc=True, format="mixed")
    if pd.isna(ts):
        return pd.NaT
    return ts.tz_convert(None)


def _parse_listish(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return []


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


class BrokerReconciliationClient(Protocol):
    def account(self) -> BrokerAccount: ...

    def positions(self) -> list[BrokerPosition]: ...

    def recent_orders(self, limit: int = 200) -> list[dict[str, Any]]: ...


@dataclass(slots=True)
class BrokerReconciliationSummary:
    captured_at: str
    gross_exposure_pct: float
    instruction_count: int
    blocker_count: int
    submitted_order_count: int
    filled_order_count: int
    partially_filled_order_count: int
    rejected_order_count: int
    queued_order_count: int
    selected_symbols: list[str]
    actual_symbols: list[str]
    missing_symbols: list[str]
    unexpected_symbols: list[str]

    def to_row(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("selected_symbols", "actual_symbols", "missing_symbols", "unexpected_symbols"):
            payload[key] = json.dumps(payload[key])
        return payload


def record_portfolio_nav(
    store: LocalResearchStore,
    account: BrokerAccount,
    *,
    captured_at: str | None = None,
) -> dict[str, Any]:
    row = {
        "captured_at": captured_at or utc_now_iso(),
        "account_id": account.account_id,
        "equity": float(account.equity),
        "buying_power": float(account.buying_power),
        "cash": float(account.cash),
        "paper": bool(account.paper),
        "status": str(account.status),
    }
    store.append_records("portfolio_nav", [row])
    return row


def reconcile_broker_state(
    store: LocalResearchStore,
    broker: BrokerReconciliationClient,
    *,
    plan: ExecutionPlan | None = None,
    submissions: Iterable[OrderSubmission] | None = None,
    captured_at: str | None = None,
    order_lookback_hours: int = 48,
) -> BrokerReconciliationSummary:
    captured_at = captured_at or utc_now_iso()
    account = broker.account()
    positions = broker.positions()
    orders = broker.recent_orders(limit=200)
    actual_symbols = sorted({str(position.symbol).upper() for position in positions if abs(float(position.market_value)) > 0.01})
    selected_symbols = sorted({str(symbol).upper() for symbol in (plan.selected_symbols if plan is not None else [])})
    missing_symbols = sorted(set(selected_symbols) - set(actual_symbols))
    unexpected_symbols = sorted(set(actual_symbols) - set(selected_symbols))
    gross_exposure_pct = (
        sum(abs(float(position.market_value)) for position in positions) / max(float(account.equity), 1e-9) * 100.0
        if positions
        else 0.0
    )

    order_frame = pd.DataFrame(orders)
    if not order_frame.empty:
        for source_column, target_column in (
            ("id", "order_id"),
            ("client_order_id", "client_order_id"),
            ("symbol", "symbol"),
            ("status", "broker_status"),
            ("filled_qty", "filled_qty"),
            ("filled_avg_price", "filled_avg_price"),
            ("submitted_at", "submitted_at"),
            ("updated_at", "updated_at"),
            ("filled_at", "filled_at"),
            ("canceled_at", "canceled_at"),
            ("failed_at", "failed_at"),
            ("expired_at", "expired_at"),
        ):
            if source_column in order_frame.columns and target_column != source_column:
                order_frame[target_column] = order_frame[source_column]
        order_frame["captured_at"] = captured_at
        store.append_records(
            "broker_order_updates",
            order_frame[
                [
                    column
                    for column in (
                        "captured_at",
                        "order_id",
                        "client_order_id",
                        "symbol",
                        "broker_status",
                        "filled_qty",
                        "filled_avg_price",
                        "submitted_at",
                        "updated_at",
                        "filled_at",
                        "canceled_at",
                        "failed_at",
                        "expired_at",
                    )
                    if column in order_frame.columns
                ]
            ].to_dict(orient="records"),
        )

    order_status_by_id = {
        str(row.get("id") or ""): str(row.get("status") or "")
        for row in orders
        if row.get("id")
    }
    order_status_by_client = {
        str(row.get("client_order_id") or ""): str(row.get("status") or "")
        for row in orders
        if row.get("client_order_id")
    }
    recent_submissions = list(submissions or [])
    if not recent_submissions:
        stored_submissions = store.read_table("order_submissions")
        if not stored_submissions.empty and "submitted_at" in stored_submissions.columns:
            stored_submissions = stored_submissions.copy()
            stored_submissions["submitted_at_ts"] = stored_submissions["submitted_at"].map(_ts)
            cutoff = _ts(captured_at) - pd.Timedelta(hours=max(int(order_lookback_hours), 1))
            stored_submissions = stored_submissions[stored_submissions["submitted_at_ts"] >= cutoff]
            recent_submissions = [
                OrderSubmission(
                    symbol=str(row.symbol),
                    action=str(row.action),
                    status=str(row.status),
                    client_order_id=str(row.client_order_id),
                    order_id=None if pd.isna(row.order_id) else str(row.order_id),
                    submitted_notional=None if pd.isna(row.submitted_notional) else float(row.submitted_notional),
                    submitted_qty=None if pd.isna(row.submitted_qty) else float(row.submitted_qty),
                    raw_status=None if pd.isna(row.raw_status) else str(row.raw_status),
                    notes="" if pd.isna(row.notes) else str(row.notes),
                )
                for row in stored_submissions.itertuples(index=False)
            ]

    filled = partial = rejected = queued = 0
    for submission in recent_submissions:
        status = order_status_by_id.get(str(submission.order_id or "")) or order_status_by_client.get(submission.client_order_id) or submission.raw_status or submission.status
        status_lc = str(status).lower()
        if status_lc in {"filled"}:
            filled += 1
        elif status_lc in {"partially_filled"}:
            partial += 1
        elif status_lc in {"canceled", "cancelled", "rejected", "expired", "failed"}:
            rejected += 1
        else:
            queued += 1

    summary = BrokerReconciliationSummary(
        captured_at=captured_at,
        gross_exposure_pct=float(gross_exposure_pct),
        instruction_count=len(plan.instructions) if plan is not None else 0,
        blocker_count=len(plan.blockers) if plan is not None else 0,
        submitted_order_count=len(recent_submissions),
        filled_order_count=filled,
        partially_filled_order_count=partial,
        rejected_order_count=rejected,
        queued_order_count=queued,
        selected_symbols=selected_symbols,
        actual_symbols=actual_symbols,
        missing_symbols=missing_symbols,
        unexpected_symbols=unexpected_symbols,
    )
    store.append_records("broker_reconciliations", [summary.to_row()])
    record_portfolio_nav(store=store, account=account, captured_at=captured_at)
    return summary


def record_trade_decision_run(
    store: LocalResearchStore,
    *,
    plan: ExecutionPlan,
    analyses: Iterable[CompanyAnalysis],
    readiness: ReadinessReport,
    settings: VNextSettings,
    account: BrokerAccount,
    submit_requested: bool,
    submit_attempted: bool,
    submissions: Iterable[OrderSubmission] | None = None,
    reconciliation: BrokerReconciliationSummary | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    analyses = list(analyses)
    submissions = list(submissions or [])
    analysis_by_symbol = {analysis.snapshot.ticker: analysis for analysis in analyses}
    submission_by_symbol = {submission.symbol: submission for submission in submissions}
    validation = load_best_validation_payload(store) if store is not None else {}
    decision_rows: list[dict[str, Any]] = []
    decisions_payload: list[dict[str, Any]] = []

    for instruction in plan.instructions:
        analysis = analysis_by_symbol.get(instruction.symbol)
        submission = submission_by_symbol.get(instruction.symbol)
        special_situation = None
        primary_event_type = None
        primary_event_bucket = None
        snapshot_as_of = instruction.as_of
        if analysis is not None:
            special_situation = str(analysis.snapshot.metadata.get("special_situation") or "") or None
            primary_event_type = analysis.signal.primary_event_type
            primary_event_bucket = analysis.signal.primary_event_bucket
            snapshot_as_of = analysis.snapshot.as_of

        row = {
            "generated_at": plan.generated_at,
            "symbol": instruction.symbol,
            "action": instruction.action,
            "side": instruction.side,
            "scenario": instruction.scenario,
            "execution_profile": instruction.execution_profile,
            "readiness_status": plan.readiness_status,
            "validation_decision": str(
                validation.get("promotion_decision")
                or plan.exposure_governor.get("validation_decision")
                or ""
            ),
            "gross_cap_pct": float(plan.gross_cap_pct),
            "gross_cap_multiplier": float(plan.gross_cap_multiplier),
            "confidence": float(instruction.confidence),
            "target_weight": float(instruction.target_weight),
            "scaled_target_weight": float(instruction.scaled_target_weight),
            "target_notional": float(instruction.target_notional),
            "current_notional": float(instruction.current_notional),
            "delta_notional": float(instruction.delta_notional),
            "internal_upside_pct": None if instruction.internal_upside_pct is None else float(instruction.internal_upside_pct),
            "floor_support_pct": None if instruction.floor_support_pct is None else float(instruction.floor_support_pct),
            "primary_event_type": primary_event_type,
            "primary_event_bucket": primary_event_bucket,
            "special_situation": special_situation,
            "snapshot_as_of": snapshot_as_of,
            "submitted": bool(submission is not None and submission.status == "submitted"),
            "submission_status": None if submission is None else submission.status,
            "order_id": None if submission is None else submission.order_id,
            "client_order_id": None if submission is None else submission.client_order_id,
            "rationale": json.dumps(instruction.rationale),
        }
        decision_rows.append(row)
        payload_row = dict(row)
        payload_row["rationale"] = list(instruction.rationale)
        decisions_payload.append(payload_row)

    payload = {
        "generated_at": plan.generated_at,
        "submit_requested": bool(submit_requested),
        "submit_attempted": bool(submit_attempted),
        "readiness": {
            "status": readiness.status,
            "blockers": list(readiness.blockers),
            "warnings": list(readiness.warnings),
        },
        "account": {
            "account_id": account.account_id,
            "equity": float(account.equity),
            "buying_power": float(account.buying_power),
            "cash": float(account.cash),
            "paper": bool(account.paper),
        },
        "settings": settings.public_metadata(),
        "plan": {
            "selected_symbols": list(plan.selected_symbols),
            "blockers": list(plan.blockers),
            "warnings": list(plan.warnings),
            "gross_cap_pct": float(plan.gross_cap_pct),
            "gross_cap_multiplier": float(plan.gross_cap_multiplier),
            "deployable_notional": float(plan.deployable_notional),
            "exposure_governor": dict(plan.exposure_governor),
        },
        "validation": validation,
        "reconciliation": None if reconciliation is None else asdict(reconciliation),
        "submissions": [submission.to_row() for submission in submissions],
        "decisions": decisions_payload,
        "error": error,
    }
    safe_key = f"trade_decision_run_{plan.generated_at.replace(':', '-')}"
    store.write_raw_payload("trade_decision_runs", safe_key, payload)
    if decision_rows:
        store.append_records("trade_decisions", decision_rows)
    return payload


def _latest_run(store: LocalResearchStore, job_name: str) -> dict[str, Any] | None:
    runs = store.read_table("pipeline_runs")
    if runs.empty or "job_name" not in runs.columns:
        return None
    subset = runs[runs["job_name"] == job_name].copy()
    if subset.empty:
        return None
    subset["finished_at_ts"] = subset["finished_at"].map(_ts)
    subset = subset.sort_values("finished_at_ts")
    row = subset.iloc[-1]
    return {
        "status": str(row.get("status") or ""),
        "started_at": str(row.get("started_at") or ""),
        "finished_at": str(row.get("finished_at") or ""),
        "notes": str(row.get("notes") or ""),
        "metrics": row.get("metrics"),
    }


def write_autonomy_health_snapshot(
    store: LocalResearchStore,
    settings: VNextSettings,
    *,
    readiness: ReadinessReport | None = None,
) -> dict[str, Any]:
    readiness = readiness or build_readiness_report(store=store, settings=settings)
    validation = load_best_validation_payload(store)
    validation_age_days = validation_payload_age_days(validation)
    validation_decision = None
    if isinstance(validation, dict):
        raw_decision = validation.get("promotion_decision") or validation.get("decision")
        if isinstance(raw_decision, str) and raw_decision.strip():
            validation_decision = raw_decision.strip()

    latest_trade_run = _latest_run(store, "trade_vnext")
    latest_evaluate_run = _latest_run(store, "evaluate_vnext")
    reconciliations = store.read_table("broker_reconciliations")
    latest_reconciliation = None
    if not reconciliations.empty and "captured_at" in reconciliations.columns:
        reconciliations = reconciliations.copy()
        reconciliations["captured_at_ts"] = reconciliations["captured_at"].map(_ts)
        latest_reconciliation = reconciliations.sort_values("captured_at_ts").iloc[-1].to_dict()

    snapshots = store.read_table("company_snapshots")
    zero_market_cap_active = 0
    if not snapshots.empty and "ticker" in snapshots.columns and "as_of" in snapshots.columns:
        snapshots = snapshots.copy()
        snapshots["as_of_ts"] = snapshots["as_of"].map(_ts)
        snapshots = snapshots.sort_values("as_of_ts").drop_duplicates(subset=["ticker"], keep="last")
        market_caps = pd.to_numeric(snapshots.get("market_cap"), errors="coerce").fillna(0.0)
        zero_market_cap_active = int((market_caps <= 0.0).sum())

    blockers = list(readiness.blockers)
    warnings = list(readiness.warnings)
    if validation_age_days is not None and validation_age_days > int(settings.validation_max_age_days):
        blockers.append(f"Validation audit is stale at {validation_age_days}d.")
    if validation_decision == "do_not_promote":
        warnings.append("Promotion decision remains do_not_promote.")
    if zero_market_cap_active > 0:
        warnings.append(f"{zero_market_cap_active} latest snapshots have zero market cap and need review.")
    if latest_trade_run is not None and latest_trade_run["status"] == "failed":
        blockers.append("Latest trade_vnext run failed.")
    if latest_reconciliation is not None:
        missing = _parse_listish(latest_reconciliation.get("missing_symbols"))
        unexpected = _parse_listish(latest_reconciliation.get("unexpected_symbols"))
        if missing:
            warnings.append(f"Broker is missing {len(missing)} planned symbols.")
        if unexpected:
            warnings.append(f"Broker is carrying {len(unexpected)} unexpected symbols.")

    if blockers:
        status = "blocked"
    elif warnings:
        status = "degraded"
    else:
        status = "healthy"

    payload = {
        "captured_at": utc_now_iso(),
        "status": status,
        "readiness_status": readiness.status,
        "validation_decision": validation_decision,
        "validation_age_days": validation_age_days,
        "latest_snapshot_age_hours": readiness.latest_snapshot_age_hours,
        "latest_trade_run": latest_trade_run,
        "latest_evaluate_run": latest_evaluate_run,
        "latest_reconciliation": latest_reconciliation,
        "zero_market_cap_active": zero_market_cap_active,
        "blockers": blockers,
        "warnings": warnings,
    }
    payload = _json_safe(payload)
    store.write_raw_payload("autonomy_health", "latest", payload)
    store.append_records(
        "autonomy_health",
        [
            {
                "captured_at": payload["captured_at"],
                "status": status,
                "readiness_status": readiness.status,
                "validation_decision": validation_decision,
                "validation_age_days": validation_age_days,
                "latest_snapshot_age_hours": readiness.latest_snapshot_age_hours,
                "zero_market_cap_active": zero_market_cap_active,
                "blocker_count": len(blockers),
                "warning_count": len(warnings),
                "latest_trade_status": None if latest_trade_run is None else latest_trade_run["status"],
                "latest_evaluate_status": None if latest_evaluate_run is None else latest_evaluate_run["status"],
            }
        ],
    )
    return payload
