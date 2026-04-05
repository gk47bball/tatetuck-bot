from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any

import pandas as pd

from .settings import VNextSettings
from .storage import LocalResearchStore


def _parse_jsonish(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}
    return {}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(parsed):
        return default
    return parsed


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def validation_payload_timestamp(payload: dict[str, Any] | None) -> pd.Timestamp | None:
    if not isinstance(payload, dict):
        return None
    for key in ("generated_at", "evaluated_at", "created_at", "finished_at", "as_of"):
        ts = pd.to_datetime(payload.get(key), errors="coerce", utc=True, format="mixed")
        if pd.isna(ts):
            continue
        return ts.tz_convert(None)
    return None


def validation_payload_age_days(
    payload: dict[str, Any] | None,
    *,
    now: datetime | None = None,
) -> int | None:
    ts = validation_payload_timestamp(payload)
    if ts is None:
        return None
    reference = pd.Timestamp(now or datetime.now(timezone.utc))
    if reference.tzinfo is not None:
        reference = reference.tz_convert(None)
    return max((reference.date() - ts.date()).days, 0)


def latest_successful_evaluate_payload(store: LocalResearchStore) -> dict[str, Any]:
    runs = store.read_table("pipeline_runs")
    if runs.empty or "job_name" not in runs.columns or "status" not in runs.columns:
        return {}
    subset = runs[
        (runs["job_name"].astype(str) == "evaluate_vnext")
        & (runs["status"].astype(str) == "success")
    ].copy()
    if subset.empty:
        return {}
    subset["finished_at_ts"] = pd.to_datetime(
        subset.get("finished_at"),
        errors="coerce",
        utc=True,
        format="mixed",
    )
    subset = subset.sort_values("finished_at_ts")
    row = subset.iloc[-1]
    metrics = _parse_jsonish(row.get("metrics"))
    if not isinstance(metrics, dict):
        metrics = {}
    stamp = row.get("finished_at") or row.get("started_at")
    payload = dict(metrics)
    payload.setdefault("generated_at", stamp)
    payload.setdefault("evaluated_at", stamp)
    payload.setdefault("created_at", stamp)
    payload.setdefault("finished_at", row.get("finished_at"))
    payload.setdefault("source_job", "evaluate_vnext")
    return payload


def _merge_payloads(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    merged = dict(secondary)
    merged.update(primary)
    for key, value in secondary.items():
        if key not in merged or _is_missing(merged.get(key)):
            merged[key] = value
    return merged


def load_best_validation_payload(store: LocalResearchStore | None) -> dict[str, Any]:
    if store is None:
        return {}
    raw_payload = store.read_latest_raw_payload("validation_audits", "latest_walkforward_audit")
    raw_payload = raw_payload if isinstance(raw_payload, dict) else {}
    run_payload = latest_successful_evaluate_payload(store)
    if not raw_payload:
        return run_payload
    if not run_payload:
        return raw_payload
    raw_ts = validation_payload_timestamp(raw_payload)
    run_ts = validation_payload_timestamp(run_payload)
    if raw_ts is None and run_ts is None:
        return _merge_payloads(run_payload, raw_payload)
    if raw_ts is None:
        return _merge_payloads(run_payload, raw_payload)
    if run_ts is None:
        return _merge_payloads(raw_payload, run_payload)
    if run_ts >= raw_ts:
        return _merge_payloads(run_payload, raw_payload)
    return _merge_payloads(raw_payload, run_payload)


def derive_promotion_record(
    payload: dict[str, Any] | None,
    *,
    settings: VNextSettings | None = None,
    now: datetime | None = None,
    model_name: str = "event_driven_ensemble",
    model_version: str = "v3",
) -> dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}
    settings = settings or VNextSettings.from_env()
    created_at = (now or datetime.now(timezone.utc)).isoformat()

    windows = int(_as_float(payload.get("num_windows", payload.get("windows")), 0.0))
    rows = int(_as_float(payload.get("num_rows", payload.get("rows")), 0.0))
    rank_ic = _as_float(payload.get("rank_ic"), 0.0)
    strict_rank_ic = _as_float(payload.get("strict_rank_ic"), 0.0)
    hit_rate = _as_float(payload.get("hit_rate"), 0.0)
    spread = _as_float(payload.get("cost_adjusted_top_bottom_spread", payload.get("top_bottom_spread")), 0.0)
    pm_context_coverage = _as_float(payload.get("pm_context_coverage"), 0.0)
    exact_primary_event_rate = _as_float(payload.get("exact_primary_event_rate"), 0.0)
    synthetic_primary_event_rate = _as_float(payload.get("synthetic_primary_event_rate"), 1.0)
    rank_ic_ci_low = payload.get("rank_ic_ci_low")
    spread_ci_low = payload.get("top_bottom_spread_ci_low")
    leakage_passed = bool(payload.get("leakage_passed"))
    institutional_blockers = _parse_jsonish(payload.get("institutional_blockers"))
    if not isinstance(institutional_blockers, list):
        institutional_blockers = []

    paper_trade_blockers: list[str] = []
    if not leakage_passed:
        paper_trade_blockers.append("Leakage audit is not passing.")
    paper_trade_blockers.extend(str(item) for item in institutional_blockers if str(item).strip())
    if windows < int(settings.min_walkforward_windows):
        paper_trade_blockers.append(
            f"Only {windows} walk-forward windows are available; need at least {settings.min_walkforward_windows}."
        )
    if rows < int(settings.min_matured_return_rows):
        paper_trade_blockers.append(
            f"Only {rows} labeled rows are available; need at least {settings.min_matured_return_rows} for paper-trade promotion."
        )
    if rank_ic < 0.10:
        paper_trade_blockers.append("Rank IC is still below the 0.10 paper-trading floor.")
    if spread <= 0.0:
        paper_trade_blockers.append("Cost-adjusted top/bottom spread is not yet positive.")
    if exact_primary_event_rate < 0.75:
        paper_trade_blockers.append("Exact primary event coverage is still below the 75% paper-trading floor.")
    if synthetic_primary_event_rate > 0.20:
        paper_trade_blockers.append("Synthetic primary event reliance is still above the 20% paper-trading ceiling.")

    paper_trade_ready = not paper_trade_blockers

    a_grade_blockers = list(paper_trade_blockers)
    if windows < 24:
        a_grade_blockers.append("Need at least 24 walk-forward windows for A-grade promotion.")
    if rank_ic < 0.15:
        a_grade_blockers.append("Overall rank IC is still below the 0.15 A-grade floor.")
    if strict_rank_ic < 0.05:
        a_grade_blockers.append("Strict exact-event rank IC is still below the 0.05 A-grade floor.")
    if spread < 0.10:
        a_grade_blockers.append("Cost-adjusted top/bottom spread is still below the 10% A-grade floor.")
    if _is_missing(rank_ic_ci_low) or _as_float(rank_ic_ci_low, -1.0) <= 0.0:
        a_grade_blockers.append("Rank-IC confidence band still has a non-positive lower bound.")
    if _is_missing(spread_ci_low) or _as_float(spread_ci_low, -1.0) <= 0.0:
        a_grade_blockers.append("Spread confidence band still has a non-positive lower bound.")

    a_grade_ready = not a_grade_blockers

    if a_grade_ready:
        decision = "promote"
        rationale = "Latest validation clears the A-grade promotion gate."
        blockers = []
    elif paper_trade_ready:
        decision = "paper_trade_ready"
        rationale = (
            "Latest validation clears the paper-trading gate for tomorrow's open, "
            "but the stricter A-grade capital-promotion bar is still not met."
        )
        blockers = a_grade_blockers
    else:
        decision = "do_not_promote"
        rationale = "Latest validation still fails the paper-trading promotion gate."
        blockers = paper_trade_blockers

    a_grade_gates = {
        "paper_trading": {
            "passed": paper_trade_ready,
            "reason": "Paper-trading gate is clear." if paper_trade_ready else "Paper-trading gate is still blocked.",
            "rows": float(rows),
            "windows": float(windows),
            "rank_ic": rank_ic,
            "hit_rate": hit_rate,
            "cost_adjusted_top_bottom_spread": spread,
            "pm_context_coverage": pm_context_coverage,
            "exact_primary_event_rate": exact_primary_event_rate,
            "synthetic_primary_event_rate": synthetic_primary_event_rate,
            "blockers": paper_trade_blockers,
        },
        "overall_model": {
            "passed": a_grade_ready,
            "reason": "A-grade promotion gate is clear." if a_grade_ready else "A-grade promotion still requires stronger confidence bounds.",
            "rows": float(rows),
            "windows": float(windows),
            "rank_ic": rank_ic,
            "strict_rank_ic": strict_rank_ic,
            "cost_adjusted_top_bottom_spread": spread,
            "rank_ic_ci_low": None if _is_missing(rank_ic_ci_low) else _as_float(rank_ic_ci_low),
            "spread_ci_low": None if _is_missing(spread_ci_low) else _as_float(spread_ci_low),
            "blockers": a_grade_blockers,
        },
        "a_grade_ready": {
            "passed": a_grade_ready,
            "reason": "A-grade promotion is clear." if a_grade_ready else "A-grade promotion is still blocked by one or more validation gates.",
            "blockers": a_grade_blockers,
        },
    }

    return {
        "created_at": created_at,
        "source_job": str(payload.get("source_job") or "evaluate_vnext"),
        "model_name": model_name,
        "model_version": model_version,
        "decision": decision,
        "rationale": rationale,
        "blockers": blockers,
        "num_windows": windows,
        "rank_ic": rank_ic,
        "cost_adjusted_top_bottom_spread": spread,
        "strict_outcome_label_coverage": exact_primary_event_rate,
        "exact_primary_event_rate": exact_primary_event_rate,
        "synthetic_primary_event_rate": synthetic_primary_event_rate,
        "leakage_passed": leakage_passed,
        "a_grade_gates": a_grade_gates,
    }


def write_model_promotion(
    store: LocalResearchStore,
    payload: dict[str, Any] | None,
    *,
    settings: VNextSettings | None = None,
    now: datetime | None = None,
    model_name: str = "event_driven_ensemble",
    model_version: str = "v3",
) -> dict[str, Any]:
    record = derive_promotion_record(
        payload,
        settings=settings,
        now=now,
        model_name=model_name,
        model_version=model_version,
    )
    store.append_records("model_promotions", [record])
    return record
