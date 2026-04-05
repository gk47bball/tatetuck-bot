from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import re
from typing import Any

import pandas as pd

from .graph import (
    apply_curated_company_overrides,
    canonical_program_name,
    curated_program_overlay,
    is_low_signal_program,
    refresh_snapshot_evidence,
    select_lead_trial,
    select_program_evidence,
)
from .replay import snapshot_from_dict
from .storage import LocalResearchStore
from .taxonomy import event_timing_priority, event_type_bucket, event_type_priority, is_synthetic_event, normalized_event_type
from .validation import load_best_validation_payload


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, set)):
        return False
    try:
        missing = pd.isna(value)
    except TypeError:
        return False
    if hasattr(missing, "all") and not isinstance(missing, (bool, int)):
        try:
            return bool(missing.all())
        except Exception:
            return False
    return bool(missing)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if _is_missing(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    return int(round(_safe_float(value, float(default))))


def _ts(value: Any) -> pd.Timestamp | pd.NaT:
    ts = pd.to_datetime(value, errors="coerce", utc=True, format="mixed")
    if pd.isna(ts):
        return pd.NaT
    return ts.tz_convert(None)


def _parse_jsonish(value: Any, default: Any) -> Any:
    if _is_missing(value):
        return default
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return default
    return default


def _parse_list(value: Any) -> list[Any]:
    parsed = _parse_jsonish(value, [])
    return parsed if isinstance(parsed, list) else []


def _parse_dict(value: Any) -> dict[str, Any]:
    parsed = _parse_jsonish(value, {})
    return parsed if isinstance(parsed, dict) else {}


def _iso(value: Any) -> str | None:
    ts = _ts(value)
    if pd.isna(ts):
        return None
    return ts.isoformat()


def _coalesce(*values: Any) -> Any:
    for value in values:
        if isinstance(value, str):
            if value.strip():
                return value
            continue
        if not _is_missing(value):
            return value
    return None


def _days_old(timestamp: Any, now: datetime) -> int | None:
    ts = _ts(timestamp)
    if pd.isna(ts):
        return None
    return max((now.date() - ts.date()).days, 0)


def _latest_cluster(frame: pd.DataFrame, column: str, window_minutes: int = 10) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return frame.iloc[0:0].copy()
    subset = frame.dropna(subset=[column]).copy()
    if subset.empty:
        return subset
    latest = subset[column].max()
    cutoff = latest - pd.Timedelta(minutes=window_minutes)
    return subset[subset[column] >= cutoff].copy()


def _rows_for_run(
    frame: pd.DataFrame,
    column: str,
    started_at: Any,
    finished_at: Any,
    *,
    fallback_to_latest: bool = True,
) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return frame.iloc[0:0].copy()
    subset = frame.dropna(subset=[column]).copy()
    if subset.empty:
        return subset
    start_ts = _ts(started_at)
    end_ts = _ts(finished_at)
    if pd.isna(start_ts) and pd.isna(end_ts):
        return _latest_cluster(subset, column) if fallback_to_latest else subset.iloc[0:0].copy()
    if pd.isna(start_ts):
        start_ts = end_ts - pd.Timedelta(minutes=15)
    if pd.isna(end_ts):
        end_ts = start_ts + pd.Timedelta(minutes=30)
    window_start = start_ts - pd.Timedelta(minutes=15)
    window_end = end_ts + pd.Timedelta(minutes=45)
    matched = subset[(subset[column] >= window_start) & (subset[column] <= window_end)].copy()
    if not matched.empty:
        return matched
    return _latest_cluster(subset, column) if fallback_to_latest else subset.iloc[0:0].copy()


def _latest_trade_run(store: LocalResearchStore) -> dict[str, Any] | None:
    runs = store.read_table("pipeline_runs")
    if runs.empty or "job_name" not in runs.columns:
        return None
    subset = runs[runs["job_name"] == "trade_vnext"].copy()
    if subset.empty:
        return None
    subset["finished_ts"] = subset["finished_at"].map(_ts)
    subset["started_ts"] = subset["started_at"].map(_ts)
    subset = subset.sort_values(["finished_ts", "started_ts"]).copy()
    row = subset.iloc[-1]
    return {
        "job_name": str(row.get("job_name") or "trade_vnext"),
        "status": str(row.get("status") or "unknown"),
        "started_at": _iso(row.get("started_at")),
        "finished_at": _iso(row.get("finished_at")),
        "duration_seconds": _safe_float(row.get("duration_seconds"), 0.0),
        "metrics": _parse_dict(row.get("metrics")),
        "config": _parse_dict(row.get("config")),
        "notes": str(row.get("notes") or ""),
    }


def _match_submission(
    submissions: pd.DataFrame,
    symbol: str,
    planned_at_ts: pd.Timestamp | pd.NaT,
) -> dict[str, Any] | None:
    if submissions.empty:
        return None
    subset = submissions[submissions["symbol"].astype(str) == str(symbol)].copy()
    if subset.empty:
        return None
    if not pd.isna(planned_at_ts):
        subset = subset[
            (subset["submitted_at_ts"] >= (planned_at_ts - pd.Timedelta(minutes=2)))
            & (subset["submitted_at_ts"] <= (planned_at_ts + pd.Timedelta(days=2)))
        ].copy()
        if subset.empty:
            return None
        subset["submission_distance"] = (subset["submitted_at_ts"] - planned_at_ts).abs()
        subset = subset.sort_values(["submission_distance", "submitted_at_ts"])
    else:
        subset = subset.sort_values("submitted_at_ts")
    row = subset.iloc[0]
    return {
        "status": str(row.get("status") or "unknown"),
        "submitted_at": _iso(row.get("submitted_at")),
        "order_id": None if _is_missing(row.get("order_id")) else str(row.get("order_id")),
        "client_order_id": None if _is_missing(row.get("client_order_id")) else str(row.get("client_order_id")),
        "notes": str(row.get("notes") or ""),
    }


def _match_feedback(feedback: pd.DataFrame, symbol: str, planned_at: Any) -> dict[str, Any] | None:
    if feedback.empty:
        return None
    subset = feedback[feedback["symbol"].astype(str) == str(symbol)].copy()
    if "planned_at" in subset.columns and not _is_missing(planned_at):
        subset = subset[subset["planned_at"].astype(str) == str(planned_at)].copy()
    if subset.empty:
        return None
    row = subset.sort_values("entry_ts").iloc[-1]
    return {
        "mark_to_market_net_return": _safe_float(row.get("mark_to_market_net_return"), 0.0),
        "mark_to_market_return": _safe_float(row.get("mark_to_market_return"), 0.0),
        "return_30d_net": None if _is_missing(row.get("return_30d_net")) else _safe_float(row.get("return_30d_net")),
        "return_90d_net": None if _is_missing(row.get("return_90d_net")) else _safe_float(row.get("return_90d_net")),
        "entry_anchor_source": str(row.get("entry_anchor_source") or ""),
    }


def _build_trade_rows(
    plans: pd.DataFrame,
    ledger: pd.DataFrame,
    submissions: pd.DataFrame,
    feedback: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()

    def append_row(source_row: pd.Series, source_type: str) -> None:
        symbol = str(source_row.get("symbol") or source_row.get("company_name") or "")
        planned_at = str(source_row.get("planned_at") or "")
        dedupe = (symbol, planned_at)
        if not symbol or dedupe in seen_keys:
            return
        seen_keys.add(dedupe)
        planned_at_ts = _ts(planned_at)
        submission = _match_submission(submissions, symbol, planned_at_ts)
        pnl = _match_feedback(feedback, symbol, planned_at)
        action = str(source_row.get("action") or "hold")
        stage = "submitted" if submission and submission["status"] == "submitted" else source_type
        rows.append(
            {
                "idea_key": f"trade:{symbol}:{planned_at or action}",
                "symbol": symbol,
                "company_name": str(source_row.get("company_name") or symbol),
                "action": action,
                "scenario": str(source_row.get("scenario") or ""),
                "company_state": None if _is_missing(source_row.get("company_state")) else str(source_row.get("company_state")),
                "setup_type": None if _is_missing(source_row.get("setup_type")) else str(source_row.get("setup_type")),
                "execution_profile": None if _is_missing(source_row.get("execution_profile")) else str(source_row.get("execution_profile")),
                "confidence": _safe_float(source_row.get("confidence"), 0.0),
                "target_weight": _safe_float(source_row.get("target_weight"), 0.0),
                "scaled_target_weight": _safe_float(source_row.get("scaled_target_weight"), _safe_float(source_row.get("target_weight"), 0.0)),
                "target_notional": _safe_float(source_row.get("target_notional"), 0.0),
                "delta_notional": _safe_float(source_row.get("delta_notional"), 0.0),
                "requested_notional": _safe_float(source_row.get("requested_notional"), _safe_float(source_row.get("notional"), 0.0)),
                "executable_notional": None if _is_missing(source_row.get("executable_notional")) else _safe_float(source_row.get("executable_notional")),
                "expected_slippage_bps": None if _is_missing(source_row.get("expected_slippage_bps")) else _safe_float(source_row.get("expected_slippage_bps")),
                "expected_round_trip_cost_bps": None if _is_missing(source_row.get("expected_round_trip_cost_bps")) else _safe_float(source_row.get("expected_round_trip_cost_bps")),
                "liquidity_fill_ratio": None if _is_missing(source_row.get("liquidity_fill_ratio")) else _safe_float(source_row.get("liquidity_fill_ratio")),
                "internal_upside_pct": None if _is_missing(source_row.get("internal_upside_pct")) else _safe_float(source_row.get("internal_upside_pct")),
                "floor_support_pct": None if _is_missing(source_row.get("floor_support_pct")) else _safe_float(source_row.get("floor_support_pct")),
                "as_of": _iso(source_row.get("as_of")),
                "planned_at": _iso(planned_at),
                "stage": stage,
                "submission": submission,
                "pnl": pnl,
                "rationale": _parse_list(source_row.get("rationale")),
            }
        )

    if not ledger.empty:
        for _, row in ledger.sort_values("planned_at_ts", ascending=False).iterrows():
            append_row(row, "simulated")
    if not plans.empty:
        for _, row in plans.sort_values("planned_at_ts", ascending=False).iterrows():
            append_row(row, "planned")

    deduped_rows = _dedupe_trade_rows(rows)
    symbols = sorted({row["symbol"] for row in deduped_rows})
    return deduped_rows, symbols


def _dedupe_trade_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    submitted_rows: dict[str, dict[str, Any]] = {}
    remaining: list[dict[str, Any]] = []
    for row in rows:
        submission = row.get("submission") or {}
        submission_key = submission.get("order_id") or submission.get("client_order_id")
        if not submission_key:
            remaining.append(row)
            continue
        existing = submitted_rows.get(str(submission_key))
        if existing is None or (_ts(row.get("planned_at")) > _ts(existing.get("planned_at"))):
            submitted_rows[str(submission_key)] = row

    deduped = [*submitted_rows.values(), *remaining]
    deduped.sort(key=lambda item: (_ts(item.get("planned_at")), item.get("symbol") or ""), reverse=True)
    return deduped


def _idea_long_sort_key(item: dict[str, Any]) -> tuple[bool, float, float, float]:
    return (
        bool(item.get("in_current_plan")),
        _safe_float(item.get("target_weight"), 0.0),
        _safe_float(item.get("expected_return"), 0.0),
        _safe_float(item.get("confidence"), 0.0),
    )


def _idea_short_sort_key(item: dict[str, Any]) -> tuple[float, float, float, bool]:
    return (
        abs(_safe_float(item.get("expected_return"), 0.0)),
        _safe_float(item.get("confidence"), 0.0),
        _safe_float(item.get("catalyst_success_prob"), 0.0),
        bool(item.get("in_current_plan")),
    )


def _blend_directional_ideas(
    company_items: list[dict[str, Any]],
    program_items: list[dict[str, Any]],
    direction: str,
    limit: int = 12,
) -> list[dict[str, Any]]:
    sorted_company = sorted(
        company_items,
        key=_idea_long_sort_key if direction == "long" else _idea_short_sort_key,
        reverse=True,
    )
    sorted_program = sorted(
        program_items,
        key=_idea_long_sort_key if direction == "long" else _idea_short_sort_key,
        reverse=True,
    )
    blended: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in [*sorted_company, *sorted_program]:
        symbol = str(item.get("symbol") or "")
        display_name = str(item.get("program_name") or item.get("company_name") or item.get("idea_key") or "")
        dedupe_key = (symbol, _normalize_match_key(display_name), str(item.get("direction") or direction))
        if not symbol or dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        blended.append(item)
        if len(blended) >= limit:
            break
    return blended


def _build_research_notice(
    freshness_days: int | None,
    oldest_days: int | None,
    company_count: int,
    program_count: int,
    mixed_vintage: bool,
) -> str:
    notes: list[str] = []
    if mixed_vintage and freshness_days is not None and oldest_days is not None:
        notes.append(f"mixed-vintage research deck spans {freshness_days}d to {oldest_days}d old snapshots")
    elif freshness_days is not None and freshness_days >= 7:
        lead = "archived research snapshot" if freshness_days >= 30 else "research snapshot"
        notes.append(f"{lead} is {freshness_days} days old")
    if company_count == 0 and program_count > 0:
        notes.append("company-level book is empty, so the deck is using program-level predictions")
    elif company_count < 4 and program_count > 0:
        notes.append("company-level book is thin, so the deck is topped up with program-level predictions")
    return " | ".join(notes)


def _normalize_match_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _match_tokens(value: Any) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", str(value or "").lower())
        if len(token) >= 3
    }


def _evidence_payload(items: list[Any], limit: int = 3) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for item in items[:limit]:
        if isinstance(item, dict):
            source = str(item.get("source") or "")
            source_id = str(item.get("source_id") or "")
            title = str(item.get("title") or "")
            excerpt = str(item.get("excerpt") or "")
            url = item.get("url")
            as_of = item.get("as_of")
            confidence = _safe_float(item.get("confidence"), 0.5)
        else:
            source = str(getattr(item, "source", "") or "")
            source_id = str(getattr(item, "source_id", "") or "")
            title = str(getattr(item, "title", "") or "")
            excerpt = str(getattr(item, "excerpt", "") or "")
            url = getattr(item, "url", None)
            as_of = getattr(item, "as_of", None)
            confidence = _safe_float(getattr(item, "confidence", 0.5), 0.5)
        payload.append(
            {
                "source": source,
                "source_id": source_id,
                "title": title,
                "excerpt": excerpt[:220],
                "url": None if _is_missing(url) else str(url),
                "as_of": None if _is_missing(as_of) else str(as_of),
                "confidence": confidence,
            }
        )
    return payload


def _evidence_sources(items: list[Any], limit: int = 3) -> list[str]:
    sources: list[str] = []
    seen: set[str] = set()
    for item in items:
        source = str(item.get("source") if isinstance(item, dict) else getattr(item, "source", "") or "")
        if not source or source in seen:
            continue
        seen.add(source)
        sources.append(source)
        if len(sources) >= limit:
            break
    return sources


def _deployment_status_for_company_idea(direction: str, stance: str, target_weight: float) -> tuple[bool, str, str]:
    stance_value = str(stance or "")
    if direction == "short" or stance_value == "short":
        return (
            False,
            "research_only_short",
            "Short thesis is research-only until borrow-aware short execution is enabled.",
        )
    if direction == "watch" or stance_value == "avoid" or target_weight <= 0.0:
        return (
            False,
            "watchlist_only",
            "This name is on the research/watchlist book, not the live PM deployment sleeve.",
        )
    return (
        True,
        "deployable_company_long",
        "Eligible for company-level PM deployment under the current long-only execution rules.",
    )


def _deployment_status_for_program_idea(direction: str) -> tuple[bool, str, str]:
    if direction == "short":
        return (
            False,
            "research_only_program_short",
            "Program-level short thesis is research-only; live execution does not route program shorts.",
        )
    return (
        False,
        "research_only_program",
        "Program prediction is a research sleeve; live execution still routes at the company level.",
    )


LISTING_SOURCE_PRIORITY = {
    "eodhd_exchange_symbols": 3,
    "sec_price_reconstruction": 2,
    "prepare_compatibility_layer": 1,
}


def _load_listing_contexts(store: LocalResearchStore, tickers: set[str]) -> dict[str, dict[str, Any]]:
    if not tickers:
        return {}
    frame = store.read_table("universe_membership")
    if frame.empty or "ticker" not in frame.columns:
        return {}
    subset = frame[frame["ticker"].astype(str).isin(sorted(tickers))].copy()
    if subset.empty:
        return {}
    subset["as_of_ts"] = subset["as_of"].map(_ts) if "as_of" in subset.columns else pd.NaT
    subset["source_rank"] = subset.get("membership_source", "").fillna("").astype(str).map(
        lambda value: LISTING_SOURCE_PRIORITY.get(value, 0)
    )
    subset = subset.sort_values(["ticker", "source_rank", "as_of_ts"], ascending=[True, True, True]).drop_duplicates(
        subset=["ticker"],
        keep="last",
    )
    contexts: dict[str, dict[str, Any]] = {}
    for row in subset.itertuples(index=False):
        contexts[str(row.ticker)] = {
            "is_delisted": bool(getattr(row, "is_delisted", False)),
            "membership_source": str(getattr(row, "membership_source", "") or ""),
            "exchange": None if _is_missing(getattr(row, "exchange", None)) else str(getattr(row, "exchange")),
            "security_type": None if _is_missing(getattr(row, "security_type", None)) else str(getattr(row, "security_type")),
            "listing_symbol": None if _is_missing(getattr(row, "listing_symbol", None)) else str(getattr(row, "listing_symbol")),
            "date_delisted": _iso(getattr(row, "date_delisted", None)),
            "as_of": _iso(getattr(row, "as_of", None)),
        }
    return contexts


def _snapshot_primary_event(snapshot) -> Any | None:
    if snapshot is None or not snapshot.catalyst_events:
        return None
    candidates = list(snapshot.catalyst_events)
    reference_ts = _ts(snapshot.as_of)
    if not pd.isna(reference_ts):
        upcoming = []
        recent_exact = []
        for event in candidates:
            event_ts = _ts(event.expected_date)
            if pd.isna(event_ts):
                continue
            if "T" not in str(event.expected_date or "") and len(str(event.expected_date or "")) <= 10:
                if event_ts.date() >= reference_ts.date():
                    upcoming.append(event)
                elif event_timing_priority(event.status, event.expected_date, event.title) >= 2 and event_ts.date() >= (reference_ts - pd.Timedelta(days=14)).date():
                    recent_exact.append(event)
            elif event_ts >= reference_ts:
                upcoming.append(event)
            elif event_timing_priority(event.status, event.expected_date, event.title) >= 2 and event_ts >= (reference_ts - pd.Timedelta(days=14)):
                recent_exact.append(event)
        if upcoming or recent_exact:
            candidates = [*recent_exact, *upcoming]
    state = str(snapshot.metadata.get("company_state") or "")
    if state in {"commercial_launch", "commercialized"}:
        strategic = [
            event
            for event in candidates
            if event_type_bucket(event.event_type, event.title) == "strategic"
            and not is_synthetic_event(event.status, event.title)
            and event.horizon_days <= 180
        ]
        exact_hard = [
            event
            for event in candidates
            if event_type_bucket(event.event_type, event.title) in {"clinical", "regulatory"}
            and not is_synthetic_event(event.status, event.title)
            and event.horizon_days <= 180
        ]
        if strategic:
            strategic_with_timing = [
                event
                for event in strategic
                if event_timing_priority(event.status, event.expected_date, event.title) >= 1
            ]
            candidates = strategic_with_timing or strategic
        elif exact_hard:
            candidates = exact_hard
        else:
            near_term = [event for event in candidates if event.horizon_days <= 90]
            if near_term:
                candidates = near_term
    return min(
        candidates,
        key=lambda event: (
            -event_timing_priority(event.status, event.expected_date, event.title),
            -event_type_priority(normalized_event_type(event.event_type, event.title) or event.event_type),
            event.horizon_days,
            -event.importance,
            event.crowdedness,
        ),
    )


def _integrity_override(
    idea: dict[str, Any],
    snapshot,
    program,
    listing_context: dict[str, Any] | None,
) -> None:
    blockers: list[str] = []
    market_cap = _safe_float(idea.get("market_cap"), 0.0)
    price_now = _safe_float(snapshot.metadata.get("price_now") if snapshot is not None else None, 0.0)

    if listing_context and bool(listing_context.get("is_delisted")):
        blockers.append("delisted_security")
    if idea.get("idea_level") == "company" and market_cap <= 0.0:
        blockers.append("zero_market_cap")
    if idea.get("idea_level") == "company" and market_cap <= 0.0 and price_now <= 0.0:
        blockers.append("missing_live_price")
    if idea.get("idea_level") == "program" and program is None and snapshot is not None and snapshot.programs:
        blockers.append("stale_program_mapping")
    if idea.get("idea_level") == "program" and str(idea.get("program_name") or "").endswith(":company"):
        blockers.append("synthetic_company_program")
    if program is not None and is_low_signal_program(program):
        blockers.append("low_signal_program")

    idea["integrity_flags"] = blockers
    idea["listing_status"] = listing_context or {}
    if not blockers:
        idea["surfaceable"] = True
        return

    if any(flag in blockers for flag in ("low_signal_program", "stale_program_mapping", "synthetic_company_program")) and len(blockers) == 1:
        idea["deployable"] = False
        idea["deployment_status"] = "research_artifact_program"
        if "stale_program_mapping" in blockers:
            idea["deployment_note"] = "Program prediction no longer maps to a current program in the latest snapshot, so it is suppressed from the PM deck."
        elif "synthetic_company_program" in blockers:
            idea["deployment_note"] = "Program prediction is just a company placeholder sleeve rather than a real asset-level thesis, so it is suppressed from the PM deck."
        else:
            idea["deployment_note"] = "Program maps to an observational, educational, screening, or natural-history study artifact rather than an investable drug catalyst."
    else:
        idea["deployable"] = False
        idea["deployment_status"] = "data_integrity_block"
        reasons: list[str] = []
        if "delisted_security" in blockers:
            reasons.append("ticker is flagged as delisted in the exchange membership feed")
        if "zero_market_cap" in blockers:
            reasons.append("snapshot market cap is zero")
        if "missing_live_price" in blockers:
            reasons.append("live price is missing")
        if "stale_program_mapping" in blockers:
            reasons.append("program prediction no longer maps to the latest snapshot")
        if "synthetic_company_program" in blockers:
            reasons.append("program prediction is only a synthetic company placeholder")
        if "low_signal_program" in blockers:
            reasons.append("program is a non-investable study artifact")
        idea["deployment_note"] = "; ".join(reasons)
    idea["surfaceable"] = False


def _load_snapshot_contexts(store: LocalResearchStore, tickers: set[str]) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    for ticker in sorted(tickers):
        payload = store.read_latest_raw_payload("snapshots", f"{ticker}_")
        if not isinstance(payload, dict):
            continue
        try:
            snapshot = snapshot_from_dict(payload)
            snapshot = apply_curated_company_overrides(snapshot)
            snapshot = refresh_snapshot_evidence(snapshot)
        except Exception:
            continue
        program_index = {_normalize_match_key(program.name): program for program in snapshot.programs}
        program_id_index = {str(program.program_id): program for program in snapshot.programs}
        contexts[ticker] = {
            "snapshot": snapshot,
            "program_index": program_index,
            "program_id_index": program_id_index,
        }
    return contexts


def _match_program_in_context(idea: dict[str, Any], context: dict[str, Any]):
    entity_id = str(idea.get("entity_id") or "")
    program_name = str(idea.get("program_name") or "")
    indication_tokens = _match_tokens(idea.get("indication"))
    phase = str(idea.get("phase") or "")

    if entity_id and entity_id in context["program_id_index"]:
        entity_match = context["program_id_index"][entity_id]
        if not program_name:
            return entity_match
        entity_tokens = _match_tokens(entity_match.name) | _match_tokens(canonical_program_name(entity_match))
        shared_name_tokens = len(_match_tokens(program_name) & entity_tokens)
        phase_match = bool(phase and str(entity_match.phase or "") == phase)
        if shared_name_tokens > 0 or phase_match:
            entity_score = (shared_name_tokens * 3.0) + (
                len(indication_tokens & _match_tokens(" ".join(entity_match.conditions))) * 2.0
            )
            if phase_match:
                entity_score += 1.5
        else:
            entity_score = 0.0
        if entity_score >= 1.5:
            return entity_match

    if program_name:
        exact = context["program_index"].get(_normalize_match_key(program_name))
        if exact is not None:
            return exact

    idea_tokens = _match_tokens(program_name)
    best_score = 0.0
    best_program = None
    for program in context["snapshot"].programs:
        program_tokens = _match_tokens(program.name)
        shared_program = len(idea_tokens & program_tokens)
        shared_condition = len(indication_tokens & _match_tokens(" ".join(program.conditions)))
        phase_match = bool(phase and str(program.phase or "") == phase)
        if shared_program == 0 and not phase_match:
            continue
        score = (shared_program * 3.0) + (shared_condition * 2.0)
        if phase_match:
            score += 1.5
        if score > best_score:
            best_score = score
            best_program = program
    return best_program if best_score >= 2.0 else None


def _enrich_idea_with_snapshot_context(
    idea: dict[str, Any],
    snapshot_contexts: dict[str, dict[str, Any]],
    listing_contexts: dict[str, dict[str, Any]],
) -> None:
    context = snapshot_contexts.get(str(idea.get("symbol") or ""))
    if context is None:
        return

    snapshot = context["snapshot"]
    program = _match_program_in_context(idea, context) if (idea.get("program_name") or idea.get("entity_id")) else None

    evidence_items = idea.get("evidence_items") or []
    evidence_count = _safe_int(idea.get("evidence_count"), 0)
    evidence_sources = idea.get("evidence_sources") or []
    matched_program_evidence = select_program_evidence(program, snapshot) if program is not None else []
    if matched_program_evidence:
        fallback_evidence = matched_program_evidence
    elif program is not None and program.evidence:
        fallback_evidence = list(program.evidence)
    else:
        fallback_evidence = list(snapshot.evidence)
    if not evidence_items and fallback_evidence:
        evidence_items = _evidence_payload(fallback_evidence)
        evidence_count = len(fallback_evidence)
        evidence_sources = _evidence_sources(fallback_evidence)

    idea["evidence_items"] = evidence_items
    idea["evidence_count"] = evidence_count
    idea["evidence_sources"] = evidence_sources
    idea["snapshot_as_of"] = snapshot.as_of
    idea["snapshot_program_count"] = len(snapshot.programs)
    idea["snapshot_catalyst_count"] = len(snapshot.catalyst_events)
    idea["company_name"] = str(snapshot.company_name or idea.get("company_name") or idea.get("symbol") or "")
    idea["special_situation"] = str(snapshot.metadata.get("special_situation") or "")
    idea["special_situation_label"] = str(snapshot.metadata.get("special_situation_label") or "")
    idea["special_situation_reason"] = str(snapshot.metadata.get("special_situation_reason") or "")
    idea["driver_label"] = str(snapshot.metadata.get("driver_label") or "")
    idea["driver_indication"] = str(snapshot.metadata.get("driver_indication") or "")
    snapshot_risk_flags = _parse_list(snapshot.metadata.get("bear_case_flags"))
    combined_risk_flags: list[str] = []
    for flag in [*_parse_list(idea.get("risk_flags")), *snapshot_risk_flags]:
        if not flag:
            continue
        flag_text = str(flag)
        if flag_text not in combined_risk_flags:
            combined_risk_flags.append(flag_text)
    idea["risk_flags"] = combined_risk_flags
    idea["price_now"] = None if _is_missing(snapshot.metadata.get("price_now")) else _safe_float(snapshot.metadata.get("price_now"))
    if _safe_float(idea.get("market_cap"), 0.0) <= 0.0 and _safe_float(snapshot.market_cap, 0.0) > 0.0:
        idea["market_cap"] = _safe_float(snapshot.market_cap, 0.0)
    if _safe_float(idea.get("revenue"), 0.0) <= 0.0 and _safe_float(snapshot.revenue, 0.0) > 0.0:
        idea["revenue"] = _safe_float(snapshot.revenue, 0.0)
    if idea.get("idea_level") == "company":
        confidence_haircut = _safe_float(snapshot.metadata.get("confidence_haircut"), 1.0)
        expected_return_haircut = _safe_float(snapshot.metadata.get("expected_return_haircut"), 1.0)
        catalyst_success_prob_haircut = _safe_float(snapshot.metadata.get("catalyst_success_prob_haircut"), 1.0)
        if confidence_haircut < 1.0:
            idea["confidence"] = max(0.0, min(_safe_float(idea.get("confidence"), 0.0) * confidence_haircut, 0.99))
        if expected_return_haircut < 1.0:
            idea["expected_return"] = _safe_float(idea.get("expected_return"), 0.0) * expected_return_haircut
        if catalyst_success_prob_haircut < 1.0:
            idea["catalyst_success_prob"] = max(
                0.0,
                min(_safe_float(idea.get("catalyst_success_prob"), 0.0) * catalyst_success_prob_haircut, 0.99),
            )

    if idea.get("idea_level") == "company":
        primary_event = _snapshot_primary_event(snapshot)
        if primary_event is not None:
            normalized_type = normalized_event_type(primary_event.event_type, primary_event.title) or str(primary_event.event_type)
            idea["primary_event_type"] = normalized_type
            idea["primary_event_bucket"] = event_type_bucket(normalized_type, primary_event.title)
            idea["primary_event_status"] = str(primary_event.status or "")
            idea["primary_event_date"] = _iso(primary_event.expected_date)
            idea["primary_event_exact"] = bool(event_timing_priority(primary_event.status, primary_event.expected_date, primary_event.title) >= 2)
            idea["catalyst_title"] = str(primary_event.title or "")

    if program is not None:
        overlay = curated_program_overlay(program)
        idea["program_name"] = str(overlay.get("name") or canonical_program_name(program))
        idea["phase"] = str(overlay.get("phase") or program.phase or "") or idea.get("phase")
        idea["modality"] = str(program.modality or "") or idea.get("modality")
        overlay_conditions = list(overlay.get("conditions") or program.conditions)
        if overlay_conditions:
            idea["indication"] = overlay_conditions[0]
        lead_trial = select_lead_trial(program)
        idea["program_trial_count"] = len(program.trials)
        idea["program_conditions"] = list(overlay_conditions[:3])
        idea["lead_trial_title"] = None if lead_trial is None else lead_trial.title
        idea["lead_trial_phase"] = None if lead_trial is None else str(overlay.get("phase") or lead_trial.phase)
        idea["lead_trial_status"] = None if lead_trial is None else lead_trial.status
        idea["lead_trial_primary_outcomes"] = [] if lead_trial is None else list(lead_trial.primary_outcomes[:3])
        if program.catalyst_events:
            primary_event = program.catalyst_events[0]
            normalized_type = normalized_event_type(primary_event.event_type, primary_event.title) or str(primary_event.event_type)
            if str(overlay.get("phase") or "") == "APPROVED" and normalized_type in {"phase1_readout", "phase2_readout", "phase3_readout", "clinical_readout"}:
                normalized_type = "commercial_update"
            idea["primary_event_type"] = normalized_type
            idea["primary_event_bucket"] = event_type_bucket(normalized_type, primary_event.title)
            idea["primary_event_status"] = str(primary_event.status or "")
            idea["primary_event_date"] = _iso(primary_event.expected_date)
            idea["primary_event_exact"] = bool(event_timing_priority(primary_event.status, primary_event.expected_date, primary_event.title) >= 2)
            idea["catalyst_title"] = str(primary_event.title or "")
        scenario_parts = [str(idea.get("program_name") or "")]
        if idea.get("primary_event_type"):
            scenario_parts.append(str(idea["primary_event_type"]).replace("_", " "))
        if idea.get("indication"):
            scenario_parts.append(f"in {idea['indication']}")
        scenario = " ".join(part for part in scenario_parts if part).strip()
        if scenario:
            idea["scenario"] = scenario
            idea["rationale_preview"] = scenario

    listing_context = listing_contexts.get(str(idea.get("symbol") or ""))
    _integrity_override(idea, snapshot, program, listing_context)
    if program is None:
        return


def _current_plan_payload(store: LocalResearchStore, now: datetime) -> dict[str, Any]:
    run = _latest_trade_run(store)
    plans = store.read_table("order_plans")
    ledger = store.read_table("order_ledger")
    submissions = store.read_table("order_submissions")
    feedback = store.read_table("execution_feedback")

    for frame, column in (
        (plans, "planned_at"),
        (ledger, "planned_at"),
        (submissions, "submitted_at"),
    ):
        if not frame.empty and column in frame.columns:
            frame[f"{column}_ts"] = frame[column].map(_ts)

    if not feedback.empty and "entry_ts" in feedback.columns:
        feedback = feedback.copy()
        feedback["entry_ts"] = feedback["entry_ts"].map(_ts)

    recent_plans = _latest_cluster(plans, "planned_at_ts")
    recent_ledger = _latest_cluster(ledger, "planned_at_ts")
    recent_submissions = _latest_cluster(submissions, "submitted_at_ts")

    if run is not None:
        plans = _rows_for_run(plans, "planned_at_ts", run["started_at"], run["finished_at"], fallback_to_latest=False)
        ledger = _rows_for_run(ledger, "planned_at_ts", run["started_at"], run["finished_at"], fallback_to_latest=False)
        submissions = _rows_for_run(submissions, "submitted_at_ts", run["started_at"], run["finished_at"], fallback_to_latest=False)
    else:
        plans = recent_plans
        ledger = recent_ledger
        submissions = recent_submissions

    trade_rows, symbols = _build_trade_rows(plans=plans, ledger=ledger, submissions=submissions, feedback=feedback)
    recent_trade_rows, recent_symbols = _build_trade_rows(
        plans=recent_plans,
        ledger=recent_ledger,
        submissions=recent_submissions,
        feedback=feedback,
    )
    buys = [row for row in trade_rows if row["action"] == "buy"]
    sells = [row for row in trade_rows if row["action"] == "sell"]
    holds = [row for row in trade_rows if row["action"] == "hold"]

    gross_target_weight = sum(max(row["scaled_target_weight"], 0.0) for row in buys)
    gross_target_notional = sum(max(row["target_notional"], 0.0) for row in buys)

    nav = store.read_table("portfolio_nav")
    nav_payload: dict[str, Any] | None = None
    if not nav.empty:
        nav = nav.copy()
        nav["captured_at_ts"] = nav["captured_at"].map(_ts)
        nav = nav.sort_values("captured_at_ts")
        latest = nav.iloc[-1]
        nav_payload = {
            "captured_at": _iso(latest.get("captured_at")),
            "equity": _safe_float(latest.get("equity"), 0.0),
            "buying_power": _safe_float(latest.get("buying_power"), 0.0),
            "cash": _safe_float(latest.get("cash"), 0.0),
            "paper": bool(latest.get("paper")) if not _is_missing(latest.get("paper")) else True,
            "history": [
                {
                    "captured_at": _iso(row.captured_at),
                    "equity": _safe_float(row.equity, 0.0),
                    "cash": _safe_float(row.cash, 0.0),
                }
                for row in nav.tail(20).itertuples(index=False)
            ],
        }

    reconciliations = store.read_table("broker_reconciliations")
    reconciliation_payload: dict[str, Any] | None = None
    if not reconciliations.empty:
        reconciliations = reconciliations.copy()
        reconciliations["captured_at_ts"] = reconciliations["captured_at"].map(_ts)
        latest = reconciliations.sort_values("captured_at_ts").iloc[-1]
        reconciliation_payload = {
            "captured_at": _iso(latest.get("captured_at")),
            "gross_exposure_pct": _safe_float(latest.get("gross_exposure_pct"), 0.0),
            "instruction_count": _safe_int(latest.get("instruction_count"), 0),
            "blocker_count": _safe_int(latest.get("blocker_count"), 0),
            "selected_symbols": _parse_list(latest.get("selected_symbols")),
            "actual_symbols": _parse_list(latest.get("actual_symbols")),
            "missing_symbols": _parse_list(latest.get("missing_symbols")),
            "unexpected_symbols": _parse_list(latest.get("unexpected_symbols")),
        }

    return {
        "run": run,
        "symbols": symbols,
        "recent_symbols": recent_symbols,
        "freshness_days": _days_old(run["finished_at"], now) if run is not None else None,
        "gross_target_weight": gross_target_weight,
        "gross_target_notional": gross_target_notional,
        "buy_orders": buys,
        "sell_orders": sells,
        "hold_orders": holds,
        "trade_rows": trade_rows[:25],
        "recent_trade_rows": recent_trade_rows[:25],
        "trade_rows_source": "current_run" if trade_rows else ("latest_activity" if recent_trade_rows else "none"),
        "nav": nav_payload,
        "reconciliation": reconciliation_payload,
    }


def _idea_book_payload(store: LocalResearchStore, current_symbols: set[str], now: datetime) -> dict[str, Any]:
    signals = store.read_table("signal_artifacts")
    recommendations = store.read_table("portfolio_recommendations")
    snapshots = store.read_table("company_snapshots")
    predictions = store.read_table("predictions")

    if not signals.empty and "as_of" in signals.columns:
        signals = signals.copy()
        signals["as_of_ts"] = signals["as_of"].map(_ts)
    if not predictions.empty and "as_of" in predictions.columns:
        predictions = predictions.copy()
        predictions["as_of_ts"] = predictions["as_of"].map(_ts)

    if not recommendations.empty:
        recommendations = recommendations.copy()
        recommendations["as_of_ts"] = recommendations["as_of"].map(_ts)
        recommendations = recommendations.sort_values("as_of_ts").drop_duplicates(subset=["ticker"], keep="last")
    company_frame = pd.DataFrame()
    if not signals.empty:
        company_frame = signals.sort_values("as_of_ts").drop_duplicates(subset=["ticker"], keep="last").copy()
    if not snapshots.empty:
        snapshots = snapshots.copy()
        snapshots["as_of_ts"] = snapshots["as_of"].map(_ts)
        exact_snapshots = snapshots.sort_values("as_of_ts").drop_duplicates(subset=["ticker"], keep="last").copy()
        latest_names = exact_snapshots[["ticker", "company_name", "market_cap", "revenue"]].copy()
    else:
        exact_snapshots = pd.DataFrame()
        latest_names = pd.DataFrame(columns=["ticker", "company_name", "market_cap", "revenue"])
    prediction_frame = pd.DataFrame()
    if not predictions.empty:
        prediction_frame = predictions.copy()
        if not exact_snapshots.empty:
            prediction_frame = prediction_frame.merge(
                exact_snapshots[["ticker", "as_of_ts"]].rename(columns={"as_of_ts": "snapshot_as_of_ts"}),
                on="ticker",
                how="inner",
            )
            prediction_frame = prediction_frame[prediction_frame["as_of_ts"] == prediction_frame["snapshot_as_of_ts"]].copy()
        prediction_frame = prediction_frame.sort_values("as_of_ts").drop_duplicates(
            subset=["ticker", "entity_id", "thesis_horizon"],
            keep="last",
        ).copy()

    timestamp_candidates: list[pd.Timestamp] = []
    if not company_frame.empty:
        timestamp_candidates.extend([item for item in company_frame["as_of_ts"].dropna().tolist() if not pd.isna(item)])
    if not prediction_frame.empty:
        timestamp_candidates.extend([item for item in prediction_frame["as_of_ts"].dropna().tolist() if not pd.isna(item)])

    if not timestamp_candidates:
        return {
            "as_of": None,
            "oldest_as_of": None,
            "freshness_days": None,
            "oldest_freshness_days": None,
            "mixed_vintage": False,
            "notice": "",
            "ideas": [],
            "company_ideas": [],
            "program_ideas": [],
            "longs": [],
            "shorts": [],
            "company_idea_count": 0,
            "program_idea_count": 0,
            "catalyst_calendar": [],
        }

    latest_as_of = max(timestamp_candidates)
    oldest_as_of = min(timestamp_candidates)
    mixed_vintage = latest_as_of.date() != oldest_as_of.date()

    ideas: list[dict[str, Any]] = []
    frame = company_frame.copy()
    if not frame.empty and not recommendations.empty:
        frame = frame.merge(
            recommendations[
                [
                    "ticker",
                    "as_of",
                    "stance",
                    "target_weight",
                    "max_weight",
                    "scenario",
                    "risk_flags",
                ]
            ],
            on=["ticker"],
            how="left",
            suffixes=("", "_recommendation"),
        )
    if not frame.empty and not exact_snapshots.empty:
        frame = frame.merge(
            exact_snapshots[["ticker", "as_of", "company_name", "market_cap", "revenue"]],
            on=["ticker"],
            how="left",
            suffixes=("", "_snapshot"),
        )
    if not frame.empty:
        frame = frame.merge(
            latest_names.rename(
                columns={
                    "company_name": "company_name_fallback",
                    "market_cap": "market_cap_fallback",
                    "revenue": "revenue_fallback",
                }
            ),
            on="ticker",
            how="left",
        )

        for row in frame.itertuples(index=False):
            rationale = _parse_list(getattr(row, "rationale", []))
            evidence = _parse_list(getattr(row, "supporting_evidence", []))
            risk_flags = _parse_list(getattr(row, "risk_flags", []))
            company_name = getattr(row, "company_name", None) or getattr(row, "company_name_fallback", None) or row.ticker
            expected_return = _safe_float(getattr(row, "expected_return", 0.0), 0.0)
            stance = str(getattr(row, "stance", "") or "")
            if stance == "short":
                direction = "short"
            elif stance == "avoid":
                direction = "watch"
            else:
                direction = "short" if expected_return < 0 else "long"
            deployable, deployment_status, deployment_note = _deployment_status_for_company_idea(
                direction,
                stance,
                _safe_float(getattr(row, "target_weight", 0.0), 0.0),
            )
            primary_event_date = _iso(getattr(row, "primary_event_date", None))
            ideas.append(
                {
                    "idea_key": f"company:{row.ticker}",
                    "idea_level": "company",
                    "symbol": str(row.ticker),
                    "company_name": str(company_name),
                    "program_name": None,
                    "direction": direction,
                    "expected_return": expected_return,
                    "catalyst_success_prob": _safe_float(getattr(row, "catalyst_success_prob", 0.0), 0.0),
                    "confidence": _safe_float(getattr(row, "confidence", 0.0), 0.0),
                    "crowding_risk": _safe_float(getattr(row, "crowding_risk", 0.0), 0.0),
                    "financing_risk": _safe_float(getattr(row, "financing_risk", 0.0), 0.0),
                    "thesis_horizon": str(getattr(row, "thesis_horizon", "") or ""),
                    "stance": stance or ("short" if direction == "short" else "long"),
                    "target_weight": _safe_float(getattr(row, "target_weight", 0.0), 0.0),
                    "max_weight": _safe_float(getattr(row, "max_weight", 0.0), 0.0),
                    "scenario": str(getattr(row, "scenario", "") or ""),
                    "primary_event_type": None if _is_missing(getattr(row, "primary_event_type", None)) else str(getattr(row, "primary_event_type")),
                    "primary_event_bucket": None if _is_missing(getattr(row, "primary_event_bucket", None)) else str(getattr(row, "primary_event_bucket")),
                    "primary_event_status": None if _is_missing(getattr(row, "primary_event_status", None)) else str(getattr(row, "primary_event_status")),
                    "primary_event_date": primary_event_date,
                    "primary_event_exact": bool(getattr(row, "primary_event_exact", False)),
                    "company_state": None if _is_missing(getattr(row, "company_state", None)) else str(getattr(row, "company_state")),
                    "setup_type": None if _is_missing(getattr(row, "setup_type", None)) else str(getattr(row, "setup_type")),
                    "phase": None,
                    "modality": None,
                    "indication": None,
                    "internal_upside_pct": None if _is_missing(getattr(row, "internal_upside_pct", None)) else _safe_float(getattr(row, "internal_upside_pct")),
                    "floor_support_pct": None if _is_missing(getattr(row, "floor_support_pct", None)) else _safe_float(getattr(row, "floor_support_pct")),
                    "market_cap": _safe_float(getattr(row, "market_cap", getattr(row, "market_cap_fallback", 0.0)), 0.0),
                    "revenue": _safe_float(getattr(row, "revenue", getattr(row, "revenue_fallback", 0.0)), 0.0),
                    "rationale": rationale,
                    "rationale_preview": rationale[0] if rationale else "",
                    "risk_flags": risk_flags,
                    "evidence_items": _evidence_payload(evidence),
                    "evidence_count": len(evidence),
                    "evidence_sources": _evidence_sources(evidence),
                    "supporting_evidence_count": _safe_int(getattr(row, "supporting_evidence_count", len(evidence)), len(evidence)),
                    "program_prediction_count": _safe_int(getattr(row, "program_prediction_count", 0), 0),
                    "as_of": _iso(getattr(row, "as_of", None)),
                    "in_current_plan": str(row.ticker) in current_symbols,
                    "deployable": deployable,
                    "deployment_status": deployment_status,
                    "deployment_note": deployment_note,
                }
            )

    program_ideas: list[dict[str, Any]] = []
    if not prediction_frame.empty:
        if not prediction_frame.empty:
            prediction_frame["metadata_dict"] = prediction_frame["metadata"].map(_parse_dict)
            prediction_frame = prediction_frame.merge(
                latest_names.rename(
                    columns={
                        "company_name": "company_name_fallback",
                        "market_cap": "market_cap_fallback",
                        "revenue": "revenue_fallback",
                    }
                ),
                on="ticker",
                how="left",
            )
            for row in prediction_frame.itertuples(index=False):
                metadata = getattr(row, "metadata_dict", {}) or {}
                program_name = str(
                    _coalesce(
                        metadata.get("program_name"),
                        metadata.get("asset_name"),
                        getattr(row, "entity_id", None),
                    )
                    or ""
                )
                event_type = _coalesce(metadata.get("event_type"), metadata.get("company_primary_event_type"))
                event_status = _coalesce(metadata.get("event_status"), metadata.get("company_primary_event_status"))
                event_date = _iso(_coalesce(metadata.get("event_expected_date"), metadata.get("company_primary_event_expected_date")))
                indication = _coalesce(metadata.get("primary_indication"), metadata.get("indication"))
                phase = _coalesce(metadata.get("phase"), metadata.get("stage"))
                modality = _coalesce(metadata.get("modality"), metadata.get("mechanism"))
                expected_return = _safe_float(getattr(row, "expected_return", 0.0), 0.0)
                direction = "short" if expected_return < 0 else "long"
                deployable, deployment_status, deployment_note = _deployment_status_for_program_idea(direction)
                scenario_parts = [program_name]
                if event_type:
                    scenario_parts.append(str(event_type).replace("_", " "))
                if indication:
                    scenario_parts.append(f"in {indication}")
                scenario = " ".join(part for part in scenario_parts if part).strip()
                rationale_preview = scenario or f"{program_name} program signal"
                program_ideas.append(
                    {
                        "idea_key": f"program:{row.entity_id}:{row.thesis_horizon}:{row.as_of}",
                        "idea_level": "program",
                        "entity_id": str(row.entity_id),
                        "symbol": str(row.ticker),
                        "company_name": str(_coalesce(getattr(row, "company_name_fallback", None), row.ticker)),
                        "program_name": program_name,
                        "direction": direction,
                        "expected_return": expected_return,
                        "catalyst_success_prob": _safe_float(getattr(row, "catalyst_success_prob", 0.0), 0.0),
                        "confidence": _safe_float(getattr(row, "confidence", 0.0), 0.0),
                        "crowding_risk": _safe_float(getattr(row, "crowding_risk", 0.0), 0.0),
                        "financing_risk": _safe_float(getattr(row, "financing_risk", 0.0), 0.0),
                        "thesis_horizon": str(getattr(row, "thesis_horizon", "") or ""),
                        "stance": direction,
                        "target_weight": 0.0,
                        "max_weight": 0.0,
                        "scenario": scenario,
                        "primary_event_type": None if event_type is None else str(event_type),
                        "primary_event_bucket": None if _is_missing(metadata.get("event_bucket")) else str(metadata.get("event_bucket")),
                        "primary_event_status": None if event_status is None else str(event_status),
                        "primary_event_date": event_date,
                        "primary_event_exact": bool("exact" in str(event_status or "")),
                        "company_state": None if _is_missing(metadata.get("company_state")) else str(metadata.get("company_state")),
                        "setup_type": None,
                        "phase": None if phase is None else str(phase),
                        "modality": None if modality is None else str(modality),
                        "indication": None if indication is None else str(indication),
                        "internal_upside_pct": None,
                        "floor_support_pct": None,
                        "market_cap": _safe_float(getattr(row, "market_cap_fallback", 0.0), 0.0),
                        "revenue": _safe_float(getattr(row, "revenue_fallback", 0.0), 0.0),
                        "rationale": [],
                        "rationale_preview": rationale_preview,
                        "risk_flags": [],
                        "evidence_items": [],
                        "evidence_count": 0,
                        "evidence_sources": [],
                        "supporting_evidence_count": 0,
                        "program_prediction_count": 1,
                        "as_of": _iso(getattr(row, "as_of", None)),
                        "in_current_plan": str(row.ticker) in current_symbols,
                        "deployable": deployable,
                        "deployment_status": deployment_status,
                        "deployment_note": deployment_note,
                    }
                )

    tracked_symbols = {str(item["symbol"]) for item in [*ideas, *program_ideas]}
    snapshot_contexts = _load_snapshot_contexts(store, tracked_symbols)
    listing_contexts = _load_listing_contexts(store, tracked_symbols)
    for item in ideas:
        _enrich_idea_with_snapshot_context(item, snapshot_contexts, listing_contexts)
    for item in program_ideas:
        _enrich_idea_with_snapshot_context(item, snapshot_contexts, listing_contexts)

    visible_ideas = [idea for idea in ideas if idea.get("surfaceable", True)]
    visible_program_ideas = [idea for idea in program_ideas if idea.get("surfaceable", True)]

    company_longs = [idea for idea in visible_ideas if idea["direction"] == "long"]
    company_shorts = [idea for idea in visible_ideas if idea["direction"] == "short"]
    program_longs = [idea for idea in visible_program_ideas if idea["direction"] == "long"]
    program_shorts = [idea for idea in visible_program_ideas if idea["direction"] == "short"]
    longs = _blend_directional_ideas(company_longs, program_longs, direction="long", limit=12)
    shorts = _blend_directional_ideas(company_shorts, program_shorts, direction="short", limit=12)

    catalysts = [
        {
            "idea_key": idea["idea_key"],
            "symbol": idea["symbol"],
            "company_name": idea["company_name"],
            "program_name": idea.get("program_name"),
            "idea_level": idea.get("idea_level"),
            "event_type": idea["primary_event_type"],
            "event_status": idea["primary_event_status"],
            "event_date": idea["primary_event_date"],
            "event_exact": idea["primary_event_exact"],
            "scenario": idea["scenario"],
            "direction": idea["direction"],
            "confidence": idea["confidence"],
            "in_current_plan": idea.get("in_current_plan", False),
            "deployable": idea.get("deployable", True),
            "as_of": idea.get("as_of"),
        }
        for idea in [*visible_ideas, *visible_program_ideas]
        if idea["primary_event_date"] and not pd.isna(_ts(idea["primary_event_date"])) and _ts(idea["primary_event_date"]).date() >= now.date()
    ]
    catalysts.sort(
        key=lambda item: (
            _ts(item["event_date"]),
            -int(bool(item.get("program_name"))),
            -int(bool(item.get("event_exact"))),
            -_safe_float(item.get("confidence"), 0.0),
        )
    )
    deduped_catalysts: list[dict[str, Any]] = []
    seen_catalysts: set[tuple[str, str | None, str | None]] = set()
    for item in catalysts:
        key = (str(item["symbol"]), item.get("event_type"), item.get("event_date"))
        if key in seen_catalysts:
            continue
        seen_catalysts.add(key)
        deduped_catalysts.append(item)

    freshness_days = _days_old(latest_as_of, now)
    oldest_freshness_days = _days_old(oldest_as_of, now)
    return {
        "as_of": latest_as_of.isoformat() if not pd.isna(latest_as_of) else None,
        "oldest_as_of": oldest_as_of.isoformat() if not pd.isna(oldest_as_of) else None,
        "freshness_days": freshness_days,
        "oldest_freshness_days": oldest_freshness_days,
        "mixed_vintage": mixed_vintage,
        "notice": _build_research_notice(freshness_days, oldest_freshness_days, len(visible_ideas), len(visible_program_ideas), mixed_vintage),
        "ideas": [*visible_ideas, *visible_program_ideas],
        "company_ideas": visible_ideas,
        "program_ideas": visible_program_ideas,
        "longs": longs[:12],
        "shorts": shorts[:12],
        "company_idea_count": len(visible_ideas),
        "program_idea_count": len(visible_program_ideas),
        "catalyst_calendar": deduped_catalysts[:12],
    }


def _validation_payload(store: LocalResearchStore, now: datetime) -> dict[str, Any]:
    audit = load_best_validation_payload(store) or {}
    promotion_rows = store.read_table("model_promotions")
    promotion: dict[str, Any] | None = None
    if not promotion_rows.empty:
        promotion_rows = promotion_rows.copy()
        promotion_rows["created_at_ts"] = promotion_rows["created_at"].map(_ts)
        row = promotion_rows.sort_values("created_at_ts").iloc[-1]
        promotion = {
            "created_at": _iso(row.get("created_at")),
            "decision": str(row.get("decision") or "unknown"),
            "rationale": str(row.get("rationale") or ""),
            "blockers": _parse_list(row.get("blockers")),
        }

    return {
        "as_of": _iso(audit.get("generated_at") or audit.get("created_at") or audit.get("evaluated_at")),
        "freshness_days": _days_old(audit.get("generated_at") or audit.get("created_at") or audit.get("evaluated_at"), now),
        "rows": _safe_int(audit.get("num_rows", audit.get("rows")), 0),
        "windows": _safe_int(audit.get("num_windows", audit.get("windows")), 0),
        "rank_ic": _safe_float(audit.get("rank_ic"), 0.0),
        "strict_rank_ic": _safe_float(audit.get("strict_rank_ic"), 0.0),
        "hit_rate": _safe_float(audit.get("hit_rate"), 0.0),
        "cost_adjusted_top_bottom_spread": _safe_float(audit.get("cost_adjusted_top_bottom_spread"), 0.0),
        "pm_context_coverage": _safe_float(audit.get("pm_context_coverage"), 0.0),
        "exact_primary_event_rate": _safe_float(audit.get("exact_primary_event_rate"), 0.0),
        "synthetic_primary_event_rate": _safe_float(audit.get("synthetic_primary_event_rate"), 0.0),
        "rank_ic_ci_low": None if _is_missing(audit.get("rank_ic_ci_low")) else _safe_float(audit.get("rank_ic_ci_low")),
        "top_bottom_spread_ci_low": None if _is_missing(audit.get("top_bottom_spread_ci_low")) else _safe_float(audit.get("top_bottom_spread_ci_low")),
        "promotion": promotion,
    }


def _execution_payload(store: LocalResearchStore) -> dict[str, Any]:
    feedback = store.read_table("execution_feedback")
    scorecards = store.read_table("execution_profile_scorecards")
    execution = {
        "feedback_rows": int(len(feedback)),
        "scorecards": [],
    }
    if not scorecards.empty:
        for row in scorecards.itertuples(index=False):
            execution["scorecards"].append(
                {
                    "execution_profile": str(getattr(row, "execution_profile", "")),
                    "trades": _safe_int(getattr(row, "trades", 0), 0),
                    "avg_mark_to_market_net_return": None
                    if _is_missing(getattr(row, "avg_mark_to_market_net_return", None))
                    else _safe_float(getattr(row, "avg_mark_to_market_net_return")),
                    "avg_net_return_30d": None
                    if _is_missing(getattr(row, "avg_net_return_30d", None))
                    else _safe_float(getattr(row, "avg_net_return_30d")),
                    "avg_net_return_90d": None
                    if _is_missing(getattr(row, "avg_net_return_90d", None))
                    else _safe_float(getattr(row, "avg_net_return_90d")),
                    "net_hit_rate_30d": None
                    if _is_missing(getattr(row, "net_hit_rate_30d", None))
                    else _safe_float(getattr(row, "net_hit_rate_30d")),
                    "avg_estimated_round_trip_cost_bps": _safe_float(getattr(row, "avg_estimated_round_trip_cost_bps", 0.0), 0.0),
                }
            )
    return execution


def build_dashboard_payload(
    store: LocalResearchStore | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    store = store or LocalResearchStore()
    now = now or datetime.now(timezone.utc)

    current_plan = _current_plan_payload(store=store, now=now)
    current_symbols = set(current_plan["symbols"])
    idea_book = _idea_book_payload(store=store, current_symbols=current_symbols, now=now)
    validation = _validation_payload(store=store, now=now)
    execution = _execution_payload(store=store)

    summary = {
        "generated_at": now.astimezone(timezone.utc).isoformat(),
        "tracked_trade_rows": len(current_plan["trade_rows"]),
        "buy_orders": len(current_plan["buy_orders"]),
        "sell_orders": len(current_plan["sell_orders"]),
        "hold_orders": len(current_plan["hold_orders"]),
        "idea_count": len(idea_book["ideas"]),
        "company_idea_count": idea_book["company_idea_count"],
        "program_idea_count": idea_book["program_idea_count"],
        "long_idea_count": len(idea_book["longs"]),
        "short_idea_count": len(idea_book["shorts"]),
        "gross_target_weight": current_plan["gross_target_weight"],
        "gross_target_notional": current_plan["gross_target_notional"],
        "research_as_of": idea_book["as_of"],
        "research_freshness_days": idea_book["freshness_days"],
        "research_notice": idea_book["notice"],
        "trade_run_finished_at": None if current_plan["run"] is None else current_plan["run"]["finished_at"],
        "trade_run_freshness_days": current_plan["freshness_days"],
        "validation_rank_ic": validation["rank_ic"],
        "validation_exact_event_rate": validation["exact_primary_event_rate"],
    }

    return {
        "summary": summary,
        "current_plan": current_plan,
        "research_book": idea_book,
        "validation": validation,
        "execution": execution,
    }
