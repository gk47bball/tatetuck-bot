from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .evaluation import WalkForwardEvaluator, _percentile_interval, _safe_mean, _spearman
from .execution_model import estimated_round_trip_cost_bps
from .labels import CompositeHistoryProvider, EODHDHistoryProvider, PointInTimeLabeler, PriceHistoryProvider, YFinanceHistoryProvider
from .settings import VNextSettings
from .storage import LocalResearchStore
from .taxonomy import (
    CLINICAL_EVENT_TYPES,
    event_pm_priority,
    event_timing_priority,
    event_type_bucket,
    is_exact_timing_event,
    normalized_event_type,
)

CATALYST_EVENT_FAMILIES = {
    "phase1_readout",
    "phase2_readout",
    "phase3_readout",
    "clinical_readout",
    "pdufa",
    "adcom",
    "label_expansion",
    "regulatory_update",
}
EXACT_SCHEDULE_SOURCES = {
    "company_calendar",
    "sec",
    "biopharmcatalyst",
    "eodhd_news",
    "eodhd_earnings",
}
EXACT_OUTCOME_SOURCES = {"sec", "eodhd_news"}
PRE_EVENT_VARIANTS = {
    "pre_20d_plus1d": 20,
    "pre_10d_plus1d": 10,
    "pre_5d_plus1d": 5,
}
POST_EVENT_VARIANT = "post_next_bar_plus5d"
SHORT_EVENT_VARIANT = "short_next_bar_plus5d"
DEFAULT_PRE_EVENT_VARIANT = "pre_10d_plus1d"
PRE_EVENT_MAX_LOOKBACK_DAYS = max(PRE_EVENT_VARIANTS.values())
POST_EVENT_WINDOW_DAYS = 6
SOURCE_PRIORITY = {
    "company_calendar": 5,
    "sec": 5,
    "biopharmcatalyst": 4,
    "eodhd_news": 3,
    "eodhd_earnings": 3,
    "estimated_calendar": 2,
    "internal_inference": 1,
}
FAMILY_PRIOR = {
    "pdufa": 0.72,
    "phase3_readout": 0.67,
    "phase2_readout": 0.56,
    "phase1_readout": 0.48,
    "clinical_readout": 0.52,
    "adcom": 0.61,
    "label_expansion": 0.63,
    "regulatory_update": 0.59,
}
ROLLING_WINDOWS = 6


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(min(max(value, lower), upper))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return float(default)
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _ts(value: Any) -> pd.Timestamp | pd.NaT:
    ts = pd.to_datetime(value, errors="coerce", utc=True, format="mixed")
    if pd.isna(ts):
        return pd.NaT
    return ts.tz_convert(None)


def _normalize_family(event_type: Any, title: Any = None, details: Any = None) -> str | None:
    normalized = normalized_event_type(
        None if event_type is None else str(event_type),
        None if title is None else str(title),
        None if details is None else str(details),
    )
    return None if normalized is None else str(normalized)


def _infer_event_family_from_text(title: str, details: str) -> str | None:
    text = f"{title} {details}".lower()
    if "pdufa" in text:
        return "pdufa"
    if "adcom" in text or "advisory committee" in text:
        return "adcom"
    if "label expansion" in text or "snda" in text or "supplemental nda" in text:
        return "label_expansion"
    if "phase 3" in text or "phase iii" in text or "pivotal" in text or "registrational" in text:
        return "phase3_readout"
    if "phase 2" in text or "phase ii" in text:
        return "phase2_readout"
    if "phase 1" in text or "phase i" in text:
        return "phase1_readout"
    if any(marker in text for marker in ("fda", "approval", "regulatory", "complete response letter", "crl")):
        return "regulatory_update"
    if any(marker in text for marker in ("topline", "top-line", "clinical", "readout", "data")):
        return "clinical_readout"
    return None


def _company_state(num_approved_products: Any, revenue: Any) -> str:
    approved = int(_safe_float(num_approved_products, 0.0))
    revenue_value = _safe_float(revenue, 0.0)
    if approved <= 0 and revenue_value < 75_000_000:
        return "pre_commercial"
    if approved > 0 and revenue_value < 500_000_000:
        return "commercial_launch"
    if approved > 0 and revenue_value >= 500_000_000:
        return "commercialized"
    if revenue_value >= 500_000_000:
        return "commercialized"
    if revenue_value >= 75_000_000:
        return "commercial_launch"
    return "pre_commercial"


def _floor_support_pct(market_cap: Any, cash: Any, debt: Any, revenue: Any) -> float:
    market_cap_value = max(_safe_float(market_cap, 0.0), 1.0)
    net_cash = max(_safe_float(cash, 0.0) - _safe_float(debt, 0.0), 0.0)
    return _clamp((net_cash + (0.35 * _safe_float(revenue, 0.0))) / market_cap_value, 0.0, 1.25)


def _financing_risk_proxy(market_cap: Any, cash: Any, debt: Any, revenue: Any) -> float:
    market_cap_value = max(_safe_float(market_cap, 0.0), 1.0)
    net_cash_pct = max(_safe_float(cash, 0.0) - _safe_float(debt, 0.0), 0.0) / market_cap_value
    revenue_support = (0.35 * _safe_float(revenue, 0.0)) / market_cap_value
    debt_pct = max(_safe_float(debt, 0.0), 0.0) / market_cap_value
    durability = _clamp(net_cash_pct + revenue_support, 0.0, 1.0)
    return _clamp((1.0 - durability) * 0.70 + min(debt_pct, 1.0) * 0.30, 0.0, 1.0)


def _regime_label(momentum: Any) -> str:
    momentum_value = _safe_float(momentum, 0.0)
    if momentum_value > 0.05:
        return "positive_momentum"
    if momentum_value < -0.05:
        return "negative_momentum"
    return "neutral_momentum"


def _source_priority(source: Any) -> int:
    return int(SOURCE_PRIORITY.get(str(source or ""), 0))


def _event_instance_id(row: pd.Series) -> str:
    ticker = str(row.get("ticker") or "").upper()
    event_id = str(row.get("event_id") or "").strip()
    family = str(row.get("event_family") or "unknown")
    event_date = row.get("expected_date_ts")
    date_label = pd.Timestamp(event_date).date().isoformat() if pd.notna(event_date) else "unknown-date"
    program_id = str(row.get("program_id") or "").strip()
    if program_id:
        return f"{ticker}|{program_id}|{family}|{date_label}"
    if event_id and event_id.lower() not in {"nan", "none"}:
        return f"{ticker}|{family}|{date_label}"
    return f"{ticker}|{family}|{date_label}"


def _horizon_bucket(days_until: float) -> str:
    days = int(max(days_until, 0.0))
    if days <= 30:
        return "07_30d"
    if days <= 60:
        return "31_60d"
    if days <= 90:
        return "61_90d"
    if days <= 120:
        return "91_120d"
    return "121d_plus"


def _knowledge_bucket(days_known: float) -> str:
    days = int(max(days_known, 0.0))
    if days <= 7:
        return "fresh_0_7d"
    if days <= 21:
        return "known_8_21d"
    if days <= 45:
        return "known_22_45d"
    return "known_46d_plus"


def _estimate_round_trip_cost(row: pd.Series) -> float:
    setup_type = (
        "hard_catalyst"
        if str(row.get("event_bucket") or "") in {"clinical", "regulatory"}
        else None
    )
    bps = estimated_round_trip_cost_bps(
        market_cap=_safe_float(row.get("market_cap"), 0.0),
        volatility=_safe_float(row.get("volatility"), 0.35),
        setup_type=setup_type,
    )
    return float(bps / 10_000.0)


def _borrow_cost_proxy(row: pd.Series) -> float:
    financing_risk = _safe_float(row.get("financing_risk_proxy"), 0.4)
    volatility = _safe_float(row.get("volatility"), 0.35)
    return float(0.002 + (0.008 * financing_risk) + (0.003 * min(max(volatility, 0.0), 1.0)))


def _window_summary(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "rows": 0.0,
            "rank_ic": 0.0,
            "hit_rate": 0.0,
            "top_bottom_spread": 0.0,
            "cost_adjusted_top_bottom_spread": 0.0,
            "beta_adjusted_return": 0.0,
        }
    usable = frame.dropna(subset=["score", "gross_return", "cost_adjusted_return"]).copy()
    if usable.empty:
        return {
            "rows": 0.0,
            "rank_ic": 0.0,
            "hit_rate": 0.0,
            "top_bottom_spread": 0.0,
            "cost_adjusted_top_bottom_spread": 0.0,
            "beta_adjusted_return": 0.0,
        }
    usable["score"] = pd.to_numeric(usable["score"], errors="coerce")
    usable["gross_return"] = pd.to_numeric(usable["gross_return"], errors="coerce")
    usable["cost_adjusted_return"] = pd.to_numeric(usable["cost_adjusted_return"], errors="coerce")
    usable = usable.dropna(subset=["score", "gross_return", "cost_adjusted_return"])
    if usable.empty:
        return {
            "rows": 0.0,
            "rank_ic": 0.0,
            "hit_rate": 0.0,
            "top_bottom_spread": 0.0,
            "cost_adjusted_top_bottom_spread": 0.0,
            "beta_adjusted_return": 0.0,
        }
    top = usable.nlargest(max(1, len(usable) // 5), "score")
    bottom = usable.nsmallest(max(1, len(usable) // 5), "score")
    threshold = float(usable["score"].median())
    pred_positive = usable["score"] >= threshold
    realized_positive = usable["cost_adjusted_return"] > 0.0
    return {
        "rows": float(len(usable)),
        "rank_ic": _spearman(usable["score"], usable["gross_return"]),
        "hit_rate": float((pred_positive == realized_positive).mean()),
        "top_bottom_spread": float(top["gross_return"].mean() - bottom["gross_return"].mean()) if len(usable) > 1 else 0.0,
        "cost_adjusted_top_bottom_spread": (
            float(top["cost_adjusted_return"].mean() - bottom["cost_adjusted_return"].mean())
            if len(usable) > 1
            else 0.0
        ),
        "beta_adjusted_return": float(top["cost_adjusted_return"].mean()) if not top.empty else 0.0,
    }


def _aggregate_window_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {
            "rows": 0.0,
            "windows": 0.0,
            "rank_ic": 0.0,
            "hit_rate": 0.0,
            "top_bottom_spread": 0.0,
            "cost_adjusted_top_bottom_spread": 0.0,
            "beta_adjusted_return": 0.0,
            "spread_ci_low": 0.0,
            "spread_ci_high": 0.0,
        }
    spread_values = [float(item.get("cost_adjusted_top_bottom_spread", 0.0) or 0.0) for item in rows]
    spread_ci_low, spread_ci_high = _percentile_interval(spread_values)
    return {
        "rows": float(sum(item.get("rows", 0.0) for item in rows)),
        "windows": float(len(rows)),
        "rank_ic": _safe_mean([float(item.get("rank_ic", 0.0) or 0.0) for item in rows]),
        "hit_rate": _safe_mean([float(item.get("hit_rate", 0.0) or 0.0) for item in rows]),
        "top_bottom_spread": _safe_mean([float(item.get("top_bottom_spread", 0.0) or 0.0) for item in rows]),
        "cost_adjusted_top_bottom_spread": _safe_mean(
            [float(item.get("cost_adjusted_top_bottom_spread", 0.0) or 0.0) for item in rows]
        ),
        "beta_adjusted_return": _safe_mean([float(item.get("beta_adjusted_return", 0.0) or 0.0) for item in rows]),
        "spread_ci_low": float(spread_ci_low),
        "spread_ci_high": float(spread_ci_high),
    }


@dataclass(slots=True)
class CatalystMaterializationSummary:
    event_master_rows: int
    outcome_master_rows: int
    trade_label_rows: int
    review_queue_rows: int


class BioPharmCatalystClient:
    def __init__(
        self,
        store: LocalResearchStore | None = None,
        settings: VNextSettings | None = None,
        session: requests.Session | None = None,
    ):
        self.store = store or LocalResearchStore()
        self.settings = settings or VNextSettings.from_env()
        self.session = session or requests.Session()
        self._local_rows: list[dict[str, Any]] | None = None

    def fetch_calendar_payload(self, ticker: str, as_of: datetime | None = None) -> dict[str, Any]:
        rows = self._load_local_rows()
        if not rows and self.settings.biopharmcatalyst_api_url:
            rows = self._fetch_remote_rows(ticker=ticker, as_of=as_of)
        normalized = [
            row
            for row in (self._normalize_row(item) for item in rows)
            if row is not None and row["ticker"] == ticker.upper()
        ]
        if as_of is not None:
            cutoff = pd.Timestamp(as_of).tz_convert(None) if pd.Timestamp(as_of).tzinfo is not None else pd.Timestamp(as_of)
            filtered: list[dict[str, Any]] = []
            for row in normalized:
                known_as_of = _ts(row.get("known_as_of"))
                if pd.notna(known_as_of) and known_as_of > cutoff:
                    continue
                filtered.append(row)
            normalized = filtered
        return {
            "ticker": ticker.upper(),
            "events": [
                {
                    "event_id": row["event_id"],
                    "event_type": row["event_type"],
                    "title": row["title"],
                    "expected_date": row["expected_date"],
                    "details": row.get("details"),
                    "url": row.get("url"),
                    "status": row.get("status", "exact_biopharmcatalyst"),
                    "source": "biopharmcatalyst",
                    "importance": row.get("importance", 0.60),
                    "crowdedness": row.get("crowdedness", 0.35),
                }
                for row in normalized
            ],
            "source": "biopharmcatalyst",
        }

    def _load_local_rows(self) -> list[dict[str, Any]]:
        if self._local_rows is not None:
            return self._local_rows
        path_value = self.settings.biopharmcatalyst_calendar_path
        if not path_value:
            self._local_rows = []
            return self._local_rows
        path = Path(path_value).expanduser()
        if not path.exists():
            self._local_rows = []
            return self._local_rows
        try:
            if path.suffix.lower() == ".json":
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict):
                    rows = payload.get("events") or payload.get("rows") or payload.get("data") or []
                elif isinstance(payload, list):
                    rows = payload
                else:
                    rows = []
            else:
                with open(path, "r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
        except (OSError, json.JSONDecodeError, csv.Error):
            rows = []
        self._local_rows = [row for row in rows if isinstance(row, dict)]
        return self._local_rows

    def _fetch_remote_rows(self, ticker: str, as_of: datetime | None = None) -> list[dict[str, Any]]:
        url = str(self.settings.biopharmcatalyst_api_url or "").strip()
        if not url:
            return []
        params = {"ticker": ticker.upper()}
        if as_of is not None:
            params["as_of"] = pd.Timestamp(as_of).date().isoformat()
        headers = {"User-Agent": self.settings.sec_user_agent}
        if self.settings.biopharmcatalyst_api_key:
            headers["Authorization"] = f"Bearer {self.settings.biopharmcatalyst_api_key}"
        try:
            response = self.session.get(url, params=params, headers=headers, timeout=20)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            return []
        if isinstance(payload, dict):
            rows = payload.get("events") or payload.get("rows") or payload.get("data") or []
        elif isinstance(payload, list):
            rows = payload
        else:
            rows = []
        return [row for row in rows if isinstance(row, dict)]

    def _normalize_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        ticker = str(
            row.get("ticker")
            or row.get("symbol")
            or row.get("Ticker")
            or row.get("Symbol")
            or ""
        ).upper()
        if not ticker:
            return None
        expected_date = (
            row.get("expected_date")
            or row.get("event_date")
            or row.get("date")
            or row.get("Date")
            or row.get("catalyst_date")
        )
        expected_ts = _ts(expected_date)
        if pd.isna(expected_ts):
            return None
        title = str(row.get("title") or row.get("event") or row.get("name") or row.get("description") or "").strip()
        details = str(row.get("details") or row.get("notes") or row.get("summary") or "").strip()
        event_type = _normalize_family(
            row.get("event_type") or row.get("family") or row.get("category") or row.get("type"),
            title,
            details,
        ) or _infer_event_family_from_text(title, details)
        if event_type not in CATALYST_EVENT_FAMILIES:
            return None
        event_id = str(row.get("event_id") or row.get("id") or f"bpc:{ticker}:{event_type}:{expected_ts.date().isoformat()}")
        return {
            "ticker": ticker,
            "event_id": event_id,
            "event_type": event_type,
            "title": title or f"{ticker} {event_type.replace('_', ' ')}",
            "expected_date": expected_ts.isoformat(),
            "details": details or None,
            "url": row.get("url") or row.get("source_url"),
            "status": str(row.get("status") or "exact_biopharmcatalyst"),
            "importance": _safe_float(row.get("importance"), 0.60),
            "crowdedness": _safe_float(row.get("crowdedness"), 0.35),
            "known_as_of": row.get("known_as_of") or row.get("published_at") or row.get("updated_at"),
        }


class CatalystEventStackBuilder:
    def __init__(
        self,
        store: LocalResearchStore | None = None,
        settings: VNextSettings | None = None,
        history_provider: PriceHistoryProvider | None = None,
    ):
        self.store = store or LocalResearchStore()
        self.settings = settings or VNextSettings.from_env()
        eodhd_key = self.settings.eodhd_api_key or os.environ.get("EODHD_API_KEY")
        self.history_provider = history_provider or CompositeHistoryProvider(
            [
                EODHDHistoryProvider(store=self.store),
                YFinanceHistoryProvider(store=self.store, allow_live=not bool(eodhd_key)),
            ]
        )
        self.labeler = PointInTimeLabeler(store=self.store, history_provider=self.history_provider)

    def materialize(self) -> CatalystMaterializationSummary:
        snapshots = self.store.read_table("company_snapshots")
        catalysts = self.store.read_table("catalysts")
        event_tape = self.store.read_table("event_tape")
        event_master = self.build_event_master(snapshots=snapshots, catalysts=catalysts)
        outcome_master = self.build_outcome_master(event_tape=event_tape)
        event_master = self.attach_outcomes(event_master=event_master, outcome_master=outcome_master)
        trade_labels = self.build_trade_labels(event_master=event_master)
        review_queue = self.build_review_queue(event_master=event_master, outcome_master=outcome_master)
        self.store.write_catalyst_event_master(event_master.to_dict(orient="records"))
        self.store.write_catalyst_outcome_master(outcome_master.to_dict(orient="records"))
        self.store.write_catalyst_trade_labels(trade_labels.to_dict(orient="records"))
        self.store.write_catalyst_review_queue(review_queue.to_dict(orient="records"))
        return CatalystMaterializationSummary(
            event_master_rows=len(event_master),
            outcome_master_rows=len(outcome_master),
            trade_label_rows=len(trade_labels),
            review_queue_rows=len(review_queue),
        )

    def build_event_master(self, snapshots: pd.DataFrame, catalysts: pd.DataFrame) -> pd.DataFrame:
        if catalysts.empty:
            return pd.DataFrame()
        events = catalysts.copy()
        if "as_of" not in events.columns or "expected_date" not in events.columns:
            return pd.DataFrame()
        events["as_of_ts"] = pd.to_datetime(events["as_of"], errors="coerce", utc=True, format="mixed").dt.tz_convert(None)
        events["expected_date_ts"] = pd.to_datetime(events["expected_date"], errors="coerce", utc=True, format="mixed").dt.tz_convert(None)
        events = events.dropna(subset=["as_of_ts", "expected_date_ts"]).copy()
        if events.empty:
            return pd.DataFrame()
        events["event_family"] = events.apply(
            lambda row: _normalize_family(row.get("event_type"), row.get("title"), row.get("details")),
            axis=1,
        )
        events["event_bucket"] = events.apply(
            lambda row: event_type_bucket(row.get("event_type"), row.get("title"), row.get("details")),
            axis=1,
        )
        events = events[events["event_bucket"].isin({"clinical", "regulatory"})].copy()
        events = events[events["event_family"].isin(CATALYST_EVENT_FAMILIES)].copy()
        if events.empty:
            return pd.DataFrame()
        if "timing_exact" not in events.columns:
            events["timing_exact"] = events.apply(
                lambda row: bool(is_exact_timing_event(row.get("status"), row.get("expected_date"), row.get("title"))),
                axis=1,
            )
        else:
            events["timing_exact"] = events["timing_exact"].fillna(False).astype(bool)
        if "timing_synthetic" not in events.columns:
            events["timing_synthetic"] = events["timing_exact"].map(lambda value: not bool(value))
        else:
            events["timing_synthetic"] = events["timing_synthetic"].fillna(False).astype(bool)
        for column in ("probability", "importance", "crowdedness", "provenance_confidence"):
            if column not in events.columns:
                events[column] = 0.0
            events[column] = pd.to_numeric(events[column], errors="coerce").fillna(0.0)
        if "source" not in events.columns:
            events["source"] = ""
        events["source"] = events["source"].fillna("").astype(str)
        events["source_priority"] = events["source"].map(_source_priority).fillna(0).astype(int)
        events["event_instance_id"] = events.apply(_event_instance_id, axis=1)
        events = events.sort_values(
            [
                "event_instance_id",
                "timing_exact",
                "source_priority",
                "provenance_confidence",
                "importance",
                "as_of_ts",
            ],
            ascending=[True, False, False, False, False, False],
        )

        snapshots_prepared = self._prepare_snapshots(snapshots)
        rows: list[dict[str, Any]] = []
        for event_instance_id, group in events.groupby("event_instance_id", sort=False):
            first_seen_ts = pd.Timestamp(group["as_of_ts"].min())
            last_seen_ts = pd.Timestamp(group["as_of_ts"].max())
            first_seen_group = group[group["as_of_ts"] == first_seen_ts].copy()
            if first_seen_group.empty:
                first_seen_group = group.copy()
            latest_group = group[group["as_of_ts"] == last_seen_ts].copy()
            if latest_group.empty:
                latest_group = group.copy()
            chosen = first_seen_group.sort_values(
                [
                    "timing_exact",
                    "source_priority",
                    "provenance_confidence",
                    "importance",
                    "as_of_ts",
                ],
                ascending=[False, False, False, False, True],
            ).iloc[0]
            latest = latest_group.sort_values(
                [
                    "timing_exact",
                    "source_priority",
                    "provenance_confidence",
                    "importance",
                    "as_of_ts",
                ],
                ascending=[False, False, False, False, False],
            ).iloc[0]
            context = self._context_for_event(
                snapshots=snapshots_prepared,
                ticker=str(chosen.get("ticker") or ""),
                context_ts=first_seen_ts,
            )
            latest_context = self._context_for_event(
                snapshots=snapshots_prepared,
                ticker=str(latest.get("ticker") or chosen.get("ticker") or ""),
                context_ts=last_seen_ts,
            )
            expected_date_ts = pd.Timestamp(chosen["expected_date_ts"])
            latest_expected_date_ts = pd.Timestamp(latest["expected_date_ts"])
            days_until = max(int((expected_date_ts.normalize() - first_seen_ts.normalize()).days), 0)
            latest_days_until = max(int((latest_expected_date_ts.normalize() - last_seen_ts.normalize()).days), 0)
            exact_sources = {
                str(item)
                for item in group.loc[group["timing_exact"].fillna(False), "source"].dropna().astype(str).tolist()
                if item
            }
            first_seen_exact_sources = {
                str(item)
                for item in first_seen_group.loc[first_seen_group["timing_exact"].fillna(False), "source"].dropna().astype(str).tolist()
                if item
            }
            corroborated = len(first_seen_exact_sources) >= 2 or (
                len(first_seen_exact_sources) >= 1
                and int(len(first_seen_group)) >= 2
                and str(chosen.get("source") or "") != "internal_inference"
            )
            latest_corroborated = len(exact_sources) >= 2 or (
                len(exact_sources) >= 1
                and int(len(group)) >= 2
                and str(latest.get("source") or "") != "internal_inference"
            )
            strict_schedule_eligible = bool(
                bool(chosen.get("timing_exact"))
                and str(chosen.get("source") or "") in EXACT_SCHEDULE_SOURCES
                and not bool(chosen.get("timing_synthetic"))
            )
            latest_strict_schedule_eligible = bool(
                bool(latest.get("timing_exact"))
                and str(latest.get("source") or "") in EXACT_SCHEDULE_SOURCES
                and not bool(latest.get("timing_synthetic"))
            )
            score_payload = self._score_event_row(
                family=str(chosen.get("event_family") or ""),
                company_state=context["company_state"],
                floor_support_pct=context["floor_support_pct"],
                financing_risk=context["financing_risk_proxy"],
                crowdedness=_safe_float(chosen.get("crowdedness"), 0.35),
                days_until=days_until,
                source=str(chosen.get("source") or ""),
                strict_schedule_eligible=strict_schedule_eligible,
                corroborated=corroborated,
            )
            latest_score_payload = self._score_event_row(
                family=str(latest.get("event_family") or ""),
                company_state=latest_context["company_state"],
                floor_support_pct=latest_context["floor_support_pct"],
                financing_risk=latest_context["financing_risk_proxy"],
                crowdedness=_safe_float(latest.get("crowdedness"), 0.35),
                days_until=latest_days_until,
                source=str(latest.get("source") or ""),
                strict_schedule_eligible=latest_strict_schedule_eligible,
                corroborated=latest_corroborated,
            )
            rows.append(
                {
                    "event_instance_id": event_instance_id,
                    "ticker": str(chosen.get("ticker") or ""),
                    "program_id": chosen.get("program_id"),
                    "event_id": chosen.get("event_id"),
                    "event_family": str(chosen.get("event_family") or ""),
                    "event_type": str(chosen.get("event_type") or ""),
                    "event_bucket": str(chosen.get("event_bucket") or ""),
                    "title": str(chosen.get("title") or ""),
                    "details": chosen.get("details"),
                    "expected_date": expected_date_ts.isoformat(),
                    "expected_date_ts": expected_date_ts,
                    "first_seen_as_of": first_seen_ts.isoformat(),
                    "first_seen_as_of_ts": first_seen_ts,
                    "last_seen_as_of": last_seen_ts.isoformat(),
                    "last_seen_as_of_ts": last_seen_ts,
                    "num_snapshot_mentions": int(len(group)),
                    "source": str(chosen.get("source") or ""),
                    "source_priority": int(chosen.get("source_priority", 0) or 0),
                    "source_precedence_label": self._source_precedence_label(str(chosen.get("source") or "")),
                    "status": str(chosen.get("status") or ""),
                    "timing_exact": bool(chosen.get("timing_exact")),
                    "timing_synthetic": bool(chosen.get("timing_synthetic")),
                    "strict_schedule_eligible": strict_schedule_eligible,
                    "corroborated": bool(corroborated),
                    "exact_sources_first_seen": sorted(first_seen_exact_sources),
                    "exact_sources_full_history": sorted(exact_sources),
                    "probability": _safe_float(chosen.get("probability"), 0.75),
                    "importance": _safe_float(chosen.get("importance"), 0.55),
                    "crowdedness": _safe_float(chosen.get("crowdedness"), 0.35),
                    "provenance_confidence": _safe_float(chosen.get("provenance_confidence"), 0.75),
                    "selection_rationale": chosen.get("selection_rationale"),
                    "days_until_first_seen": days_until,
                    "days_until_last_seen": max(int((expected_date_ts.normalize() - last_seen_ts.normalize()).days), 0),
                    "horizon_bucket": _horizon_bucket(days_until),
                    "event_pm_priority": int(
                        event_pm_priority(
                            chosen.get("event_type"),
                            chosen.get("status"),
                            chosen.get("expected_date"),
                            chosen.get("title"),
                        )
                    ),
                    "market_cap": context["market_cap"],
                    "revenue": context["revenue"],
                    "cash": context["cash"],
                    "debt": context["debt"],
                    "momentum_3mo": context["momentum_3mo"],
                    "volatility": context["volatility"],
                    "num_approved_products": context["num_approved_products"],
                    "company_state": context["company_state"],
                    "floor_support_pct": context["floor_support_pct"],
                    "financing_risk_proxy": context["financing_risk_proxy"],
                    "regime": context["regime"],
                    "pre_event_score": score_payload["pre_event_score"],
                    "confidence_proxy": score_payload["confidence_proxy"],
                    "success_probability_proxy": score_payload["success_probability_proxy"],
                    "deployable_pre_event": bool(
                        strict_schedule_eligible
                        and str(chosen.get("event_family") or "") in CATALYST_EVENT_FAMILIES
                        and 7 <= days_until <= 90
                    ),
                    "latest_event_id": latest.get("event_id"),
                    "latest_event_type": str(latest.get("event_type") or ""),
                    "latest_title": str(latest.get("title") or ""),
                    "latest_details": latest.get("details"),
                    "latest_expected_date": latest_expected_date_ts.isoformat(),
                    "latest_source": str(latest.get("source") or ""),
                    "latest_source_priority": int(latest.get("source_priority", 0) or 0),
                    "latest_source_precedence_label": self._source_precedence_label(str(latest.get("source") or "")),
                    "latest_status": str(latest.get("status") or ""),
                    "latest_timing_exact": bool(latest.get("timing_exact")),
                    "latest_timing_synthetic": bool(latest.get("timing_synthetic")),
                    "latest_strict_schedule_eligible": latest_strict_schedule_eligible,
                    "latest_corroborated": bool(latest_corroborated),
                    "latest_probability": _safe_float(latest.get("probability"), 0.75),
                    "latest_importance": _safe_float(latest.get("importance"), 0.55),
                    "latest_crowdedness": _safe_float(latest.get("crowdedness"), 0.35),
                    "latest_provenance_confidence": _safe_float(latest.get("provenance_confidence"), 0.75),
                    "latest_selection_rationale": latest.get("selection_rationale"),
                    "latest_days_until": latest_days_until,
                    "latest_horizon_bucket": _horizon_bucket(latest_days_until),
                    "latest_event_pm_priority": int(
                        event_pm_priority(
                            latest.get("event_type"),
                            latest.get("status"),
                            latest.get("expected_date"),
                            latest.get("title"),
                        )
                    ),
                    "latest_market_cap": latest_context["market_cap"],
                    "latest_revenue": latest_context["revenue"],
                    "latest_cash": latest_context["cash"],
                    "latest_debt": latest_context["debt"],
                    "latest_momentum_3mo": latest_context["momentum_3mo"],
                    "latest_volatility": latest_context["volatility"],
                    "latest_num_approved_products": latest_context["num_approved_products"],
                    "latest_company_state": latest_context["company_state"],
                    "latest_floor_support_pct": latest_context["floor_support_pct"],
                    "latest_financing_risk_proxy": latest_context["financing_risk_proxy"],
                    "latest_regime": latest_context["regime"],
                    "latest_pre_event_score": latest_score_payload["pre_event_score"],
                    "latest_confidence_proxy": latest_score_payload["confidence_proxy"],
                    "latest_success_probability_proxy": latest_score_payload["success_probability_proxy"],
                    "latest_deployable_pre_event": bool(
                        latest_strict_schedule_eligible
                        and str(latest.get("event_family") or "") in CATALYST_EVENT_FAMILIES
                        and 7 <= latest_days_until <= 90
                    ),
                }
            )
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        frame = self._apply_cluster_controls(frame)
        return frame.drop(columns=["expected_date_ts", "first_seen_as_of_ts", "last_seen_as_of_ts"], errors="ignore")

    def build_outcome_master(self, event_tape: pd.DataFrame) -> pd.DataFrame:
        snapshots = self.store.read_table("company_snapshots")
        if snapshots.empty or "ticker" not in snapshots.columns:
            tickers: list[str] = []
        else:
            tickers = sorted(snapshots["ticker"].dropna().astype(str).unique().tolist())
        rows: list[dict[str, Any]] = []
        for ticker in tickers:
            ticker_event_tape = event_tape[event_tape["ticker"] == ticker].copy() if not event_tape.empty else pd.DataFrame()
            frame = self.labeler._build_ticker_event_outcome_tape(ticker=ticker, ticker_event_tape=ticker_event_tape)
            if frame.empty:
                continue
            frame = frame.copy()
            frame["event_family"] = frame.apply(
                lambda row: _normalize_family(row.get("event_type"), row.get("title"), row.get("details")),
                axis=1,
            )
            inferred = frame.apply(
                lambda row: self.labeler._infer_outcome_from_text(
                    str(row.get("event_type") or ""),
                    str(row.get("title") or ""),
                    str(row.get("status") or ""),
                    str(row.get("details") or ""),
                ),
                axis=1,
            )
            frame["outcome_label"] = inferred.map(
                lambda item: np.nan if item is None else int(item.get("success", np.nan))
            )
            frame["outcome_source_label"] = inferred.map(
                lambda item: None if item is None else str(item.get("source") or "")
            )
            frame["parser_confidence"] = inferred.map(lambda item: 0.92 if item is not None else 0.0)
            frame = frame.sort_values(["ticker", "event_timestamp_ts", "parser_confidence"], ascending=[True, True, False])
            rows.extend(frame.drop(columns=["event_timestamp_ts"], errors="ignore").to_dict(orient="records"))
        outcome_master = pd.DataFrame(rows)
        if outcome_master.empty:
            return outcome_master
        outcome_master["event_timestamp_ts"] = pd.to_datetime(
            outcome_master["event_timestamp"],
            errors="coerce",
            utc=True,
            format="mixed",
        ).dt.tz_convert(None)
        outcome_master = outcome_master.dropna(subset=["event_timestamp_ts"]).copy()
        outcome_master = outcome_master.sort_values(["ticker", "event_timestamp_ts", "source"])
        outcome_master = outcome_master.drop_duplicates(
            subset=["ticker", "event_id", "event_timestamp", "source"],
            keep="first",
        )
        return outcome_master.drop(columns=["event_timestamp_ts"], errors="ignore")

    def attach_outcomes(self, event_master: pd.DataFrame, outcome_master: pd.DataFrame) -> pd.DataFrame:
        if event_master.empty:
            return event_master
        frame = event_master.copy()
        if outcome_master.empty:
            frame["matched_outcome_id"] = None
            frame["matched_outcome_timestamp"] = None
            frame["matched_outcome_label"] = np.nan
            frame["matched_outcome_source"] = None
            frame["matched_outcome_url"] = None
            frame["matched_outcome_window"] = None
            frame["exact_outcome_available"] = False
            return frame
        outcomes = outcome_master.copy()
        outcomes["event_timestamp_ts"] = pd.to_datetime(
            outcomes["event_timestamp"],
            errors="coerce",
            utc=True,
            format="mixed",
        ).dt.tz_convert(None)
        attached_rows: list[dict[str, Any]] = []
        for _, event in frame.iterrows():
            match = self._match_outcome_for_event(event=event, outcome_master=outcomes)
            payload = event.to_dict()
            if match is None:
                payload.update(
                    {
                        "matched_outcome_id": None,
                        "matched_outcome_timestamp": None,
                        "matched_outcome_label": np.nan,
                        "matched_outcome_source": None,
                        "matched_outcome_url": None,
                        "matched_outcome_window": None,
                        "exact_outcome_available": False,
                    }
                )
            else:
                payload.update(
                    {
                        "matched_outcome_id": match.get("event_id"),
                        "matched_outcome_timestamp": match.get("event_timestamp"),
                        "matched_outcome_label": pd.to_numeric(match.get("outcome_label"), errors="coerce"),
                        "matched_outcome_source": match.get("source"),
                        "matched_outcome_url": match.get("source_url"),
                        "matched_outcome_window": match.get("_matched_window"),
                        "exact_outcome_available": bool(pd.notna(pd.to_numeric(match.get("outcome_label"), errors="coerce"))),
                    }
                )
            attached_rows.append(payload)
        return pd.DataFrame(attached_rows)

    def build_trade_labels(self, event_master: pd.DataFrame) -> pd.DataFrame:
        if event_master.empty:
            return pd.DataFrame()
        frame = event_master.copy()
        frame["expected_date_ts"] = pd.to_datetime(frame["expected_date"], errors="coerce", utc=True, format="mixed").dt.tz_convert(None)
        frame["first_seen_as_of_ts"] = pd.to_datetime(frame["first_seen_as_of"], errors="coerce", utc=True, format="mixed").dt.tz_convert(None)
        frame["matched_outcome_ts"] = pd.to_datetime(
            frame.get("matched_outcome_timestamp"),
            errors="coerce",
            utc=True,
            format="mixed",
        ).dt.tz_convert(None)
        frame = frame.dropna(subset=["expected_date_ts", "first_seen_as_of_ts"])
        if frame.empty:
            return pd.DataFrame()
        if "cluster_active" in frame.columns:
            frame = frame[frame["cluster_active"].fillna(False).astype(bool)].copy()
        if frame.empty:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for ticker, group in frame.groupby("ticker", sort=False):
            start_ts = min(group["first_seen_as_of_ts"].min(), group["expected_date_ts"].min()) - timedelta(days=40)
            end_candidates = [group["expected_date_ts"].max() + timedelta(days=120)]
            if group["matched_outcome_ts"].notna().any():
                end_candidates.append(group["matched_outcome_ts"].max() + timedelta(days=30))
            history = self.history_provider.load_history(
                str(ticker),
                start=start_ts.date().isoformat(),
                end=max(end_candidates).date().isoformat(),
            )
            if history.empty or "close" not in history.columns:
                continue
            close = history["close"].astype(float)
            close.index = pd.to_datetime(close.index).astype("datetime64[ns]")
            if close.empty:
                continue
            for _, event in group.iterrows():
                rows.extend(self._pre_event_trade_rows(event=event, close=close))
                rows.extend(self._post_event_trade_rows(event=event, close=close))
                rows.extend(self._short_trade_rows(event=event, close=close))
        if not rows:
            return pd.DataFrame()
        trades = pd.DataFrame(rows)
        trades = self._apply_trade_cluster_controls(trades)
        trades = trades.sort_values(["entry_date", "ticker", "variant"]).reset_index(drop=True)
        return trades

    @staticmethod
    def _cluster_window_bounds(row: pd.Series) -> tuple[pd.Timestamp | pd.NaT, pd.Timestamp | pd.NaT]:
        expected_date_ts = _ts(row.get("expected_date"))
        first_seen_ts = _ts(row.get("first_seen_as_of"))
        matched_outcome_ts = _ts(row.get("matched_outcome_timestamp"))
        if pd.isna(expected_date_ts) or pd.isna(first_seen_ts):
            return pd.NaT, pd.NaT
        pre_event_start = expected_date_ts - timedelta(days=PRE_EVENT_MAX_LOOKBACK_DAYS)
        active_start = max(first_seen_ts, pre_event_start)
        active_end = (matched_outcome_ts if pd.notna(matched_outcome_ts) else expected_date_ts) + timedelta(days=POST_EVENT_WINDOW_DAYS)
        return active_start, active_end

    def _apply_cluster_controls(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        clustered = frame.copy()
        bounds = clustered.apply(self._cluster_window_bounds, axis=1)
        clustered["_cluster_start_ts"] = bounds.map(lambda item: item[0])
        clustered["_cluster_end_ts"] = bounds.map(lambda item: item[1])
        clustered["cluster_active"] = False
        clustered["cluster_blocked_by_event_instance_id"] = None
        clustered["cluster_reason"] = None
        clustered = clustered.sort_values(
            [
                "ticker",
                "_cluster_start_ts",
                "strict_schedule_eligible",
                "source_priority",
                "pre_event_score",
                "importance",
                "expected_date_ts",
            ],
            ascending=[True, True, False, False, False, False, True],
        )
        for _, group in clustered.groupby(["ticker"], sort=False):
            last_kept_idx = None
            last_kept_end = pd.NaT
            for idx, row in group.iterrows():
                start_ts = row.get("_cluster_start_ts")
                end_ts = row.get("_cluster_end_ts")
                if pd.isna(start_ts) or pd.isna(end_ts):
                    clustered.at[idx, "cluster_reason"] = "invalid_cluster_window"
                    continue
                if last_kept_idx is None or pd.isna(last_kept_end) or start_ts > last_kept_end:
                    clustered.at[idx, "cluster_active"] = True
                    last_kept_idx = idx
                    last_kept_end = end_ts
                    continue
                clustered.at[idx, "cluster_active"] = False
                clustered.at[idx, "cluster_blocked_by_event_instance_id"] = clustered.at[last_kept_idx, "event_instance_id"]
                clustered.at[idx, "cluster_reason"] = "overlapping_active_event_window"
        clustered["deployable_pre_event"] = (
            clustered["deployable_pre_event"].fillna(False).astype(bool)
            & clustered["cluster_active"].fillna(False).astype(bool)
        )
        return clustered

    @staticmethod
    def _apply_trade_cluster_controls(trades: pd.DataFrame) -> pd.DataFrame:
        if trades.empty:
            return trades
        frame = trades.copy()
        frame["entry_date_ts"] = pd.to_datetime(frame["entry_date"], errors="coerce", utc=True, format="mixed").dt.tz_convert(None)
        frame["exit_date_ts"] = pd.to_datetime(frame["exit_date"], errors="coerce", utc=True, format="mixed").dt.tz_convert(None)
        frame = frame.sort_values(
            ["sleeve", "variant", "ticker", "entry_date_ts", "exact_timing", "score", "event_instance_id"],
            ascending=[True, True, True, True, False, False, True],
        )
        keep_indices: list[int] = []
        for _, group in frame.groupby(["sleeve", "variant", "ticker"], sort=False):
            last_exit = pd.NaT
            for idx, row in group.iterrows():
                entry_ts = row.get("entry_date_ts")
                exit_ts = row.get("exit_date_ts")
                if pd.isna(entry_ts) or pd.isna(exit_ts):
                    continue
                if pd.notna(last_exit) and entry_ts <= last_exit:
                    continue
                keep_indices.append(idx)
                last_exit = exit_ts
        filtered = frame.loc[keep_indices].copy() if keep_indices else frame.iloc[0:0].copy()
        return filtered.drop(columns=["entry_date_ts", "exit_date_ts"], errors="ignore")

    def build_review_queue(self, event_master: pd.DataFrame, outcome_master: pd.DataFrame) -> pd.DataFrame:
        gaps = self.store.read_table("label_gap_audits")
        if gaps.empty:
            return pd.DataFrame()
        gaps = gaps.copy()
        gaps["evaluated_row"] = gaps["evaluated_row"].fillna(False).astype(bool) if "evaluated_row" in gaps.columns else False
        gaps = gaps[gaps["evaluated_row"]].copy()
        if "primary_event_source" in gaps.columns:
            gaps = gaps[gaps["primary_event_source"].fillna("").isin(EXACT_OUTCOME_SOURCES)].copy()
        if gaps.empty:
            return pd.DataFrame()
        if outcome_master.empty:
            return pd.DataFrame()
        outcomes = outcome_master.copy()
        outcomes["event_timestamp_ts"] = pd.to_datetime(
            outcomes["event_timestamp"],
            errors="coerce",
            utc=True,
            format="mixed",
        ).dt.tz_convert(None)
        review_rows: list[dict[str, Any]] = []
        for _, gap in gaps.iterrows():
            ticker = str(gap.get("ticker") or "")
            family = _normalize_family(gap.get("primary_event_type"))
            if family not in CATALYST_EVENT_FAMILIES:
                continue
            event_candidates = event_master[
                (event_master["ticker"] == ticker)
                & (event_master["event_family"] == family)
            ].copy()
            if event_candidates.empty:
                continue
            as_of_ts = _ts(gap.get("as_of"))
            if pd.isna(as_of_ts):
                continue
            event_candidates["first_seen_as_of_ts"] = pd.to_datetime(
                event_candidates["first_seen_as_of"],
                errors="coerce",
                utc=True,
                format="mixed",
            ).dt.tz_convert(None)
            event_candidates["distance"] = (
                (event_candidates["first_seen_as_of_ts"] - as_of_ts).abs().dt.total_seconds() / 86400.0
            )
            anchor = event_candidates.sort_values("distance").iloc[0]
            ambiguous = self._match_outcome_for_event(event=anchor, outcome_master=outcomes, require_labeled=False)
            if ambiguous is None or pd.notna(pd.to_numeric(ambiguous.get("outcome_label"), errors="coerce")):
                continue
            review_rows.append(
                {
                    "ticker": ticker,
                    "as_of": gap.get("as_of"),
                    "event_id": ambiguous.get("event_id"),
                    "source": ambiguous.get("source"),
                    "source_url": ambiguous.get("source_url"),
                    "event_timestamp": ambiguous.get("event_timestamp"),
                    "primary_event_type": gap.get("primary_event_type"),
                    "candidate_text_excerpt": str(ambiguous.get("details") or ambiguous.get("title") or "")[:600],
                    "suggested_reason": gap.get("gap_reason"),
                }
            )
        return pd.DataFrame(review_rows)

    def _prepare_snapshots(self, snapshots: pd.DataFrame) -> pd.DataFrame:
        if snapshots.empty:
            return pd.DataFrame()
        frame = snapshots.copy()
        frame["as_of_ts"] = pd.to_datetime(frame["as_of"], errors="coerce", utc=True, format="mixed").dt.tz_convert(None)
        frame = frame.dropna(subset=["as_of_ts"]).copy()
        if frame.empty:
            return frame
        for column in ("market_cap", "revenue", "cash", "debt", "momentum_3mo", "volatility", "num_approved_products"):
            if column not in frame.columns:
                frame[column] = 0.0
        frame["company_state"] = frame.apply(
            lambda row: _company_state(row.get("num_approved_products"), row.get("revenue")),
            axis=1,
        )
        frame["floor_support_pct"] = frame.apply(
            lambda row: _floor_support_pct(row.get("market_cap"), row.get("cash"), row.get("debt"), row.get("revenue")),
            axis=1,
        )
        frame["financing_risk_proxy"] = frame.apply(
            lambda row: _financing_risk_proxy(row.get("market_cap"), row.get("cash"), row.get("debt"), row.get("revenue")),
            axis=1,
        )
        frame["regime"] = frame["momentum_3mo"].map(_regime_label)
        return frame

    def _context_for_event(self, snapshots: pd.DataFrame, ticker: str, context_ts: pd.Timestamp) -> dict[str, Any]:
        subset = snapshots[snapshots["ticker"] == ticker].copy() if not snapshots.empty else pd.DataFrame()
        if subset.empty:
            return {
                "market_cap": 0.0,
                "revenue": 0.0,
                "cash": 0.0,
                "debt": 0.0,
                "momentum_3mo": 0.0,
                "volatility": 0.35,
                "num_approved_products": 0,
                "company_state": "pre_commercial",
                "floor_support_pct": 0.0,
                "financing_risk_proxy": 0.65,
                "regime": "neutral_momentum",
            }
        subset = subset[subset["as_of_ts"] <= context_ts].copy()
        if subset.empty:
            subset = snapshots[snapshots["ticker"] == ticker].copy()
        row = subset.sort_values("as_of_ts").iloc[-1]
        return {
            "market_cap": _safe_float(row.get("market_cap"), 0.0),
            "revenue": _safe_float(row.get("revenue"), 0.0),
            "cash": _safe_float(row.get("cash"), 0.0),
            "debt": _safe_float(row.get("debt"), 0.0),
            "momentum_3mo": _safe_float(row.get("momentum_3mo"), 0.0),
            "volatility": _safe_float(row.get("volatility"), 0.35),
            "num_approved_products": int(_safe_float(row.get("num_approved_products"), 0.0)),
            "company_state": str(row.get("company_state") or _company_state(row.get("num_approved_products"), row.get("revenue"))),
            "floor_support_pct": _safe_float(row.get("floor_support_pct"), 0.0),
            "financing_risk_proxy": _safe_float(row.get("financing_risk_proxy"), 0.65),
            "regime": str(row.get("regime") or _regime_label(row.get("momentum_3mo"))),
        }

    @staticmethod
    def _source_precedence_label(source: str) -> str:
        if source in {"company_calendar", "sec"}:
            return "company_exact"
        if source == "biopharmcatalyst":
            return "biopharmcatalyst_exact"
        if source in {"eodhd_news", "eodhd_earnings"}:
            return "news_exact"
        if source == "estimated_calendar":
            return "estimated_calendar"
        return "synthetic"

    @staticmethod
    def _score_event_row(
        *,
        family: str,
        company_state: str,
        floor_support_pct: float,
        financing_risk: float,
        crowdedness: float,
        days_until: int,
        source: str,
        strict_schedule_eligible: bool,
        corroborated: bool,
    ) -> dict[str, float]:
        family_prior = _safe_float(FAMILY_PRIOR.get(family, 0.50), 0.50)
        horizon_pref = 1.0 - min(abs(float(days_until) - 35.0) / 55.0, 1.0)
        source_bonus = {
            "company_calendar": 0.15,
            "sec": 0.14,
            "biopharmcatalyst": 0.12,
            "eodhd_news": 0.09,
            "eodhd_earnings": 0.05,
        }.get(source, 0.02)
        state_bonus = 0.0
        if company_state == "commercialized" and family in {"pdufa", "label_expansion", "regulatory_update"}:
            state_bonus = 0.05
        elif company_state == "pre_commercial" and family in CLINICAL_EVENT_TYPES:
            state_bonus = 0.04
        elif company_state == "commercial_launch":
            state_bonus = 0.03
        exact_bonus = 0.08 if strict_schedule_eligible else -0.06
        corroboration_bonus = 0.06 if corroborated else 0.0
        floor_bonus = 0.10 * min(max(floor_support_pct, 0.0), 0.50)
        risk_penalty = 0.18 * min(max(financing_risk, 0.0), 1.0)
        crowding_penalty = 0.08 * min(max(crowdedness, 0.0), 1.0)
        pre_event_score = _clamp(
            family_prior
            + source_bonus
            + exact_bonus
            + corroboration_bonus
            + (0.10 * horizon_pref)
            + state_bonus
            + floor_bonus
            - risk_penalty
            - crowding_penalty,
            -1.0,
            1.5,
        )
        confidence_proxy = _clamp(
            0.38
            + (0.25 * family_prior)
            + (0.12 if strict_schedule_eligible else -0.08)
            + (0.08 if corroborated else 0.0)
            + (0.06 * horizon_pref)
            - (0.12 * financing_risk)
            - (0.06 * crowdedness),
            0.0,
            1.0,
        )
        success_probability_proxy = _clamp(
            family_prior
            + (0.08 if strict_schedule_eligible else -0.05)
            + (0.05 if corroborated else 0.0)
            - (0.10 * financing_risk)
            - (0.05 * crowdedness),
            0.0,
            1.0,
        )
        return {
            "pre_event_score": float(pre_event_score),
            "confidence_proxy": float(confidence_proxy),
            "success_probability_proxy": float(success_probability_proxy),
        }

    def _match_outcome_for_event(
        self,
        event: pd.Series,
        outcome_master: pd.DataFrame,
        *,
        require_labeled: bool = True,
    ) -> pd.Series | None:
        if outcome_master.empty:
            return None
        ticker = str(event.get("ticker") or "")
        candidates = outcome_master[outcome_master["ticker"] == ticker].copy()
        if candidates.empty:
            return None
        if require_labeled:
            candidates = candidates[pd.to_numeric(candidates["outcome_label"], errors="coerce").notna()].copy()
            if candidates.empty:
                return None
        primary_event = pd.Series(
            {
                "event_type": event.get("event_family") or event.get("event_type"),
                "title": event.get("title"),
                "details": event.get("details"),
                "status": event.get("status"),
                "expected_date": event.get("expected_date"),
                "expected_date_ts": _ts(event.get("expected_date")),
                "strict_primary_corroborated": bool(event.get("corroborated", False)),
            }
        )
        as_of_ts = _ts(event.get("first_seen_as_of"))
        if pd.isna(as_of_ts):
            return None
        for window_name, lookback_days, lookahead_days in self.labeler._strict_matching_windows(primary_event):
            window_candidates = self.labeler._strict_window_candidates(
                candidates=candidates,
                primary_event=primary_event,
                as_of_ts=pd.Timestamp(as_of_ts),
                lookback_days=lookback_days,
                lookahead_days=lookahead_days,
            )
            if window_candidates.empty:
                continue
            candidate = window_candidates.iloc[0].copy()
            candidate["_matched_window"] = window_name
            return candidate
        return None

    def _pre_event_trade_rows(self, event: pd.Series, close: pd.Series) -> list[dict[str, Any]]:
        if not bool(event.get("strict_schedule_eligible", False)):
            return []
        rows: list[dict[str, Any]] = []
        expected_date_ts = _ts(event.get("expected_date"))
        first_seen_ts = _ts(event.get("first_seen_as_of"))
        if pd.isna(expected_date_ts) or pd.isna(first_seen_ts):
            return []
        for variant, lookback_days in PRE_EVENT_VARIANTS.items():
            entry_anchor_ts = expected_date_ts - timedelta(days=lookback_days)
            if first_seen_ts > entry_anchor_ts:
                continue
            entry_price = self.labeler._price_at_or_after(close, entry_anchor_ts.to_datetime64().astype("datetime64[ns]"))
            exit_price = self.labeler._price_at_or_after(
                close,
                (expected_date_ts + timedelta(days=1)).to_datetime64().astype("datetime64[ns]"),
            )
            if entry_price is None or exit_price is None or entry_price <= 0:
                continue
            gross_return = float((exit_price / entry_price) - 1.0)
            round_trip_cost = _estimate_round_trip_cost(event)
            rows.append(
                self._trade_row(
                    sleeve="pre_event_long",
                    variant=variant,
                    event=event,
                    entry_date=entry_anchor_ts,
                    exit_date=expected_date_ts + timedelta(days=1),
                    entry_price=entry_price,
                    exit_price=exit_price,
                    gross_return=gross_return,
                    cost_adjusted_return=gross_return - round_trip_cost,
                    score=_safe_float(event.get("pre_event_score"), 0.0),
                    exact_outcome=bool(event.get("exact_outcome_available", False)),
                    outcome_label=pd.to_numeric(event.get("matched_outcome_label"), errors="coerce"),
                    reaction_archetype="pre_event_setup",
                )
            )
        return rows

    def _post_event_trade_rows(self, event: pd.Series, close: pd.Series) -> list[dict[str, Any]]:
        outcome_label = pd.to_numeric(event.get("matched_outcome_label"), errors="coerce")
        outcome_ts = _ts(event.get("matched_outcome_timestamp"))
        if pd.isna(outcome_label) or pd.isna(outcome_ts) or int(outcome_label) != 1:
            return []
        entry_price = self.labeler._price_at_or_after(
            close,
            (outcome_ts + timedelta(days=1)).to_datetime64().astype("datetime64[ns]"),
        )
        exit_price = self.labeler._price_at_or_after(
            close,
            (outcome_ts + timedelta(days=6)).to_datetime64().astype("datetime64[ns]"),
        )
        anchor_price = self.labeler._price_at_or_before(close, outcome_ts.to_datetime64().astype("datetime64[ns]"))
        if entry_price is None or exit_price is None or anchor_price is None or entry_price <= 0 or anchor_price <= 0:
            return []
        initial_reaction = float((entry_price / anchor_price) - 1.0)
        reaction_penalty = 0.12 * min(max(initial_reaction, 0.0), 0.6)
        score = (
            _safe_float(FAMILY_PRIOR.get(str(event.get("event_family") or ""), 0.50), 0.50)
            + (0.25 * _safe_float(event.get("confidence_proxy"), 0.0))
            - reaction_penalty
        )
        gross_return = float((exit_price / entry_price) - 1.0)
        round_trip_cost = _estimate_round_trip_cost(event)
        return [
            self._trade_row(
                sleeve="post_event_reaction_long",
                variant=POST_EVENT_VARIANT,
                event=event,
                entry_date=outcome_ts + timedelta(days=1),
                exit_date=outcome_ts + timedelta(days=6),
                entry_price=entry_price,
                exit_price=exit_price,
                gross_return=gross_return,
                cost_adjusted_return=gross_return - round_trip_cost,
                score=score,
                exact_outcome=True,
                outcome_label=outcome_label,
                reaction_archetype="positive_continuation" if initial_reaction > 0 else "positive_fade",
            )
        ]

    def _short_trade_rows(self, event: pd.Series, close: pd.Series) -> list[dict[str, Any]]:
        outcome_label = pd.to_numeric(event.get("matched_outcome_label"), errors="coerce")
        outcome_ts = _ts(event.get("matched_outcome_timestamp"))
        if pd.isna(outcome_label) or pd.isna(outcome_ts) or int(outcome_label) != 0:
            return []
        entry_price = self.labeler._price_at_or_after(
            close,
            (outcome_ts + timedelta(days=1)).to_datetime64().astype("datetime64[ns]"),
        )
        exit_price = self.labeler._price_at_or_after(
            close,
            (outcome_ts + timedelta(days=6)).to_datetime64().astype("datetime64[ns]"),
        )
        if entry_price is None or exit_price is None or entry_price <= 0:
            return []
        gross_return = float(-((exit_price / entry_price) - 1.0))
        round_trip_cost = _estimate_round_trip_cost(event) + _borrow_cost_proxy(event)
        score = (
            _safe_float(FAMILY_PRIOR.get(str(event.get("event_family") or ""), 0.50), 0.50)
            + (0.20 * _safe_float(event.get("confidence_proxy"), 0.0))
            - (0.15 * _safe_float(event.get("financing_risk_proxy"), 0.0))
        )
        return [
            self._trade_row(
                sleeve="event_short_or_pairs",
                variant=SHORT_EVENT_VARIANT,
                event=event,
                entry_date=outcome_ts + timedelta(days=1),
                exit_date=outcome_ts + timedelta(days=6),
                entry_price=entry_price,
                exit_price=exit_price,
                gross_return=gross_return,
                cost_adjusted_return=gross_return - round_trip_cost,
                score=score,
                exact_outcome=True,
                outcome_label=outcome_label,
                reaction_archetype="negative_follow_through",
            )
        ]

    @staticmethod
    def _trade_row(
        *,
        sleeve: str,
        variant: str,
        event: pd.Series,
        entry_date: pd.Timestamp,
        exit_date: pd.Timestamp,
        entry_price: float,
        exit_price: float,
        gross_return: float,
        cost_adjusted_return: float,
        score: float,
        exact_outcome: bool,
        outcome_label: float | int | np.floating | None,
        reaction_archetype: str,
    ) -> dict[str, Any]:
        days_known = max(int((entry_date.normalize() - _ts(event.get("first_seen_as_of")).normalize()).days), 0)
        return {
            "sleeve": sleeve,
            "variant": variant,
            "event_instance_id": event.get("event_instance_id"),
            "ticker": event.get("ticker"),
            "event_family": event.get("event_family"),
            "event_type": event.get("event_type"),
            "event_bucket": event.get("event_bucket"),
            "source": event.get("source"),
            "expected_date": event.get("expected_date"),
            "entry_date": entry_date.isoformat(),
            "exit_date": exit_date.isoformat(),
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "gross_return": float(gross_return),
            "cost_adjusted_return": float(cost_adjusted_return),
            "score": float(score),
            "regime": event.get("regime"),
            "company_state": event.get("company_state"),
            "floor_support_pct": _safe_float(event.get("floor_support_pct"), 0.0),
            "financing_risk_proxy": _safe_float(event.get("financing_risk_proxy"), 0.0),
            "crowdedness": _safe_float(event.get("crowdedness"), 0.0),
            "horizon_bucket": event.get("horizon_bucket"),
            "knowledge_bucket": _knowledge_bucket(days_known),
            "days_known": days_known,
            "exact_timing": bool(event.get("timing_exact", False)),
            "exact_outcome": bool(exact_outcome),
            "matched_outcome_id": event.get("matched_outcome_id"),
            "matched_outcome_label": float(outcome_label) if outcome_label is not None and not pd.isna(outcome_label) else np.nan,
            "matched_outcome_source": event.get("matched_outcome_source"),
            "matched_outcome_window": event.get("matched_outcome_window"),
            "reaction_archetype": reaction_archetype,
            "confidence_proxy": _safe_float(event.get("confidence_proxy"), 0.0),
            "success_probability_proxy": _safe_float(event.get("success_probability_proxy"), 0.0),
            "market_cap": _safe_float(event.get("market_cap"), 0.0),
            "volatility": _safe_float(event.get("volatility"), 0.35),
        }


class CatalystEventEvaluator:
    def __init__(self, store: LocalResearchStore | None = None, settings: VNextSettings | None = None):
        self.store = store or LocalResearchStore()
        self.settings = settings or VNextSettings.from_env()

    def evaluate(
        self,
        trade_labels: pd.DataFrame | None = None,
        event_master: pd.DataFrame | None = None,
        source_job: str = "evaluate_vnext",
    ) -> dict[str, Any]:
        event_master = event_master.copy() if event_master is not None else self.store.read_table("catalyst_event_master").copy()
        trade_labels = trade_labels.copy() if trade_labels is not None else self.store.read_table("catalyst_trade_labels").copy()
        if event_master.empty or trade_labels.empty:
            return {
                "generated_at": _utc_now_iso(),
                "source_job": source_job,
                "event_master_rows": int(len(event_master)),
                "trade_label_rows": int(len(trade_labels)),
                "sleeve_scorecards": {},
                "family_scorecards": {},
                "regime_scorecards": {},
                "horizon_bucket_scorecards": {},
                "knowledge_bucket_scorecards": {},
                "rolling_sleeve_scorecards": {},
                "rolling_family_scorecards": {},
                "window_diagnostics": {},
                "gates": {
                    "pre_event_long": {"passed": False, "reason": "missing_trade_labels"},
                    "post_event_reaction_long": {"passed": False, "reason": "missing_trade_labels"},
                    "event_short_or_pairs": {"passed": False, "reason": "missing_trade_labels"},
                    "family_depth": {"passed": False, "reason": "missing_trade_labels"},
                    "overall_catalyst_bot": {"passed": False, "reason": "missing_trade_labels"},
                },
                "exact_timing_rate": 0.0,
                "exact_outcome_rate": 0.0,
                "review_queue_rows": int(len(self.store.read_table("catalyst_review_queue"))),
            }

        trades = trade_labels.copy()
        trades["entry_date_ts"] = pd.to_datetime(trades["entry_date"], errors="coerce", utc=True, format="mixed").dt.tz_convert(None)
        trades = trades.dropna(subset=["entry_date_ts"]).copy()
        sleeve_variants = {
            "pre_event_long": DEFAULT_PRE_EVENT_VARIANT,
            "post_event_reaction_long": POST_EVENT_VARIANT,
            "event_short_or_pairs": SHORT_EVENT_VARIANT,
        }
        default_frames = {
            sleeve: trades[(trades["sleeve"] == sleeve) & (trades["variant"] == variant)].copy()
            for sleeve, variant in sleeve_variants.items()
        }
        sleeve_exactness = {
            sleeve: {
                "exact_timing_rate": float(frame["exact_timing"].fillna(False).astype(bool).mean()) if not frame.empty else 0.0,
                "exact_outcome_rate": float(frame["exact_outcome"].fillna(False).astype(bool).mean()) if not frame.empty else 0.0,
                "rows": float(len(frame)),
            }
            for sleeve, frame in default_frames.items()
        }
        sleeve_windows = {
            sleeve: self._window_metrics(frame)
            for sleeve, frame in default_frames.items()
        }
        sleeve_scorecards = {
            sleeve: _aggregate_window_rows(rows)
            for sleeve, rows in sleeve_windows.items()
        }
        family_scorecards, rolling_family_scorecards = self._dimension_scorecards(
            default_frames["pre_event_long"],
            column="event_family",
        )
        regime_scorecards, _ = self._dimension_scorecards(default_frames["pre_event_long"], column="regime")
        horizon_bucket_scorecards, _ = self._dimension_scorecards(default_frames["pre_event_long"], column="horizon_bucket")
        knowledge_bucket_scorecards, _ = self._dimension_scorecards(default_frames["pre_event_long"], column="knowledge_bucket")
        rolling_sleeve_scorecards = {
            sleeve: _aggregate_window_rows(rows[-max(int(self.settings.rolling_validation_windows), 1):]) if rows else {}
            for sleeve, rows in sleeve_windows.items()
        }
        evaluated_frames = [frame for frame in default_frames.values() if not frame.empty]
        combined_exactness_frame = pd.concat(evaluated_frames, ignore_index=True) if evaluated_frames else pd.DataFrame()
        exact_timing_rate = (
            float(combined_exactness_frame["exact_timing"].fillna(False).astype(bool).mean())
            if not combined_exactness_frame.empty
            else 0.0
        )
        exact_outcome_rate = (
            float(combined_exactness_frame["exact_outcome"].fillna(False).astype(bool).mean())
            if not combined_exactness_frame.empty
            else 0.0
        )
        gates = self._build_gates(
            sleeve_windows=sleeve_windows,
            sleeve_scorecards=sleeve_scorecards,
            family_scorecards=family_scorecards,
            sleeve_exactness=sleeve_exactness,
        )
        return {
            "generated_at": _utc_now_iso(),
            "source_job": source_job,
            "event_master_rows": int(len(event_master)),
            "outcome_master_rows": int(len(self.store.read_table("catalyst_outcome_master"))),
            "trade_label_rows": int(len(trades)),
            "sleeve_scorecards": sleeve_scorecards,
            "family_scorecards": family_scorecards,
            "regime_scorecards": regime_scorecards,
            "horizon_bucket_scorecards": horizon_bucket_scorecards,
            "knowledge_bucket_scorecards": knowledge_bucket_scorecards,
            "rolling_sleeve_scorecards": rolling_sleeve_scorecards,
            "rolling_family_scorecards": rolling_family_scorecards,
            "sleeve_exactness": sleeve_exactness,
            "window_diagnostics": {
                sleeve: rows
                for sleeve, rows in sleeve_windows.items()
            },
            "gates": gates,
            "exact_timing_rate": exact_timing_rate,
            "exact_outcome_rate": exact_outcome_rate,
            "review_queue_rows": int(len(self.store.read_table("catalyst_review_queue"))),
        }

    def _window_metrics(self, frame: pd.DataFrame) -> list[dict[str, float]]:
        if frame.empty:
            return []
        unique_dates = sorted(pd.to_datetime(frame["entry_date_ts"]).dt.normalize().unique().tolist())
        rebalance_dates = WalkForwardEvaluator(store=self.store, settings=self.settings)._rebalance_dates(unique_dates)
        rows: list[dict[str, float]] = []
        for index, rebalance_date in enumerate(rebalance_dates):
            start_ts = pd.Timestamp(rebalance_date)
            end_ts = pd.Timestamp(rebalance_dates[index + 1]) if index + 1 < len(rebalance_dates) else None
            window_frame = frame[frame["entry_date_ts"] >= start_ts].copy()
            if end_ts is not None:
                window_frame = window_frame[window_frame["entry_date_ts"] < end_ts].copy()
            if window_frame.empty:
                continue
            metrics = _window_summary(window_frame)
            metrics["window_start"] = start_ts.isoformat()
            metrics["window_end"] = end_ts.isoformat() if end_ts is not None else None
            rows.append(metrics)
        return rows

    def _dimension_scorecards(
        self,
        frame: pd.DataFrame,
        *,
        column: str,
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        if frame.empty or column not in frame.columns:
            return {}, {}
        scorecards: dict[str, dict[str, float]] = {}
        rolling_scorecards: dict[str, dict[str, float]] = {}
        for value, group in frame.groupby(column, dropna=True):
            label = str(value)
            rows = self._window_metrics(group.copy())
            scorecards[label] = _aggregate_window_rows(rows)
            if rows:
                rolling_scorecards[label] = _aggregate_window_rows(rows[-max(int(self.settings.rolling_validation_windows), 1):])
        return scorecards, rolling_scorecards

    def _build_gates(
        self,
        *,
        sleeve_windows: dict[str, list[dict[str, float]]],
        sleeve_scorecards: dict[str, dict[str, float]],
        family_scorecards: dict[str, dict[str, float]],
        sleeve_exactness: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, object]]:
        def _non_negative_trailing(rows: list[dict[str, float]], trailing: int = 4) -> int:
            subset = rows[-max(int(trailing), 1):]
            return sum(float(item.get("cost_adjusted_top_bottom_spread", 0.0) or 0.0) >= 0.0 for item in subset)

        def _gate(passed: bool, reason: str, **extra: object) -> dict[str, object]:
            payload: dict[str, object] = {"passed": bool(passed)}
            if reason:
                payload["reason"] = reason
            payload.update(extra)
            return payload

        pre = sleeve_scorecards.get("pre_event_long", {})
        pre_windows = sleeve_windows.get("pre_event_long", [])
        pre_exactness = sleeve_exactness.get("pre_event_long", {})
        pre_pass = (
            float(pre.get("rows", 0.0) or 0.0) >= float(self.settings.catalyst_pre_event_min_rows)
            and float(pre.get("windows", 0.0) or 0.0) >= float(self.settings.catalyst_pre_event_min_windows)
            and float(pre.get("rank_ic", 0.0) or 0.0) >= float(self.settings.catalyst_pre_event_min_rank_ic)
            and float(pre.get("cost_adjusted_top_bottom_spread", 0.0) or 0.0) >= float(self.settings.catalyst_pre_event_min_net_spread)
            and float(pre.get("hit_rate", 0.0) or 0.0) >= float(self.settings.catalyst_pre_event_min_hit_rate)
            and float(pre.get("spread_ci_low", 0.0) or 0.0) > 0.0
            and _non_negative_trailing(pre_windows) >= int(self.settings.catalyst_pre_event_min_non_negative_trailing_windows)
            and float(pre_exactness.get("exact_timing_rate", 0.0) or 0.0) >= float(self.settings.catalyst_exact_timing_rate_min)
            and float(pre_exactness.get("exact_outcome_rate", 0.0) or 0.0) >= float(self.settings.catalyst_exact_outcome_rate_min)
        )
        post = sleeve_scorecards.get("post_event_reaction_long", {})
        post_windows = sleeve_windows.get("post_event_reaction_long", [])
        post_pass = (
            float(post.get("rows", 0.0) or 0.0) >= float(self.settings.catalyst_post_event_min_rows)
            and float(post.get("windows", 0.0) or 0.0) >= float(self.settings.catalyst_post_event_min_windows)
            and float(post.get("rank_ic", 0.0) or 0.0) >= float(self.settings.catalyst_post_event_min_rank_ic)
            and float(post.get("cost_adjusted_top_bottom_spread", 0.0) or 0.0) >= float(self.settings.catalyst_post_event_min_net_spread)
            and float(post.get("spread_ci_low", 0.0) or 0.0) > 0.0
        )
        short_metrics = sleeve_scorecards.get("event_short_or_pairs", {})
        short_pass = (
            float(short_metrics.get("rows", 0.0) or 0.0) >= float(self.settings.catalyst_short_min_rows)
            and float(short_metrics.get("windows", 0.0) or 0.0) >= float(self.settings.catalyst_short_min_windows)
            and float(short_metrics.get("rank_ic", 0.0) or 0.0) >= float(self.settings.catalyst_short_min_rank_ic)
            and float(short_metrics.get("cost_adjusted_top_bottom_spread", 0.0) or 0.0) >= float(self.settings.catalyst_short_min_net_spread)
            and float(short_metrics.get("spread_ci_low", 0.0) or 0.0) > 0.0
        )
        family_pass_count = sum(
            1
            for metrics in family_scorecards.values()
            if (
                float(metrics.get("rows", 0.0) or 0.0) >= float(self.settings.catalyst_family_min_rows)
                and float(metrics.get("windows", 0.0) or 0.0) >= float(self.settings.catalyst_family_min_windows)
                and float(metrics.get("cost_adjusted_top_bottom_spread", 0.0) or 0.0) > 0.0
                and float(metrics.get("rank_ic", 0.0) or 0.0) >= 0.0
            )
        )
        family_depth_pass = family_pass_count >= 2
        overall_pass = pre_pass and post_pass and family_depth_pass
        return {
            "pre_event_long": _gate(
                pre_pass,
                "pre_event_gate_clear" if pre_pass else "pre_event_long has not cleared its validation bar yet",
                **pre,
                exact_timing_rate=float(pre_exactness.get("exact_timing_rate", 0.0) or 0.0),
                exact_outcome_rate=float(pre_exactness.get("exact_outcome_rate", 0.0) or 0.0),
                non_negative_trailing_windows=_non_negative_trailing(pre_windows),
            ),
            "post_event_reaction_long": _gate(
                post_pass,
                "post_event_gate_clear" if post_pass else "post_event_reaction_long has not cleared its validation bar yet",
                **post,
                non_negative_trailing_windows=_non_negative_trailing(post_windows),
            ),
            "event_short_or_pairs": _gate(
                short_pass,
                "short_gate_clear" if short_pass else "event_short_or_pairs stays gated until short-side event alpha validates on its own",
                **short_metrics,
            ),
            "family_depth": _gate(
                family_depth_pass,
                "family_depth_clear" if family_depth_pass else "not enough catalyst families have independent positive evidence yet",
                qualifying_families=family_pass_count,
                required_families=2,
            ),
            "overall_catalyst_bot": _gate(
                overall_pass,
                "catalyst_bot_live" if overall_pass else "the catalyst bot remains gated until pre-event, post-event, and family depth all validate independently",
                exact_timing_rate=float(pre_exactness.get("exact_timing_rate", 0.0) or 0.0),
                exact_outcome_rate=float(pre_exactness.get("exact_outcome_rate", 0.0) or 0.0),
            ),
        }
