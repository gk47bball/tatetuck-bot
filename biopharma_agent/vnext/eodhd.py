from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
import re
from typing import Any

import pandas as pd
import requests

from .storage import LocalResearchStore
from .taxonomy import normalized_event_type


def _coerce_iso_timestamp(value: str | None, fallback_hour: int = 12, fallback_minute: int = 0) -> str | None:
    if not value:
        return None
    ts = pd.to_datetime(value, errors="coerce", utc=True, format="mixed")
    if pd.isna(ts):
        return None
    ts = ts.tz_convert(None)
    if ts.hour == 0 and ts.minute == 0 and ts.second == 0 and "T" not in str(value):
        ts = ts.replace(hour=fallback_hour, minute=fallback_minute, second=0)
    return ts.isoformat()


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_guidance_year(token: str) -> int | None:
    try:
        value = int(token)
    except (TypeError, ValueError):
        return None
    if value < 100:
        return 2000 + value
    if value < 2000:
        return None
    return value


def _period_end_timestamp(year: int, month: int) -> str | None:
    try:
        ts = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    except ValueError:
        return None
    return ts.isoformat()


def _future_guidance_timestamp(text: str, published_at: str | None) -> str | None:
    published_ts = pd.to_datetime(published_at, errors="coerce", utc=True, format="mixed")
    if not pd.isna(published_ts):
        published_ts = published_ts.tz_convert(None)

    quarter_map = {
        "first": 1,
        "1st": 1,
        "second": 2,
        "2nd": 2,
        "third": 3,
        "3rd": 3,
        "fourth": 4,
        "4th": 4,
    }
    half_map = {
        "first": 1,
        "1st": 1,
        "second": 2,
        "2nd": 2,
    }

    patterns: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"\bq([1-4])[\s'-]*(\d{2,4})\b", flags=re.IGNORECASE), "quarter"),
        (
            re.compile(
                r"\b(first|1st|second|2nd|third|3rd|fourth|4th)\s+quarter(?:\s+of)?\s+(\d{2,4})\b",
                flags=re.IGNORECASE,
            ),
            "quarter_named",
        ),
        (re.compile(r"\b([12])h[\s'-]*(\d{2,4})\b", flags=re.IGNORECASE), "half"),
        (
            re.compile(r"\b(first|1st|second|2nd)\s+half(?:\s+of)?\s+(\d{2,4})\b", flags=re.IGNORECASE),
            "half_named",
        ),
    ]

    for pattern, pattern_type in patterns:
        match = pattern.search(text)
        if not match:
            continue
        year = _normalize_guidance_year(match.group(2))
        if year is None:
            continue
        if pattern_type == "quarter":
            quarter = int(match.group(1))
            month = quarter * 3
        elif pattern_type == "quarter_named":
            quarter = quarter_map.get(match.group(1).lower())
            if quarter is None:
                continue
            month = quarter * 3
        elif pattern_type == "half":
            half = int(match.group(1))
            month = 6 if half == 1 else 12
        else:
            half = half_map.get(match.group(1).lower())
            if half is None:
                continue
            month = 6 if half == 1 else 12
        candidate = _period_end_timestamp(year, month)
        if candidate is None:
            continue
        candidate_ts = pd.to_datetime(candidate, errors="coerce", utc=True, format="mixed")
        if pd.isna(candidate_ts):
            continue
        candidate_ts = candidate_ts.tz_convert(None)
        if not pd.isna(published_ts) and candidate_ts.normalize() < published_ts.normalize():
            continue
        return candidate_ts.isoformat()
    return None


@dataclass(slots=True)
class UniverseSyncSummary:
    exchanges_requested: int
    active_rows: int
    delisted_rows: int
    total_rows: int


class EODHDClientBase:
    def __init__(
        self,
        store: LocalResearchStore | None = None,
        api_key: str | None = None,
        session: requests.Session | None = None,
    ):
        self.store = store or LocalResearchStore()
        self.api_key = os.environ.get("EODHD_API_KEY") if api_key is None else api_key
        self.session = session or requests.Session()

    @staticmethod
    def normalize_symbol(ticker: str, exchange_suffix: str = "US") -> str:
        return ticker if "." in ticker else f"{ticker.upper()}.{exchange_suffix}"

    def _fetch_json(self, namespace: str, cache_key: str, url: str, params: dict[str, Any]) -> Any:
        cached = self.store.read_latest_raw_payload(namespace, cache_key)
        if cached is not None:
            return cached
        if not self.api_key:
            return None
        params = dict(params)
        params.setdefault("api_token", self.api_key)
        params.setdefault("fmt", "json")
        try:
            response = self.session.get(url, params=params, timeout=20)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            return None
        self.store.write_raw_payload(namespace, cache_key, payload)
        return payload


class EODHDUniverseClient(EODHDClientBase):
    EXCHANGE_SYMBOLS_URL = "https://eodhd.com/api/exchange-symbol-list/{exchange}"

    def fetch_exchange_symbols(self, exchange: str, delisted: bool) -> list[dict[str, Any]]:
        payload = self._fetch_json(
            namespace="eodhd_exchange_symbols",
            cache_key=f"{exchange}_{'delisted' if delisted else 'active'}",
            url=self.EXCHANGE_SYMBOLS_URL.format(exchange=exchange),
            params={"delisted": 1 if delisted else 0},
        )
        if not isinstance(payload, list):
            return []
        return payload

    def sync_universe_membership(
        self,
        exchanges: tuple[str, ...] = ("NASDAQ", "NYSE", "AMEX"),
        as_of: datetime | None = None,
    ) -> UniverseSyncSummary:
        as_of = as_of or datetime.now(timezone.utc)
        as_of_iso = as_of.isoformat()
        rows: list[dict[str, Any]] = []
        active_rows = 0
        delisted_rows = 0
        seen: set[tuple[str, str, bool]] = set()

        for exchange in exchanges:
            for delisted in (False, True):
                payload_rows = self.fetch_exchange_symbols(exchange, delisted=delisted)
                for row in payload_rows:
                    normalized = self._normalize_symbol_row(row, exchange=exchange, as_of_iso=as_of_iso, delisted=delisted)
                    if normalized is None:
                        continue
                    dedupe = (str(normalized["ticker"]), str(normalized["exchange"]), bool(normalized["is_delisted"]))
                    if dedupe in seen:
                        continue
                    seen.add(dedupe)
                    rows.append(normalized)
                    if normalized["is_delisted"]:
                        delisted_rows += 1
                    else:
                        active_rows += 1

        if rows:
            existing = self.store.read_table("universe_membership")
            snapshot_rows: list[dict[str, Any]] = []
            if not existing.empty:
                snapshot_rows = existing[
                    existing["membership_source"].fillna("").isin(
                        {"snapshot_archive", "sec_price_reconstruction", "prepare_compatibility_layer"}
                    )
                ].to_dict(orient="records")
            self.store.replace_table("universe_membership", snapshot_rows + rows)

        return UniverseSyncSummary(
            exchanges_requested=len(exchanges),
            active_rows=active_rows,
            delisted_rows=delisted_rows,
            total_rows=active_rows + delisted_rows,
        )

    @staticmethod
    def _normalize_symbol_row(
        row: dict[str, Any],
        exchange: str,
        as_of_iso: str,
        delisted: bool,
    ) -> dict[str, Any] | None:
        code = row.get("Code") or row.get("code") or row.get("Symbol") or row.get("symbol")
        if not code:
            return None
        listed_at = (
            row.get("IPODate")
            or row.get("ipoDate")
            or row.get("ListingDate")
            or row.get("listingDate")
        )
        delisted_at = (
            row.get("DelistedAt")
            or row.get("delistedAt")
            or row.get("DelistedDate")
            or row.get("delistedDate")
        )
        name = row.get("Name") or row.get("name") or code
        instrument_type = row.get("Type") or row.get("type") or "Common Stock"
        is_delisted = bool(delisted or row.get("IsDelisted") or row.get("isDelisted"))
        ticker = str(code).split(".", 1)[0].upper()
        return {
            "ticker": ticker,
            "company_name": name,
            "as_of": as_of_iso,
            "exchange": row.get("Exchange") or row.get("exchange") or exchange,
            "security_type": instrument_type,
            "listing_symbol": str(code),
            "date_listed": listed_at,
            "date_delisted": delisted_at,
            "membership_source": "eodhd_exchange_symbols",
            "is_delisted": is_delisted,
            "has_exact_event": False,
        }


class EODHDEventTapeClient(EODHDClientBase):
    EARNINGS_URL = "https://eodhd.com/api/calendar/earnings"
    NEWS_URL = "https://eodhd.com/api/news"

    def fetch_event_payload(
        self,
        ticker: str,
        as_of: datetime,
        lookback_days: int = 7,
        lookahead_days: int = 120,
    ) -> dict[str, Any]:
        start = (as_of - timedelta(days=lookback_days)).date().isoformat()
        end = (as_of + timedelta(days=lookahead_days)).date().isoformat()
        normalized_symbol = self.normalize_symbol(ticker)

        earnings_payload = self._fetch_json(
            namespace="eodhd_earnings",
            cache_key=f"{ticker}_{start}_{end}",
            url=self.EARNINGS_URL,
            params={"from": start, "to": end},
        )
        news_payload = self._fetch_json(
            namespace="eodhd_news",
            cache_key=f"{ticker}_{start}_{as_of.date().isoformat()}",
            url=self.NEWS_URL,
            params={
                "s": normalized_symbol,
                "from": start,
                "to": as_of.date().isoformat(),
                "limit": 100,
            },
        )
        events = self._normalize_earnings_events(earnings_payload, normalized_symbol) + self._normalize_news_events(news_payload, normalized_symbol)
        events = self._merge_events(
            events,
            self._cached_news_fallback_events(
                ticker=ticker,
                as_of=as_of,
                lookahead_days=lookahead_days,
            ),
        )
        return {
            "ticker": ticker.upper(),
            "symbol": normalized_symbol,
            "as_of": as_of.isoformat(),
            "events": events,
            "source": "eodhd_event_tape",
        }

    @staticmethod
    def _merge_events(*event_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[tuple[str, str | None, str | None, str | None]] = set()
        for event_list in event_lists:
            for event in event_list:
                key = (
                    str(event.get("event_type") or ""),
                    event.get("expected_date"),
                    event.get("status"),
                    str(event.get("title") or ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                merged.append(event)
        return merged

    def _cached_news_fallback_events(
        self,
        ticker: str,
        as_of: datetime,
        lookahead_days: int = 120,
        max_history_days: int = 180,
        recent_exact_grace_days: int = 30,
    ) -> list[dict[str, Any]]:
        symbol = self.normalize_symbol(ticker)
        reference_end = (as_of + timedelta(days=lookahead_days)).date()
        exact_grace_cutoff = (as_of - timedelta(days=recent_exact_grace_days)).date()
        history_cutoff = (as_of - timedelta(days=max_history_days)).date()
        collected: list[dict[str, Any]] = []
        paths = sorted(
            self.store.list_raw_payload_paths("eodhd_news", f"{ticker.upper()}_"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in paths:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, list):
                continue
            for row in payload:
                published_ts = pd.to_datetime(row.get("date"), errors="coerce", utc=True, format="mixed")
                if not pd.isna(published_ts) and published_ts.tz_convert(None).date() < history_cutoff:
                    continue
                classified = self._classify_news_item(row, symbol)
                if classified is None:
                    continue
                expected_ts = pd.to_datetime(classified.get("expected_date"), errors="coerce", utc=True, format="mixed")
                if pd.isna(expected_ts):
                    continue
                expected_date = expected_ts.tz_convert(None).date()
                if expected_date > reference_end:
                    continue
                status = str(classified.get("status") or "")
                if status.startswith("exact") and expected_date < exact_grace_cutoff:
                    continue
                collected.append(classified)
        return self._merge_events(collected)

    def _normalize_earnings_events(self, payload: Any, symbol: str) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            return []
        earnings_rows = payload.get("earnings", [])
        events: list[dict[str, Any]] = []
        for row in earnings_rows:
            if str(row.get("code") or "").upper() != symbol.upper():
                continue
            before_after = str(row.get("before_after_market") or "").lower()
            hour = 8 if before_after == "beforemarket" else 16 if before_after == "aftermarket" else 12
            minute = 0 if before_after in {"beforemarket", "aftermarket"} else 0
            expected_at = _coerce_iso_timestamp(row.get("report_date"), fallback_hour=hour, fallback_minute=minute)
            if not expected_at:
                continue
            surprise = _safe_float(row.get("percent"))
            importance = 0.60 if surprise is not None else 0.55
            events.append(
                {
                    "event_id": f"{symbol}:earnings:{expected_at}",
                    "event_type": "earnings",
                    "title": f"{symbol} earnings report",
                    "expected_date": expected_at,
                    "status": "exact_company_calendar",
                    "importance": importance,
                    "crowdedness": 0.35,
                    "source": "eodhd_earnings",
                }
            )
        return events

    def _normalize_news_events(self, payload: Any, symbol: str) -> list[dict[str, Any]]:
        if not isinstance(payload, list):
            return []
        events: list[dict[str, Any]] = []
        for row in payload:
            classified = self._classify_news_item(row, symbol)
            if classified is not None:
                events.append(classified)
        return events

    @staticmethod
    def _classify_news_item(row: dict[str, Any], symbol: str) -> dict[str, Any] | None:
        raw_symbols = {
            str(item).upper()
            for item in (row.get("symbols") or [])
            if item not in (None, "")
        }
        # Vendor news queries can still return industry-adjacent headlines. If
        # the row declares explicit symbols and the requested one is absent,
        # discard it rather than polluting the catalyst tape with another
        # company's press release.
        if raw_symbols and symbol.upper() not in raw_symbols:
            return None

        title = str(row.get("title") or "")
        content = str(row.get("content") or "")
        tags = [str(tag).lower() for tag in row.get("tags", []) if tag]
        headline = f"{title} {content[:400]}".lower()
        if not any(
            verb in headline
            for verb in (
                "announce",
                "announces",
                "reported",
                "reports",
                "provide",
                "provides",
                "receive",
                "receives",
                "filed",
                "files",
                "granted",
                "vote",
                "votes",
                "voted",
                "recommend",
                "recommends",
                "acquire",
                "acquires",
                "withdraw",
                "withdraws",
                "discontinue",
                "divest",
                "expand",
                "repurchase",
            )
        ):
            return None

        if any(keyword in headline for keyword in ("acquisition", "acquire ", "acquires", "acquiring", "merger", "definitive agreement", "buyout", "m&a")):
            event_type = "strategic_transaction"
            importance = 0.84
        elif any(
            keyword in headline
            for keyword in (
                "withdraw",
                "withdrawal",
                "withdraws",
                "discontinue",
                "discontinuation",
                "divest",
                "divestiture",
                "strategic review",
                "out-license",
                "outlicense",
                "reprioritize",
                "deprioritize",
            )
        ):
            event_type = "portfolio_repositioning"
            importance = 0.76
        elif any(keyword in headline for keyword in ("label expansion", "expanded indication", "supplemental nda", "snda", "new indication")):
            event_type = "label_expansion"
            importance = 0.79
        elif any(keyword in headline for keyword in ("share repurchase", "buyback", "repurchase", "capital allocation", "special dividend", "deleveraging")):
            event_type = "capital_allocation"
            importance = 0.68
        elif any(keyword in headline for keyword in ("offering", "atm", "shelf", "registered direct", "private placement")):
            event_type = "recent_offering_filing"
            importance = 0.57
        elif any(keyword in headline for keyword in ("earnings", "financial results", "quarterly results", "annual results")) or any(
            tag in tags for tag in ("financial results", "quarterly results", "earnings release", "earnings results")
        ):
            event_type = "earnings"
            importance = 0.58
        else:
            inferred_event_type = normalized_event_type(None, title, content)
            importance_map = {
                "phase3_readout": 0.84,
                "phase2_readout": 0.81,
                "phase1_readout": 0.74,
                "clinical_readout": 0.78,
                "pdufa": 0.88,
                "adcom": 0.85,
                "regulatory_update": 0.80,
                "commercial_update": 0.62,
            }
            if inferred_event_type not in importance_map:
                return None
            event_type = str(inferred_event_type)
            importance = float(importance_map[event_type])

        event_at = _coerce_iso_timestamp(row.get("date"))
        if not event_at:
            return None
        guided_at = _future_guidance_timestamp(headline, event_at) if event_type in {
            "strategic_transaction",
            "portfolio_repositioning",
            "label_expansion",
            "capital_allocation",
        } else None
        expected_at = guided_at or event_at
        status = "guided_company_event" if guided_at else "exact_press_release"
        return {
            "event_id": f"{symbol}:news:{event_type}:{expected_at}",
            "event_type": event_type,
            "title": title[:180] or f"{symbol} press release",
            "expected_date": expected_at,
            "status": status,
            "importance": importance,
            "crowdedness": 0.35,
            "source": "eodhd_news",
            "url": row.get("link"),
            "details": content[:2000],
        }
