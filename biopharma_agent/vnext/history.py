from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import os
from typing import Any, Iterable

import pandas as pd

from .entities import CompanySnapshot
from .graph import build_company_snapshot
from .labels import CompositeHistoryProvider, EODHDHistoryProvider, PriceHistoryProvider, YFinanceHistoryProvider
from .replay import snapshot_from_dict
from .sources import CorporateCalendarClient, enrich_snapshot_with_external_data
from .storage import LocalResearchStore


ANCHOR_FORMS = {"10-Q", "10-Q/A", "10-K", "10-K/A", "20-F", "20-F/A", "6-K"}
QUARTERLY_FORMS = {"10-Q", "10-Q/A", "6-K"}
ANNUAL_FORMS = {"10-K", "10-K/A", "20-F", "20-F/A"}
OFFERING_FORMS = {"S-3", "424B5", "424B3", "F-3", "S-1"}

REVENUE_KEYS = [
    ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
    ("us-gaap", "SalesRevenueNet"),
    ("us-gaap", "Revenues"),
]
CASH_KEYS = [
    ("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),
    ("us-gaap", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
]
OPERATING_CASHFLOW_KEYS = [
    ("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
    ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
]
DEBT_KEYS = [
    ("us-gaap", "LongTermDebt"),
    ("us-gaap", "LongTermDebtNoncurrent"),
    ("us-gaap", "LongTermDebtAndCapitalLeaseObligations"),
    ("us-gaap", "LongTermDebtCurrent"),
    ("us-gaap", "LongTermDebtAndCapitalLeaseObligationsCurrent"),
]
SHARE_KEYS = [
    ("dei", "EntityCommonStockSharesOutstanding"),
    ("dei", "CommonStockSharesOutstanding"),
]


@dataclass(slots=True)
class HistoryBootstrapSummary:
    generated_snapshots: int
    tickers_processed: int
    tickers_with_history: int
    skipped_tickers: int
    distinct_anchor_dates: int
    earliest_as_of: str | None
    latest_as_of: str | None
    store_dir: str


class HistoricalSnapshotBootstrapper:
    def __init__(
        self,
        store: LocalResearchStore | None = None,
        history_provider: PriceHistoryProvider | None = None,
    ):
        self.store = store or LocalResearchStore()
        eodhd_key = os.environ.get("EODHD_API_KEY")
        self.history_provider = history_provider or CompositeHistoryProvider(
            [
                EODHDHistoryProvider(store=self.store),
                YFinanceHistoryProvider(store=self.store, allow_live=not bool(eodhd_key)),
            ]
        )
        self.calendar_client = CorporateCalendarClient()

    def materialize(
        self,
        ticker: str | None = None,
        ticker_limit: int = 0,
        max_anchors_per_ticker: int = 8,
        min_anchor_spacing_days: int = 45,
    ) -> HistoryBootstrapSummary:
        tickers = self._discover_tickers(ticker=ticker, ticker_limit=ticker_limit)
        generated_snapshots = 0
        tickers_with_history = 0
        skipped_tickers = 0
        anchor_dates: list[str] = []

        for symbol in tickers:
            latest_legacy = self.store.read_latest_raw_payload("legacy_prepare", f"{symbol}_")
            latest_sec = self.store.read_latest_raw_payload("sec_xbrl", f"{symbol}_")
            latest_snapshot_payload = self.store.read_latest_raw_payload("snapshots", f"{symbol}_")
            if not isinstance(latest_legacy, dict) or not isinstance(latest_sec, dict) or not isinstance(latest_snapshot_payload, dict):
                skipped_tickers += 1
                continue

            latest_snapshot = snapshot_from_dict(latest_snapshot_payload)
            anchor_rows = self._anchor_rows(
                latest_sec,
                latest_as_of=latest_snapshot.as_of,
                max_anchors=max_anchors_per_ticker,
                min_spacing_days=min_anchor_spacing_days,
            )
            if not anchor_rows:
                skipped_tickers += 1
                continue

            earliest_anchor = pd.Timestamp(anchor_rows[0]["filingDate"]) - pd.Timedelta(days=220)
            latest_anchor = pd.Timestamp(anchor_rows[-1]["filingDate"]) + pd.Timedelta(days=5)
            history = self.history_provider.load_history(
                symbol,
                start=earliest_anchor.date().isoformat(),
                end=latest_anchor.date().isoformat(),
            )
            if history.empty:
                skipped_tickers += 1
                continue

            history["close"] = pd.to_numeric(history["close"], errors="coerce")
            history = history.dropna(subset=["close"]).sort_index()
            if history.empty:
                skipped_tickers += 1
                continue

            ticker_generated = 0
            for anchor in anchor_rows:
                snapshot = self._reconstruct_snapshot(
                    ticker=symbol,
                    anchor=anchor,
                    base_raw=latest_legacy,
                    sec_payload=latest_sec,
                    latest_snapshot=latest_snapshot,
                    history=history["close"],
                )
                if snapshot is None:
                    continue
                self.store.write_snapshot(snapshot)
                self.store.write_raw_payload(
                    "historical_reconstruction",
                    f"{symbol}_{snapshot.as_of.replace(':', '-')}",
                    {
                        "ticker": symbol,
                        "as_of": snapshot.as_of,
                        "anchor_form": anchor.get("form"),
                        "anchor_filing_date": anchor.get("filingDate"),
                        "market_cap": snapshot.market_cap,
                        "revenue": snapshot.revenue,
                        "cash": snapshot.cash,
                        "debt": snapshot.debt,
                    },
                )
                generated_snapshots += 1
                ticker_generated += 1
                anchor_dates.append(snapshot.as_of.split("T")[0])

            if ticker_generated:
                tickers_with_history += 1
            else:
                skipped_tickers += 1

        unique_anchor_dates = sorted(set(anchor_dates))
        return HistoryBootstrapSummary(
            generated_snapshots=generated_snapshots,
            tickers_processed=len(tickers),
            tickers_with_history=tickers_with_history,
            skipped_tickers=skipped_tickers,
            distinct_anchor_dates=len(unique_anchor_dates),
            earliest_as_of=f"{unique_anchor_dates[0]}T00:00:00+00:00" if unique_anchor_dates else None,
            latest_as_of=f"{unique_anchor_dates[-1]}T00:00:00+00:00" if unique_anchor_dates else None,
            store_dir=str(self.store.base_dir),
        )

    def _discover_tickers(self, ticker: str | None, ticker_limit: int) -> list[str]:
        if ticker:
            return [ticker.upper()]
        tickers = sorted({path.name.split("_", 1)[0] for path in self.store.list_raw_payload_paths("sec_xbrl")})
        if ticker_limit > 0:
            return tickers[:ticker_limit]
        return tickers

    def _anchor_rows(
        self,
        sec_payload: dict[str, Any],
        latest_as_of: str,
        max_anchors: int,
        min_spacing_days: int,
    ) -> list[dict[str, Any]]:
        filings = [row for row in self._recent_filings(sec_payload) if str(row.get("form") or "") in ANCHOR_FORMS]
        if not filings:
            return []

        cutoff = pd.Timestamp(latest_as_of)
        if cutoff.tzinfo is not None:
            cutoff = cutoff.tz_convert(None)
        cutoff = cutoff.normalize()
        filings = sorted(filings, key=lambda row: (row.get("filingDate") or "", row.get("form") or ""))
        anchors: list[dict[str, Any]] = []
        for row in filings:
            filing_date = row.get("filingDate")
            if not filing_date:
                continue
            filing_ts = pd.Timestamp(filing_date).normalize()
            if filing_ts >= cutoff:
                continue
            if anchors:
                prior_ts = pd.Timestamp(anchors[-1]["filingDate"]).normalize()
                if (filing_ts - prior_ts).days < min_spacing_days:
                    continue
            anchors.append(row)
        if max_anchors > 0:
            anchors = anchors[-max_anchors:]
        return anchors

    def _reconstruct_snapshot(
        self,
        ticker: str,
        anchor: dict[str, Any],
        base_raw: dict[str, Any],
        sec_payload: dict[str, Any],
        latest_snapshot: CompanySnapshot,
        history: pd.Series,
    ) -> CompanySnapshot | None:
        filing_date = anchor.get("filingDate")
        if not filing_date:
            return None
        as_of_ts = pd.Timestamp(filing_date).tz_localize("UTC")
        as_of_naive = as_of_ts.tz_convert(None)

        price_now = self._price_at_or_before(history, as_of_naive)
        if price_now is None or price_now <= 0:
            return None

        momentum_3mo = self._return_over_window(history, as_of_naive, 90)
        trailing_6mo_return = self._return_over_window(history, as_of_naive, 180)
        volatility = self._volatility(history, as_of_naive, 63)

        sec_as_of = self._sec_payload_as_of(sec_payload, as_of_naive, ticker)
        cash_latest = float(sec_as_of["parsed"].get("cash_latest") or latest_snapshot.cash or 0.0)
        debt_latest = float(sec_as_of["parsed"].get("debt_latest") or latest_snapshot.debt or 0.0)
        revenue_ttm = float(sec_as_of["parsed"].get("revenue_ttm") or 0.0)
        shares_outstanding, shares_source = self._shares_outstanding(
            sec_payload=sec_payload,
            as_of=as_of_naive,
            current_market_cap=float(base_raw.get("finance", {}).get("marketCap") or latest_snapshot.market_cap or 0.0),
            current_price=float(base_raw.get("finance", {}).get("price_now") or 0.0),
            fallback_price=float(self._price_at_or_before(history, pd.Timestamp(latest_snapshot.as_of).tz_convert(None)) or price_now),
        )

        market_cap = max(price_now * shares_outstanding, 0.0) if shares_outstanding else float(latest_snapshot.market_cap or 0.0)
        enterprise_value = max(market_cap + debt_latest - cash_latest, 0.0)

        finance = dict(base_raw.get("finance", {}))
        finance.update(
            {
                "ticker": ticker,
                "marketCap": market_cap,
                "enterpriseValue": enterprise_value,
                "totalRevenue": revenue_ttm,
                "cash": cash_latest,
                "debt": debt_latest,
                "momentum_3mo": momentum_3mo,
                "trailing_6mo_return": trailing_6mo_return,
                "volatility": volatility,
                "price_now": price_now,
            }
        )
        raw = dict(base_raw)
        raw["finance"] = finance

        snapshot = build_company_snapshot(raw, as_of=as_of_ts.to_pydatetime())
        calendar_payload = self.calendar_client.fetch_company_calendar(ticker, sec_as_of)
        snapshot = enrich_snapshot_with_external_data(snapshot, sec_as_of, calendar_payload)
        snapshot.metadata.update(
            {
                "history_reconstruction": True,
                "history_source": "sec_price_reconstruction",
                "history_anchor_form": anchor.get("form"),
                "history_anchor_filing_date": filing_date,
                "history_shares_source": shares_source,
            }
        )
        return snapshot

    def _sec_payload_as_of(self, sec_payload: dict[str, Any], as_of: pd.Timestamp, ticker: str) -> dict[str, Any]:
        recent_filings = [row for row in self._recent_filings(sec_payload) if row.get("filingDate")]
        eligible = [row for row in recent_filings if pd.Timestamp(row["filingDate"]) <= as_of.normalize()]
        eligible = sorted(eligible, key=lambda row: (row.get("filingDate") or "", row.get("form") or ""), reverse=True)
        last_10q = next((row["filingDate"] for row in eligible if row.get("form") in QUARTERLY_FORMS), None)
        last_10k = next((row["filingDate"] for row in eligible if row.get("form") in ANNUAL_FORMS), None)
        return {
            "ticker": ticker,
            "cik": sec_payload.get("cik"),
            "source": "sec_xbrl_history",
            "status": "ok",
            "parsed": {
                "recent_filings": [self._filing_dict(sec_payload.get("cik"), row) for row in eligible[:10]],
                "last_10q_date": last_10q,
                "last_10k_date": last_10k,
                "recent_offering_forms": [self._filing_dict(sec_payload.get("cik"), row) for row in eligible if row.get("form") in OFFERING_FORMS][:5],
                "revenue_ttm": self._latest_duration_value(sec_payload, REVENUE_KEYS, as_of),
                "cash_latest": self._latest_instant_value(sec_payload, CASH_KEYS, as_of),
                "operating_cashflow": self._latest_duration_value(sec_payload, OPERATING_CASHFLOW_KEYS, as_of),
                "debt_latest": self._latest_instant_value(sec_payload, DEBT_KEYS, as_of),
                "shares_outstanding": self._latest_instant_value(sec_payload, SHARE_KEYS, as_of),
            },
        }

    @staticmethod
    def _filing_dict(cik: Any, row: dict[str, Any]) -> dict[str, Any]:
        accession = row.get("accessionNumber")
        primary_doc = row.get("primaryDocument")
        accession_clean = str(accession).replace("-", "") if accession else None
        url = None
        if cik and accession_clean and primary_doc:
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_clean}/{primary_doc}"
        return {
            "form": row.get("form"),
            "filing_date": row.get("filingDate"),
            "accession_number": accession,
            "primary_document": primary_doc,
            "url": url,
        }

    @staticmethod
    def _recent_filings(sec_payload: dict[str, Any]) -> list[dict[str, Any]]:
        recent = sec_payload.get("submissions", {}).get("filings", {}).get("recent", {})
        if not isinstance(recent, dict):
            return []
        keys = [key for key, value in recent.items() if isinstance(value, list)]
        if not keys:
            return []
        count = max(len(recent[key]) for key in keys)
        rows: list[dict[str, Any]] = []
        for idx in range(count):
            row = {key: recent[key][idx] if idx < len(recent[key]) else None for key in keys}
            rows.append(row)
        return rows

    @staticmethod
    def _fact_entries(sec_payload: dict[str, Any], key_space: Iterable[tuple[str, str]]) -> list[dict[str, Any]]:
        facts_root = sec_payload.get("company_facts", {}).get("facts", {})
        entries: list[dict[str, Any]] = []
        for namespace, key in key_space:
            item = facts_root.get(namespace, {}).get(key)
            if not isinstance(item, dict):
                continue
            for unit_entries in item.get("units", {}).values():
                if isinstance(unit_entries, list):
                    entries.extend(unit_entries)
        return entries

    @classmethod
    def _latest_instant_value(
        cls,
        sec_payload: dict[str, Any],
        key_space: Iterable[tuple[str, str]],
        as_of: pd.Timestamp,
    ) -> float | None:
        rows = []
        for entry in cls._fact_entries(sec_payload, key_space):
            filed = entry.get("filed")
            end = entry.get("end")
            val = entry.get("val")
            if filed is None or end is None or val is None:
                continue
            filed_ts = pd.Timestamp(filed)
            end_ts = pd.Timestamp(end)
            if filed_ts > as_of or end_ts > as_of:
                continue
            rows.append((end_ts, filed_ts, float(val)))
        if not rows:
            return None
        rows.sort(key=lambda item: (item[0], item[1]))
        return rows[-1][2]

    @classmethod
    def _latest_duration_value(
        cls,
        sec_payload: dict[str, Any],
        key_space: Iterable[tuple[str, str]],
        as_of: pd.Timestamp,
    ) -> float | None:
        rows: list[dict[str, Any]] = []
        for entry in cls._fact_entries(sec_payload, key_space):
            start = entry.get("start")
            end = entry.get("end")
            filed = entry.get("filed")
            val = entry.get("val")
            if start is None or end is None or filed is None or val is None:
                continue
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
            filed_ts = pd.Timestamp(filed)
            if filed_ts > as_of or end_ts > as_of:
                continue
            duration = max((end_ts - start_ts).days, 1)
            rows.append(
                {
                    "start": start_ts,
                    "end": end_ts,
                    "filed": filed_ts,
                    "duration": duration,
                    "val": float(val),
                }
            )
        if not rows:
            return None

        annuals = [row for row in rows if 300 <= row["duration"] <= 380]
        if annuals:
            annuals.sort(key=lambda row: (row["end"], row["filed"]))
            return annuals[-1]["val"]

        quarterlies = [row for row in rows if 60 <= row["duration"] <= 120]
        if quarterlies:
            deduped: dict[pd.Timestamp, dict[str, Any]] = {}
            for row in sorted(quarterlies, key=lambda item: (item["end"], item["filed"], item["duration"], item["val"])):
                current = deduped.get(row["end"])
                if current is None or row["duration"] < current["duration"] or abs(row["val"]) < abs(current["val"]):
                    deduped[row["end"]] = row
            latest = sorted(deduped.values(), key=lambda row: row["end"], reverse=True)[:4]
            if latest:
                total_days = sum(row["duration"] for row in latest)
                total_value = sum(row["val"] for row in latest)
                if total_days > 0:
                    return total_value * (365.0 / total_days)

        rows.sort(key=lambda row: (row["end"], row["filed"]))
        latest = rows[-1]
        return latest["val"] * (365.0 / latest["duration"])

    def _shares_outstanding(
        self,
        sec_payload: dict[str, Any],
        as_of: pd.Timestamp,
        current_market_cap: float,
        current_price: float,
        fallback_price: float,
    ) -> tuple[float, str]:
        shares = self._latest_instant_value(sec_payload, SHARE_KEYS, as_of)
        if shares and shares > 0:
            return float(shares), "sec_dei"
        price = current_price if current_price and current_price > 0 else fallback_price
        if current_market_cap > 0 and price and price > 0:
            return float(current_market_cap / price), "estimated_current"
        return 0.0, "unavailable"

    @staticmethod
    def _price_at_or_before(close: pd.Series, timestamp: pd.Timestamp) -> float | None:
        idx = close.index.values.searchsorted(timestamp.to_datetime64(), side="right") - 1
        if idx < 0:
            return None
        return float(close.iloc[idx])

    def _return_over_window(self, close: pd.Series, as_of: pd.Timestamp, days: int) -> float | None:
        current = self._price_at_or_before(close, as_of)
        prior = self._price_at_or_before(close, as_of - pd.Timedelta(days=days))
        if current is None or prior is None or prior <= 0:
            return None
        return (current / prior) - 1.0

    @staticmethod
    def _volatility(close: pd.Series, as_of: pd.Timestamp, lookback_days: int) -> float | None:
        window = close.loc[(close.index <= as_of) & (close.index >= as_of - pd.Timedelta(days=lookback_days))]
        returns = window.pct_change().dropna()
        if len(returns) < 10:
            return None
        return float(returns.std())
