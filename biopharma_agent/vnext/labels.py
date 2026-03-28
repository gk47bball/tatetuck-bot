from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import os
from typing import Protocol

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from .storage import LocalResearchStore
from .taxonomy import event_timing_priority, event_type_bucket, event_type_priority

POSITIVE_OUTCOME_KEYWORDS = (
    "approved",
    "approval",
    "acceptance",
    "accepted",
    "positive topline",
    "positive top-line",
    "met the primary endpoint",
    "met primary endpoint",
    "met its primary endpoint",
    "met endpoint",
    "achieved primary endpoint",
    "priority review",
    "breakthrough therapy",
)

NEGATIVE_OUTCOME_KEYWORDS = (
    "complete response",
    "crl",
    "clinical hold",
    "terminated",
    "terminate",
    "discontinue",
    "discontinued",
    "failed",
    "failure",
    "missed the primary endpoint",
    "did not meet",
    "not approve",
    "rejected",
    "withdrawn",
)

MIXED_OUTCOME_KEYWORDS = (
    "business update",
    "corporate progress",
    "operational highlights",
    "quarterly results",
    "financial results",
)


class PriceHistoryProvider(Protocol):
    def load_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Return a DataFrame indexed by timestamp with a `close` column."""


@dataclass(slots=True)
class LabelSummary:
    snapshot_label_rows: int
    event_label_rows: int
    matured_return_90d_rows: int
    matured_event_rows: int
    num_tickers: int


class CompositeHistoryProvider:
    def __init__(self, providers: list[PriceHistoryProvider]):
        self.providers = providers

    def load_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        for provider in self.providers:
            history = provider.load_history(ticker, start, end)
            if not history.empty:
                return history
        return pd.DataFrame(columns=["close"])


class EODHDHistoryProvider:
    BASE_URL = "https://eodhd.com/api/eod/{symbol}"

    def __init__(
        self,
        store: LocalResearchStore | None = None,
        api_key: str | None = None,
        exchange_suffix: str = "US",
        session: requests.Session | None = None,
    ):
        self.store = store or LocalResearchStore()
        self.api_key = api_key or os.environ.get("EODHD_API_KEY")
        self.exchange_suffix = exchange_suffix
        self.session = session or requests.Session()

    def load_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        cache_key = f"{ticker}_{start}_{end}"
        cached = self.store.read_latest_raw_payload("market_prices_eodhd", cache_key)
        if cached is not None:
            return self._frame_from_payload(cached)
        if not self.api_key:
            return pd.DataFrame(columns=["close"])

        symbol = self._normalize_symbol(ticker)
        try:
            response = self.session.get(
                self.BASE_URL.format(symbol=symbol),
                params={
                    "api_token": self.api_key,
                    "fmt": "json",
                    "period": "d",
                    "order": "a",
                    "from": start,
                    "to": end,
                },
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            return pd.DataFrame(columns=["close"])

        if not isinstance(payload, list):
            return pd.DataFrame(columns=["close"])

        normalized_payload = [
            {
                "date": item.get("date"),
                "close": float(item["adjusted_close"] if item.get("adjusted_close") is not None else item.get("close")),
            }
            for item in payload
            if item.get("date") and (item.get("adjusted_close") is not None or item.get("close") is not None)
        ]
        self.store.write_raw_payload("market_prices_eodhd", cache_key, normalized_payload)
        return self._frame_from_payload(normalized_payload)

    def _normalize_symbol(self, ticker: str) -> str:
        if "." in ticker:
            return ticker
        return f"{ticker}.{self.exchange_suffix}"

    @staticmethod
    def _frame_from_payload(payload: list[dict[str, object]]) -> pd.DataFrame:
        if not payload:
            return pd.DataFrame(columns=["close"])
        frame = pd.DataFrame(payload)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.dropna(subset=["date", "close"]).sort_values("date")
        if frame.empty:
            return pd.DataFrame(columns=["close"])
        frame = frame.set_index("date")
        return frame[["close"]]


class YFinanceHistoryProvider:
    def __init__(self, store: LocalResearchStore | None = None, allow_live: bool = True):
        self.store = store or LocalResearchStore()
        self.allow_live = allow_live

    def load_history(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        cache_key = f"{ticker}_{start}_{end}"
        cached = self.store.read_latest_raw_payload("market_prices", cache_key)
        if cached:
            frame = pd.DataFrame(cached)
            if frame.empty:
                return pd.DataFrame(columns=["close"])
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame = frame.dropna(subset=["date"]).sort_values("date")
            frame = frame.set_index("date")
            return frame[["close"]]
        if not self.allow_live:
            return pd.DataFrame(columns=["close"])

        try:
            history = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
        except Exception:
            history = pd.DataFrame()
        if history.empty or "Close" not in history.columns:
            self.store.write_raw_payload("market_prices", cache_key, [])
            return pd.DataFrame(columns=["close"])

        close = history["Close"].astype(float)
        index = pd.to_datetime(close.index, utc=True).tz_convert(None).astype("datetime64[ns]")
        frame = pd.DataFrame({"close": close.to_numpy(dtype=float)}, index=index)
        frame.index.name = "date"
        payload = [
            {"date": pd.Timestamp(ts).isoformat(), "close": float(value)}
            for ts, value in zip(frame.index.tolist(), frame["close"].tolist())
        ]
        self.store.write_raw_payload("market_prices", cache_key, payload)
        return frame


class PointInTimeLabeler:
    def __init__(self, store: LocalResearchStore | None = None, history_provider: PriceHistoryProvider | None = None):
        self.store = store or LocalResearchStore()
        eodhd_key = os.environ.get("EODHD_API_KEY")
        self.history_provider = history_provider or CompositeHistoryProvider(
            [
                EODHDHistoryProvider(store=self.store),
                YFinanceHistoryProvider(store=self.store, allow_live=not bool(eodhd_key)),
            ]
        )

    def materialize_labels(self, snapshots: pd.DataFrame | None = None, catalysts: pd.DataFrame | None = None) -> LabelSummary:
        snapshot_labels, event_labels = self.build_label_frames(
            snapshots=snapshots,
            catalysts=catalysts,
            event_tape=self.store.read_table("event_tape"),
        )
        self.store.write_labels(snapshot_labels.to_dict(orient="records"))
        self.store.write_event_labels(event_labels.to_dict(orient="records"))
        matured_90d = int(snapshot_labels["target_return_90d"].notna().sum()) if not snapshot_labels.empty else 0
        matured_events = int(event_labels["target_event_return_10d"].notna().sum()) if not event_labels.empty else 0
        return LabelSummary(
            snapshot_label_rows=len(snapshot_labels),
            event_label_rows=len(event_labels),
            matured_return_90d_rows=matured_90d,
            matured_event_rows=matured_events,
            num_tickers=int(snapshot_labels["ticker"].nunique()) if not snapshot_labels.empty else 0,
        )

    def build_label_frames(
        self,
        snapshots: pd.DataFrame | None = None,
        catalysts: pd.DataFrame | None = None,
        event_tape: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        snapshots = snapshots.copy() if snapshots is not None else self.store.read_table("company_snapshots").copy()
        catalysts = catalysts.copy() if catalysts is not None else self.store.read_table("catalysts").copy()
        event_tape = event_tape.copy() if event_tape is not None else self.store.read_table("event_tape").copy()
        if snapshots.empty:
            return pd.DataFrame(), pd.DataFrame()

        snapshots["as_of_ts"] = pd.to_datetime(
            snapshots["as_of"],
            errors="coerce",
            utc=True,
            format="mixed",
        ).dt.tz_convert(None)
        snapshots = snapshots.dropna(subset=["as_of_ts"]).sort_values(["ticker", "as_of_ts"])
        if not catalysts.empty and "expected_date" in catalysts.columns:
            catalysts["expected_date_ts"] = pd.to_datetime(
                catalysts["expected_date"],
                errors="coerce",
                utc=True,
                format="mixed",
            ).dt.tz_convert(None)
        else:
            catalysts = pd.DataFrame()
        if not event_tape.empty and "event_timestamp" in event_tape.columns:
            event_tape["event_timestamp_ts"] = pd.to_datetime(
                event_tape["event_timestamp"],
                errors="coerce",
                utc=True,
                format="mixed",
            ).dt.tz_convert(None)
        else:
            event_tape = pd.DataFrame()

        snapshot_rows: list[dict[str, object]] = []
        event_rows: list[dict[str, object]] = []
        for ticker in sorted(snapshots["ticker"].dropna().unique().tolist()):
            ticker_snaps = snapshots[snapshots["ticker"] == ticker].copy()
            ticker_catalysts = catalysts[catalysts["ticker"] == ticker].copy() if not catalysts.empty else pd.DataFrame()
            ticker_event_tape = event_tape[event_tape["ticker"] == ticker].copy() if not event_tape.empty else pd.DataFrame()

            start_ts = ticker_snaps["as_of_ts"].min() - timedelta(days=14)
            end_ts = ticker_snaps["as_of_ts"].max() + timedelta(days=220)
            if not ticker_catalysts.empty and ticker_catalysts["expected_date_ts"].notna().any():
                end_ts = max(end_ts, ticker_catalysts["expected_date_ts"].max() + timedelta(days=14))

            history = self.history_provider.load_history(
                ticker,
                start=start_ts.date().isoformat(),
                end=end_ts.date().isoformat(),
            )
            if history.empty:
                continue
            close = history["close"].astype(float)
            close.index = pd.to_datetime(close.index).astype("datetime64[ns]")

            for snapshot in ticker_snaps.itertuples(index=False):
                row = {
                    "ticker": ticker,
                    "as_of": snapshot.as_of,
                }
                base_ts = pd.Timestamp(snapshot.as_of_ts).to_datetime64().astype("datetime64[ns]")
                base_price = self._price_at_or_before(close, base_ts)
                if base_price is None or base_price <= 0:
                    continue

                for horizon, days in (("30d", 30), ("90d", 90), ("180d", 180)):
                    target_ts = base_ts + np.timedelta64(days, "D")
                    future_price = self._price_at_or_after(close, target_ts)
                    row[f"target_return_{horizon}"] = (
                        np.nan if future_price is None else (future_price / base_price) - 1.0
                    )

                primary_event = self._select_primary_event(
                    ticker_catalysts=ticker_catalysts,
                    as_of=snapshot.as_of,
                    as_of_ts=pd.Timestamp(snapshot.as_of_ts),
                )
                if primary_event is not None:
                    event_return = self._event_window_return(close, pd.Timestamp(primary_event["expected_date_ts"]))
                    reaction_success = int(pd.notna(event_return) and float(event_return) > 0.05)
                    outcome_label = self._exact_outcome_label(
                        ticker_event_tape=ticker_event_tape,
                        primary_event=primary_event,
                        as_of_ts=pd.Timestamp(snapshot.as_of_ts),
                    )
                    row["target_event_return_10d"] = event_return
                    row["target_event_success_market"] = reaction_success
                    row["target_event_success_outcome"] = (
                        outcome_label["success"] if outcome_label is not None else np.nan
                    )
                    row["target_primary_event_days"] = int(primary_event["horizon_days"])
                    row["target_primary_event_type"] = primary_event["event_type"]
                    row["target_primary_event_bucket"] = event_type_bucket(primary_event["event_type"])
                    row["target_catalyst_success_source"] = (
                        outcome_label["source"] if outcome_label is not None else "event_price_reaction"
                    )
                    row["target_event_success"] = (
                        int(outcome_label["success"])
                        if outcome_label is not None
                        else reaction_success
                    )
                    event_rows.append(
                        {
                            "ticker": ticker,
                            "as_of": snapshot.as_of,
                            "event_id": primary_event["event_id"],
                            "event_type": primary_event["event_type"],
                            "event_bucket": event_type_bucket(primary_event["event_type"]),
                            "expected_date": primary_event["expected_date"],
                            "horizon_days": int(primary_event["horizon_days"]),
                            "target_event_return_10d": event_return,
                            "target_event_success_market": reaction_success,
                            "target_event_success_outcome": (
                                outcome_label["success"] if outcome_label is not None else np.nan
                            ),
                            "target_event_success": (
                                int(outcome_label["success"])
                                if outcome_label is not None
                                else reaction_success
                            ),
                            "target_event_success_source": (
                                outcome_label["source"] if outcome_label is not None else "event_price_reaction"
                            ),
                        }
                    )
                    catalyst_success = (
                        int(outcome_label["success"])
                        if outcome_label is not None
                        else reaction_success
                    )
                else:
                    row["target_event_return_10d"] = np.nan
                    row["target_event_success"] = np.nan
                    row["target_event_success_market"] = np.nan
                    row["target_event_success_outcome"] = np.nan
                    row["target_primary_event_days"] = np.nan
                    row["target_primary_event_type"] = None
                    row["target_primary_event_bucket"] = "none"
                    row["target_catalyst_success_source"] = "return_90d_fallback"
                    catalyst_success = int(
                        pd.notna(row.get("target_return_90d"))
                        and float(row["target_return_90d"]) > 0.08
                    )

                row["target_catalyst_success"] = catalyst_success
                snapshot_rows.append(row)

        return pd.DataFrame(snapshot_rows), pd.DataFrame(event_rows)

    @classmethod
    def _exact_outcome_label(
        cls,
        ticker_event_tape: pd.DataFrame,
        primary_event: pd.Series,
        as_of_ts: pd.Timestamp,
    ) -> dict[str, object] | None:
        if ticker_event_tape.empty:
            return None
        expected_date_ts = primary_event.get("expected_date_ts")
        if expected_date_ts is None or pd.isna(expected_date_ts):
            return None
        primary_bucket = event_type_bucket(primary_event.get("event_type"))
        if primary_bucket not in {"clinical", "regulatory"}:
            return None
        candidates = ticker_event_tape.dropna(subset=["event_timestamp_ts"]).copy()
        if candidates.empty:
            return None
        candidates = candidates[
            (candidates["event_timestamp_ts"] >= as_of_ts)
            & (candidates["event_timestamp_ts"] >= (expected_date_ts - timedelta(days=2)))
            & (candidates["event_timestamp_ts"] <= (expected_date_ts + timedelta(days=21)))
        ].copy()
        if candidates.empty:
            return None
        candidates["event_bucket"] = candidates["event_type"].map(event_type_bucket)
        candidates = candidates[candidates["event_bucket"] == primary_bucket].copy()
        if candidates.empty:
            return None
        candidates["type_match"] = (
            candidates["event_type"].astype(str) == str(primary_event.get("event_type") or "")
        ).astype(float)
        candidates["timing_priority"] = candidates.apply(
            lambda row: event_timing_priority(row.get("status"), row.get("event_timestamp"), row.get("title")),
            axis=1,
        )
        candidates["days_from_expected"] = (
            (candidates["event_timestamp_ts"] - expected_date_ts).abs().dt.total_seconds() / 86400.0
        )
        candidates = candidates.sort_values(
            ["type_match", "timing_priority", "days_from_expected"],
            ascending=[False, False, True],
        )
        for _, row in candidates.iterrows():
            inferred = cls._infer_outcome_from_text(
                event_type=str(row.get("event_type") or ""),
                title=str(row.get("title") or ""),
                status=str(row.get("status") or ""),
            )
            if inferred is not None:
                return inferred
        return None

    @staticmethod
    def _infer_outcome_from_text(
        event_type: str,
        title: str,
        status: str,
    ) -> dict[str, object] | None:
        bucket = event_type_bucket(event_type)
        if bucket not in {"clinical", "regulatory"}:
            return None
        text = f"{title} {status}".lower()
        if any(keyword in text for keyword in MIXED_OUTCOME_KEYWORDS):
            return None
        if any(keyword in text for keyword in NEGATIVE_OUTCOME_KEYWORDS):
            return {"success": 0, "source": "exact_event_outcome_negative"}
        if any(keyword in text for keyword in POSITIVE_OUTCOME_KEYWORDS):
            return {"success": 1, "source": "exact_event_outcome_positive"}
        return None

    @staticmethod
    def _select_primary_event(
        ticker_catalysts: pd.DataFrame,
        as_of: str,
        as_of_ts: pd.Timestamp,
    ) -> pd.Series | None:
        if ticker_catalysts.empty:
            return None
        exact = ticker_catalysts[ticker_catalysts["as_of"] == as_of].copy()
        if exact.empty:
            return None
        exact = exact.dropna(subset=["expected_date_ts"])
        if exact.empty:
            return None
        exact = exact[exact["expected_date_ts"] >= as_of_ts]
        if exact.empty:
            return None
        if "importance" not in exact.columns:
            exact["importance"] = 0.0
        if "crowdedness" not in exact.columns:
            exact["crowdedness"] = 0.50
        if "status" not in exact.columns:
            exact["status"] = None
        if "title" not in exact.columns:
            exact["title"] = None
        exact["event_priority"] = exact["event_type"].map(event_type_priority).fillna(0)
        exact["timing_priority"] = exact.apply(
            lambda row: event_timing_priority(row.get("status"), row.get("expected_date"), row.get("title")),
            axis=1,
        )
        exact = exact.sort_values(
            ["timing_priority", "event_priority", "importance", "expected_date_ts", "crowdedness"],
            ascending=[False, False, False, True, True],
        )
        return exact.iloc[0]

    @staticmethod
    def _price_at_or_after(close: pd.Series, timestamp: np.datetime64) -> float | None:
        idx = close.index.values.searchsorted(timestamp, side="left")
        if idx >= len(close):
            return None
        return float(close.iloc[idx])

    @staticmethod
    def _price_at_or_before(close: pd.Series, timestamp: np.datetime64) -> float | None:
        idx = close.index.values.searchsorted(timestamp, side="right") - 1
        if idx < 0:
            return None
        return float(close.iloc[idx])

    @classmethod
    def _event_window_return(cls, close: pd.Series, event_ts: pd.Timestamp) -> float:
        start_price = cls._price_at_or_before(
            close,
            (event_ts - timedelta(days=5)).to_datetime64().astype("datetime64[ns]"),
        )
        end_price = cls._price_at_or_after(
            close,
            (event_ts + timedelta(days=5)).to_datetime64().astype("datetime64[ns]"),
        )
        if start_price is None or end_price is None or start_price <= 0:
            return np.nan
        return float((end_price / start_price) - 1.0)
