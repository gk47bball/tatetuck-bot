from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Iterable, Protocol

from .eodhd import EODHDEventTapeClient
from .sources import SECXBRLClient, exact_sec_event_tape_rows
from .storage import LocalResearchStore
from .taxonomy import is_exact_timing_event, is_synthetic_event


class EventTapeFetcher(Protocol):
    def fetch_event_payload(
        self,
        ticker: str,
        as_of: datetime,
        lookback_days: int = 7,
        lookahead_days: int = 120,
    ) -> dict[str, Any]: ...


class SECFetcher(Protocol):
    def fetch_company_facts(self, ticker: str) -> dict[str, Any]: ...


@dataclass(slots=True)
class TriggerIngestionSummary:
    captured_at: str
    symbols_requested: int
    symbols_polled: int
    symbols_with_new_events: int
    eodhd_event_rows: int
    sec_event_rows: int
    triggered_symbols: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


class RealTimeTriggerIngestor:
    def __init__(
        self,
        store: LocalResearchStore | None = None,
        event_client: EventTapeFetcher | None = None,
        sec_client: SECFetcher | None = None,
    ):
        self.store = store or LocalResearchStore()
        self.event_client = event_client or EODHDEventTapeClient(store=self.store)
        self.sec_client = sec_client or SECXBRLClient()

    def ingest_symbols(
        self,
        symbols: Iterable[str],
        *,
        as_of: datetime,
        event_lookback_days: int = 7,
        event_lookahead_days: int = 120,
        sec_lookback_days: int = 7,
    ) -> TriggerIngestionSummary:
        captured_at = as_of.isoformat()
        requested = [str(symbol).upper().strip() for symbol in symbols if str(symbol).strip()]
        deduped_symbols = list(dict.fromkeys(requested))
        event_rows: list[dict[str, Any]] = []
        triggered_symbols: list[str] = []
        warnings: list[str] = []
        eodhd_count = 0
        sec_count = 0
        polled = 0

        for symbol in deduped_symbols:
            polled += 1
            symbol_rows: list[dict[str, Any]] = []
            try:
                event_payload = self.event_client.fetch_event_payload(
                    symbol,
                    as_of=as_of,
                    lookback_days=event_lookback_days,
                    lookahead_days=event_lookahead_days,
                )
                if isinstance(event_payload, dict):
                    raw_key = f"{symbol}_{captured_at.replace(':', '-')}"
                    self.store.write_raw_payload("eodhd_event_tape", raw_key, event_payload)
                    eodhd_rows = self._event_rows_from_payload(symbol=symbol, payload=event_payload, as_of=captured_at)
                    symbol_rows.extend(eodhd_rows)
                    eodhd_count += len(eodhd_rows)
            except Exception as exc:
                warnings.append(f"{symbol} EODHD trigger ingest failed: {type(exc).__name__}: {exc}")

            try:
                sec_payload = self.sec_client.fetch_company_facts(symbol)
                if isinstance(sec_payload, dict):
                    raw_key = f"{symbol}_{captured_at.replace(':', '-')}"
                    self.store.write_raw_payload("sec_xbrl", raw_key, sec_payload)
                    sec_rows = exact_sec_event_tape_rows(
                        symbol,
                        sec_payload,
                        as_of=as_of,
                        lookback_days=sec_lookback_days,
                    )
                    symbol_rows.extend(sec_rows)
                    sec_count += len(sec_rows)
            except Exception as exc:
                warnings.append(f"{symbol} SEC trigger ingest failed: {type(exc).__name__}: {exc}")

            if symbol_rows:
                deduped = self._dedupe_rows(symbol_rows)
                event_rows.extend(deduped)
                if any(bool(row.get("timing_exact")) for row in deduped):
                    triggered_symbols.append(symbol)

        if event_rows:
            self.store.append_records("event_tape", event_rows)

        summary = TriggerIngestionSummary(
            captured_at=captured_at,
            symbols_requested=len(deduped_symbols),
            symbols_polled=polled,
            symbols_with_new_events=len(set(triggered_symbols)),
            eodhd_event_rows=eodhd_count,
            sec_event_rows=sec_count,
            triggered_symbols=sorted(set(triggered_symbols)),
            warnings=warnings,
        )
        self.store.append_records("trigger_event_ingestion", [summary.to_row()])
        self.store.write_raw_payload(
            "trigger_event_ingestion",
            f"trigger_ingestion_{captured_at.replace(':', '-')}",
            asdict(summary),
        )
        return summary

    @staticmethod
    def _event_rows_from_payload(symbol: str, payload: dict[str, Any], as_of: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for event in payload.get("events", []):
            expected = event.get("expected_date")
            status = str(event.get("status") or "")
            title = str(event.get("title") or "")
            rows.append(
                {
                    "ticker": str(symbol).upper(),
                    "as_of": as_of,
                    "event_id": str(event.get("event_id") or f"{symbol}:event:{expected}"),
                    "event_type": str(event.get("event_type") or ""),
                    "title": title,
                    "event_timestamp": expected,
                    "status": status,
                    "timing_exact": bool(is_exact_timing_event(status, expected, title)),
                    "timing_synthetic": bool(is_synthetic_event(status, title)),
                    "source": str(event.get("source") or payload.get("source") or "external_event"),
                    "url": event.get("url"),
                }
            )
        return rows

    @staticmethod
    def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in rows:
            key = (
                str(row.get("event_id") or ""),
                str(row.get("event_timestamp") or ""),
                str(row.get("status") or ""),
            )
            existing = deduped.get(key)
            if existing is None:
                deduped[key] = row
                continue
            current_exact = bool(row.get("timing_exact"))
            existing_exact = bool(existing.get("timing_exact"))
            if current_exact and not existing_exact:
                deduped[key] = row
        return list(deduped.values())
