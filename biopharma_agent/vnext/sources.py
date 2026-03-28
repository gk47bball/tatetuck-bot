from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .entities import CompanySnapshot
from .graph import build_company_snapshot, fetch_legacy_snapshot
from .storage import LocalResearchStore


class SECXBRLClient:
    """Placeholder for future SEC/XBRL enrichment.

    The vNext platform exposes a canonical interface today so new sources can be
    integrated without changing downstream feature, modeling, or UI code.
    """

    def fetch_company_facts(self, ticker: str) -> dict[str, Any]:
        return {"ticker": ticker, "records": [], "source": "sec_xbrl_placeholder"}


class CorporateCalendarClient:
    def fetch_company_calendar(self, ticker: str) -> dict[str, Any]:
        return {"ticker": ticker, "events": [], "source": "calendar_placeholder"}


class IngestionService:
    def __init__(self, store: LocalResearchStore | None = None):
        self.store = store or LocalResearchStore()
        self.sec_client = SECXBRLClient()
        self.calendar_client = CorporateCalendarClient()

    def ingest_company(self, ticker: str, company_name: str | None = None, as_of: datetime | None = None) -> CompanySnapshot:
        as_of = as_of or datetime.now(timezone.utc)
        raw = fetch_legacy_snapshot(ticker, company_name)
        snapshot = build_company_snapshot(raw, as_of=as_of)

        raw_key = f"{ticker}_{as_of.isoformat().replace(':', '-')}"
        self.store.write_raw_payload("legacy_prepare", raw_key, raw)
        self.store.write_raw_payload("sec_xbrl", raw_key, self.sec_client.fetch_company_facts(ticker))
        self.store.write_raw_payload("corp_calendar", raw_key, self.calendar_client.fetch_company_calendar(ticker))
        self.store.write_snapshot(snapshot)
        return snapshot
