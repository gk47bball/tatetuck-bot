from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from .entities import CatalystEvent, CompanySnapshot, EvidenceSnippet, FinancingEvent
from .graph import build_company_snapshot, fetch_legacy_snapshot, infer_runway_months
from .market_profile import update_snapshot_profile
from .storage import LocalResearchStore


def _has_finance_data(payload: dict[str, Any]) -> bool:
    finance = payload.get("finance", {})
    if not isinstance(finance, dict):
        return False
    keys = ("marketCap", "enterpriseValue", "totalRevenue", "cash", "debt", "price_now")
    return any(finance.get(key) not in (None, "") for key in keys)


def _has_trial_data(payload: dict[str, Any]) -> bool:
    return bool(payload.get("trials")) or bool(payload.get("num_total_trials")) or bool(payload.get("drug_names"))


def _merge_legacy_payload(primary: dict[str, Any], fallback: dict[str, Any] | None) -> tuple[dict[str, Any], bool]:
    if not fallback:
        return primary, False

    merged = dict(fallback)
    merged.update(primary)

    fallback_finance = fallback.get("finance", {}) if isinstance(fallback.get("finance"), dict) else {}
    primary_finance = primary.get("finance", {}) if isinstance(primary.get("finance"), dict) else {}
    merged["finance"] = dict(fallback_finance)
    merged["finance"].update({key: value for key, value in primary_finance.items() if value is not None})

    used_fallback = False
    if not _has_finance_data(primary):
        merged["finance"] = dict(fallback_finance)
        used_fallback = True

    if not _has_trial_data(primary):
        trial_keys = (
            "trials",
            "num_trials",
            "num_total_trials",
            "num_inactive_trials",
            "drug_names",
            "best_phase",
            "base_pos",
            "total_enrollment",
            "max_single_enrollment",
            "phase_enrollment",
            "phase_trial_counts",
            "fda_adverse_events",
            "fda_serious_events",
            "fda_serious_ratio",
            "pubmed_papers",
            "num_papers",
            "conditions",
        )
        for key in trial_keys:
            if key in fallback:
                merged[key] = fallback[key]
        used_fallback = True

    return merged, used_fallback


def _recover_sec_payload(primary: dict[str, Any], fallback: dict[str, Any] | None) -> tuple[dict[str, Any], bool]:
    parsed = primary.get("parsed", {})
    if primary.get("cik") or parsed.get("recent_filings"):
        return primary, False
    if fallback:
        recovered = dict(fallback)
        if primary.get("ticker"):
            recovered["ticker"] = primary["ticker"]
        return recovered, True
    return primary, False


def _recover_calendar_payload(primary: dict[str, Any], fallback: dict[str, Any] | None) -> tuple[dict[str, Any], bool]:
    if primary.get("events"):
        return primary, False
    if fallback:
        recovered = dict(fallback)
        if primary.get("ticker"):
            recovered["ticker"] = primary["ticker"]
        return recovered, True
    return primary, False


class SECXBRLClient:
    """Placeholder for future SEC/XBRL enrichment.

    The vNext platform exposes a canonical interface today so new sources can be
    integrated without changing downstream feature, modeling, or UI code.
    """

    TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
    COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

    def __init__(self):
        user_agent = os.environ.get("SEC_USER_AGENT", "TatetuckBot/1.0 support@tatetuck.local")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self._ticker_map: dict[str, str] | None = None

    def _get_json(self, url: str) -> dict[str, Any] | None:
        try:
            response = self.session.get(url, timeout=20)
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            return None

    def _ticker_lookup(self) -> dict[str, str]:
        if self._ticker_map is not None:
            return self._ticker_map
        payload = self._get_json(self.TICKER_MAP_URL) or {}
        lookup: dict[str, str] = {}
        for item in payload.values():
            ticker = str(item.get("ticker", "")).upper()
            cik = str(item.get("cik_str", "")).zfill(10)
            if ticker and cik:
                lookup[ticker] = cik
        self._ticker_map = lookup
        return lookup

    def resolve_cik(self, ticker: str) -> str | None:
        return self._ticker_lookup().get(ticker.upper())

    def fetch_company_facts(self, ticker: str) -> dict[str, Any]:
        cik = self.resolve_cik(ticker)
        if not cik:
            return {"ticker": ticker, "records": [], "source": "sec_xbrl", "status": "missing_cik"}

        submissions = self._get_json(self.SUBMISSIONS_URL.format(cik=cik)) or {}
        company_facts = self._get_json(self.COMPANY_FACTS_URL.format(cik=cik)) or {}
        parsed = self._parse_company_payloads(cik, submissions, company_facts)
        return {
            "ticker": ticker,
            "cik": cik,
            "submissions": submissions,
            "company_facts": company_facts,
            "parsed": parsed,
            "source": "sec_xbrl",
            "status": "ok",
        }

    def _parse_company_payloads(
        self,
        cik: str,
        submissions: dict[str, Any],
        company_facts: dict[str, Any],
    ) -> dict[str, Any]:
        recent = submissions.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        accession_numbers = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        recent_filings: list[dict[str, Any]] = []
        for form, filing_date, accession, primary_doc in zip(forms[:25], filing_dates[:25], accession_numbers[:25], primary_docs[:25]):
            accession_clean = str(accession).replace("-", "")
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_clean}/{primary_doc}" if accession and primary_doc else None
            recent_filings.append(
                {
                    "form": form,
                    "filing_date": filing_date,
                    "accession_number": accession,
                    "primary_document": primary_doc,
                    "url": filing_url,
                }
            )

        facts = company_facts.get("facts", {}).get("us-gaap", {})
        revenue_ttm = self._extract_latest_value(
            facts,
            ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
        )
        cash_latest = self._extract_latest_value(facts, ["CashAndCashEquivalentsAtCarryingValue"])
        operating_cashflow = self._extract_latest_value(
            facts,
            ["NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
        )

        last_10q = next((item["filing_date"] for item in recent_filings if item["form"] == "10-Q"), None)
        last_10k = next((item["filing_date"] for item in recent_filings if item["form"] == "10-K"), None)
        recent_offering_forms = [item for item in recent_filings if item["form"] in {"S-3", "424B5", "424B3", "F-3", "S-1"}]

        return {
            "recent_filings": recent_filings,
            "last_10q_date": last_10q,
            "last_10k_date": last_10k,
            "revenue_ttm": revenue_ttm,
            "cash_latest": cash_latest,
            "operating_cashflow": operating_cashflow,
            "recent_offering_forms": recent_offering_forms,
        }

    @staticmethod
    def _extract_latest_value(facts: dict[str, Any], candidates: list[str]) -> float | None:
        for key in candidates:
            item = facts.get(key)
            if not item:
                continue
            units = item.get("units", {})
            usd_entries = units.get("USD", [])
            if not usd_entries:
                continue
            sorted_entries = sorted(
                (entry for entry in usd_entries if entry.get("val") is not None),
                key=lambda entry: (entry.get("end") or "", entry.get("fy") or 0),
                reverse=True,
            )
            if sorted_entries:
                return float(sorted_entries[0]["val"])
        return None


class CorporateCalendarClient:
    def fetch_company_calendar(self, ticker: str, sec_payload: dict[str, Any] | None = None) -> dict[str, Any]:
        sec_payload = sec_payload or {}
        parsed = sec_payload.get("parsed", {})
        events: list[dict[str, Any]] = []

        last_10q = parsed.get("last_10q_date")
        last_10k = parsed.get("last_10k_date")
        if last_10q:
            next_earnings = datetime.fromisoformat(last_10q) + timedelta(days=85)
            events.append(
                {
                    "event_type": "earnings",
                    "title": f"{ticker} estimated quarterly update",
                    "expected_date": next_earnings.date().isoformat(),
                }
            )
        elif last_10k:
            next_earnings = datetime.fromisoformat(last_10k) + timedelta(days=90)
            events.append(
                {
                    "event_type": "earnings",
                    "title": f"{ticker} estimated quarterly update",
                    "expected_date": next_earnings.date().isoformat(),
                }
            )

        return {"ticker": ticker, "events": events, "source": "calendar_from_sec"}


def enrich_snapshot_with_external_data(
    snapshot: CompanySnapshot,
    sec_payload: dict[str, Any],
    calendar_payload: dict[str, Any],
) -> CompanySnapshot:
    parsed = sec_payload.get("parsed", {})
    if sec_payload.get("cik"):
        snapshot.metadata["sec_cik"] = sec_payload["cik"]
    if parsed.get("revenue_ttm") is not None:
        snapshot.metadata["sec_revenue_ttm"] = parsed["revenue_ttm"]
    if parsed.get("cash_latest") is not None:
        snapshot.metadata["sec_cash_latest"] = parsed["cash_latest"]
    if parsed.get("operating_cashflow") is not None:
        snapshot.metadata["sec_operating_cashflow"] = parsed["operating_cashflow"]
    if parsed.get("last_10q_date"):
        snapshot.metadata["last_10q_date"] = parsed["last_10q_date"]
    if parsed.get("last_10k_date"):
        snapshot.metadata["last_10k_date"] = parsed["last_10k_date"]

    recent_filings = parsed.get("recent_filings", [])[:3]
    for filing in recent_filings:
        snapshot.evidence.append(
            EvidenceSnippet(
                source="sec",
                source_id=filing.get("accession_number") or filing.get("form") or "sec-filing",
                title=f"{filing.get('form', 'SEC filing')} filed {filing.get('filing_date', 'unknown')}",
                excerpt=f"Recent filing {filing.get('primary_document', '')}",
                url=filing.get("url"),
                as_of=filing.get("filing_date"),
                confidence=0.75,
            )
        )

    for event in calendar_payload.get("events", []):
        expected_date = event.get("expected_date")
        if not expected_date:
            continue
        horizon_days = max(
            0,
            (datetime.fromisoformat(expected_date).date() - datetime.fromisoformat(snapshot.as_of).date()).days,
        )
        snapshot.catalyst_events.append(
            CatalystEvent(
                event_id=f"{snapshot.ticker}:calendar:{event['event_type']}:{expected_date}",
                program_id=None,
                event_type=event["event_type"],
                title=event["title"],
                expected_date=expected_date,
                horizon_days=horizon_days,
                probability=0.80,
                importance=0.45,
                crowdedness=0.35,
                status="calendar_estimate",
            )
        )

    refreshed_runway = infer_runway_months(snapshot.revenue, snapshot.cash, snapshot.debt, len(snapshot.programs))
    sec_operating_cashflow = parsed.get("operating_cashflow")
    net_cash = snapshot.cash - snapshot.debt
    if sec_operating_cashflow is not None and net_cash > 0:
        cashflow = float(sec_operating_cashflow)
        if cashflow < 0:
            sec_runway = min((net_cash / abs(cashflow)) * 12.0, 120.0)
            refreshed_runway = min(refreshed_runway, sec_runway)
        elif snapshot.revenue > 100_000_000:
            refreshed_runway = min(refreshed_runway, 120.0)
    snapshot.metadata["runway_months"] = refreshed_runway
    snapshot.metadata["runway_months_capped"] = refreshed_runway >= 120.0
    snapshot.financing_events = [
        event for event in snapshot.financing_events if event.event_type != "expected_financing"
    ]
    if refreshed_runway < 18:
        snapshot.financing_events.append(
            FinancingEvent(
                event_id=f"{snapshot.ticker}:financing",
                event_type="expected_financing",
                probability=0.80 if refreshed_runway < 12 else 0.45,
                horizon_days=90 if refreshed_runway < 12 else 180,
                expected_dilution_pct=0.18 if refreshed_runway < 12 else 0.08,
                summary=f"Estimated runway is {refreshed_runway:.1f} months.",
            )
        )

    offering_forms = parsed.get("recent_offering_forms", [])
    if offering_forms:
        latest = offering_forms[0]
        snapshot.financing_events.append(
            FinancingEvent(
                event_id=f"{snapshot.ticker}:sec-offering:{latest.get('form', 'offering')}",
                event_type="recent_offering_filing",
                probability=0.70,
                horizon_days=45,
                expected_dilution_pct=0.10,
                summary=f"Recent {latest.get('form')} filing on {latest.get('filing_date')}.",
            )
        )
        snapshot.metadata["recent_offering_signal"] = 1.0
    else:
        snapshot.metadata["recent_offering_signal"] = 0.0

    return update_snapshot_profile(snapshot)


class IngestionService:
    def __init__(self, store: LocalResearchStore | None = None):
        self.store = store or LocalResearchStore()
        self.sec_client = SECXBRLClient()
        self.calendar_client = CorporateCalendarClient()

    def ingest_company(
        self,
        ticker: str,
        company_name: str | None = None,
        as_of: datetime | None = None,
        persist: bool = True,
    ) -> CompanySnapshot:
        as_of = as_of or datetime.now(timezone.utc)
        fallback_sources: list[str] = []

        raw = fetch_legacy_snapshot(ticker, company_name)
        raw, used_cached_legacy = _merge_legacy_payload(
            raw,
            self.store.read_latest_raw_payload("legacy_prepare", f"{ticker}_"),
        )
        if used_cached_legacy:
            fallback_sources.append("legacy_prepare")

        snapshot = build_company_snapshot(raw, as_of=as_of)

        sec_payload = self.sec_client.fetch_company_facts(ticker)
        sec_payload, used_cached_sec = _recover_sec_payload(
            sec_payload,
            self.store.read_latest_raw_payload("sec_xbrl", f"{ticker}_"),
        )
        if used_cached_sec:
            fallback_sources.append("sec_xbrl")

        calendar_payload = self.calendar_client.fetch_company_calendar(ticker, sec_payload)
        calendar_payload, used_cached_calendar = _recover_calendar_payload(
            calendar_payload,
            self.store.read_latest_raw_payload("corp_calendar", f"{ticker}_"),
        )
        if used_cached_calendar:
            fallback_sources.append("corp_calendar")

        snapshot = enrich_snapshot_with_external_data(snapshot, sec_payload, calendar_payload)
        if fallback_sources:
            snapshot.metadata["fallback_sources"] = sorted(set(fallback_sources))

        if persist:
            raw_key = f"{ticker}_{as_of.isoformat().replace(':', '-')}"
            self.store.write_raw_payload("legacy_prepare", raw_key, raw)
            self.store.write_raw_payload("sec_xbrl", raw_key, sec_payload)
            self.store.write_raw_payload("corp_calendar", raw_key, calendar_payload)
            self.store.write_snapshot(snapshot)
        return snapshot
