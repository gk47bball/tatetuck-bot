from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import prepare

from .storage import LocalResearchStore


BIOPHARMA_NAME_KEYWORDS = (
    "bio",
    "biopharma",
    "biotherapeutics",
    "biotech",
    "biosciences",
    "genomics",
    "gene",
    "genetic",
    "medicine",
    "medicines",
    "oncology",
    "pharma",
    "pharmaceutical",
    "pharmaceuticals",
    "therapeutic",
    "therapeutics",
    "vaccine",
)


@dataclass(slots=True)
class UniverseCandidate:
    ticker: str
    company_name: str
    source: str
    rank_score: float


class UniverseResolver:
    def __init__(self, store: LocalResearchStore | None = None):
        self.store = store or LocalResearchStore()

    def resolve_default_universe(self, limit: int = 0, prefer_archive: bool = True) -> list[tuple[str, str]]:
        archived = self.archived_scored_universe(limit=limit)
        if prefer_archive and archived:
            return archived
        live_candidates = self.live_biopharma_candidates(limit=limit)
        if live_candidates:
            return live_candidates
        fallback = list(prepare.BENCHMARK_TICKERS)
        if limit and limit > 0:
            return fallback[:limit]
        return fallback

    def archived_scored_universe(self, limit: int = 0) -> list[tuple[str, str]]:
        snapshots = self.store.read_table("company_snapshots")
        if snapshots.empty:
            return []
        latest = snapshots.sort_values("as_of").drop_duplicates(subset=["ticker"], keep="last").copy()
        latest["num_catalysts"] = pd.to_numeric(latest.get("num_catalysts"), errors="coerce").fillna(0.0)
        latest["market_cap"] = pd.to_numeric(latest.get("market_cap"), errors="coerce").fillna(0.0)
        latest["revenue"] = pd.to_numeric(latest.get("revenue"), errors="coerce").fillna(0.0)
        latest["archive_score"] = (
            latest["num_catalysts"] * 3.0
            + (latest["revenue"] > 0.0).astype(float)
            + latest["market_cap"].rank(pct=True, method="average").fillna(0.0)
        )
        latest = latest.sort_values(["archive_score", "ticker"], ascending=[False, True])
        rows = [
            (str(row["ticker"]).upper(), str(row.get("company_name") or row["ticker"]))
            for _, row in latest.iterrows()
        ]
        if limit and limit > 0:
            return rows[:limit]
        return rows

    def live_biopharma_candidates(self, limit: int = 0) -> list[tuple[str, str]]:
        membership = self.store.read_table("universe_membership")
        if membership.empty:
            return []
        latest = membership.sort_values("as_of").drop_duplicates(subset=["ticker"], keep="last").copy()
        latest = latest[~latest["is_delisted"].fillna(False).astype(bool)].copy()
        if latest.empty:
            return []
        latest["company_name"] = latest["company_name"].fillna(latest["ticker"])
        if "security_type" not in latest.columns:
            latest["security_type"] = "Common Stock"
        latest["security_type"] = latest["security_type"].fillna("").astype(str)
        latest["keyword_hit"] = latest["company_name"].astype(str).str.lower().map(self._is_biopharma_name)
        latest = latest[
            latest["keyword_hit"]
            & latest["security_type"].str.contains("stock|common|adr|ordinary", case=False, regex=True)
        ].copy()
        if latest.empty:
            return []
        latest["has_exact_event"] = latest.get("has_exact_event", False).fillna(False).astype(bool)
        latest["candidate_score"] = latest["has_exact_event"].astype(float) * 2.0
        latest = latest.sort_values(["candidate_score", "ticker"], ascending=[False, True])
        rows = [
            (str(row["ticker"]).upper(), str(row["company_name"]))
            for _, row in latest.iterrows()
        ]
        if limit and limit > 0:
            return rows[:limit]
        return rows

    @staticmethod
    def _is_biopharma_name(name: str) -> bool:
        name_lc = str(name).lower()
        return any(keyword in name_lc for keyword in BIOPHARMA_NAME_KEYWORDS)
