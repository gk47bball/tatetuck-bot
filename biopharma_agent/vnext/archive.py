from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from .entities import CompanyAnalysis
from .facade import TatetuckPlatform


@dataclass(slots=True)
class ArchiveSummary:
    archived_companies: int
    archived_at: str
    sec_enriched_companies: int
    financing_flagged_companies: int
    snapshot_rows: int
    feature_rows: int
    prediction_rows: int
    store_dir: str
    top_ideas: list[dict[str, str | float]]


def archive_universe(
    platform: TatetuckPlatform,
    universe: Iterable[tuple[str, str]],
    include_literature: bool = False,
    as_of: datetime | None = None,
) -> tuple[list[CompanyAnalysis], ArchiveSummary]:
    analyses = platform.analyze_universe(universe, include_literature=include_literature, as_of=as_of)
    archived_at = analyses[0].snapshot.as_of if analyses else datetime.utcnow().isoformat()
    sec_enriched = sum(1 for analysis in analyses if analysis.snapshot.metadata.get("sec_cik"))
    financing_flagged = sum(1 for analysis in analyses if analysis.snapshot.financing_events)
    snapshot_rows = len(platform.store.read_table("company_snapshots"))
    feature_rows = len(platform.store.read_table("feature_vectors"))
    prediction_rows = len(platform.store.read_table("predictions"))
    top_ideas = [
        {
            "ticker": analysis.snapshot.ticker,
            "target_weight": round(float(analysis.portfolio.target_weight), 2),
            "scenario": analysis.portfolio.scenario,
            "thesis_horizon": analysis.portfolio.thesis_horizon,
        }
        for analysis in sorted(analyses, key=lambda item: item.portfolio.target_weight, reverse=True)[:5]
    ]
    summary = ArchiveSummary(
        archived_companies=len(analyses),
        archived_at=archived_at,
        sec_enriched_companies=sec_enriched,
        financing_flagged_companies=financing_flagged,
        snapshot_rows=snapshot_rows,
        feature_rows=feature_rows,
        prediction_rows=prediction_rows,
        store_dir=str(platform.store.base_dir),
        top_ideas=top_ideas,
    )
    return analyses, summary
