from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .entities import (
    ApprovedProduct,
    CatalystEvent,
    CompanySnapshot,
    EvidenceSnippet,
    FinancingEvent,
    Program,
    Trial,
)
from .features import FeatureEngineer
from .models import EventDrivenEnsemble
from .storage import LocalResearchStore


@dataclass(slots=True)
class ReplaySummary:
    replayed_snapshots: int
    replayed_tickers: int
    feature_rows_written: int
    prediction_rows_written: int
    earliest_as_of: str | None
    latest_as_of: str | None
    store_dir: str


def snapshot_from_dict(payload: dict) -> CompanySnapshot:
    programs = [
        Program(
            program_id=program["program_id"],
            name=program["name"],
            modality=program["modality"],
            phase=program["phase"],
            conditions=list(program.get("conditions", [])),
            trials=[
                Trial(
                    trial_id=trial["trial_id"],
                    title=trial["title"],
                    phase=trial["phase"],
                    status=trial["status"],
                    conditions=list(trial.get("conditions", [])),
                    interventions=list(trial.get("interventions", [])),
                    enrollment=int(trial.get("enrollment", 0) or 0),
                    primary_outcomes=list(trial.get("primary_outcomes", [])),
                )
                for trial in program.get("trials", [])
            ],
            pos_prior=float(program["pos_prior"]),
            tam_estimate=float(program["tam_estimate"]),
            catalyst_events=[
                CatalystEvent(
                    event_id=event["event_id"],
                    program_id=event.get("program_id"),
                    event_type=event["event_type"],
                    title=event["title"],
                    expected_date=event.get("expected_date"),
                    horizon_days=int(event["horizon_days"]),
                    probability=float(event["probability"]),
                    importance=float(event["importance"]),
                    crowdedness=float(event["crowdedness"]),
                    status=event.get("status", "anticipated"),
                )
                for event in program.get("catalyst_events", [])
            ],
            evidence=[
                EvidenceSnippet(
                    source=item["source"],
                    source_id=item["source_id"],
                    title=item["title"],
                    excerpt=item["excerpt"],
                    url=item.get("url"),
                    as_of=item.get("as_of"),
                    confidence=float(item.get("confidence", 0.5)),
                )
                for item in program.get("evidence", [])
            ],
        )
        for program in payload.get("programs", [])
    ]

    return CompanySnapshot(
        ticker=payload["ticker"],
        company_name=payload["company_name"],
        as_of=payload["as_of"],
        market_cap=float(payload["market_cap"]),
        enterprise_value=float(payload["enterprise_value"]),
        revenue=float(payload["revenue"]),
        cash=float(payload["cash"]),
        debt=float(payload["debt"]),
        momentum_3mo=payload.get("momentum_3mo"),
        trailing_6mo_return=payload.get("trailing_6mo_return"),
        volatility=payload.get("volatility"),
        programs=programs,
        approved_products=[
            ApprovedProduct(
                product_id=item["product_id"],
                name=item["name"],
                indication=item["indication"],
                annual_revenue=float(item["annual_revenue"]),
                growth_signal=float(item["growth_signal"]),
            )
            for item in payload.get("approved_products", [])
        ],
        catalyst_events=[
            CatalystEvent(
                event_id=item["event_id"],
                program_id=item.get("program_id"),
                event_type=item["event_type"],
                title=item["title"],
                expected_date=item.get("expected_date"),
                horizon_days=int(item["horizon_days"]),
                probability=float(item["probability"]),
                importance=float(item["importance"]),
                crowdedness=float(item["crowdedness"]),
                status=item.get("status", "anticipated"),
            )
            for item in payload.get("catalyst_events", [])
        ],
        financing_events=[
            FinancingEvent(
                event_id=item["event_id"],
                event_type=item["event_type"],
                probability=float(item["probability"]),
                horizon_days=int(item["horizon_days"]),
                expected_dilution_pct=float(item["expected_dilution_pct"]),
                summary=item["summary"],
            )
            for item in payload.get("financing_events", [])
        ],
        evidence=[
            EvidenceSnippet(
                source=item["source"],
                source_id=item["source_id"],
                title=item["title"],
                excerpt=item["excerpt"],
                url=item.get("url"),
                as_of=item.get("as_of"),
                confidence=float(item.get("confidence", 0.5)),
            )
            for item in payload.get("evidence", [])
        ],
        metadata=dict(payload.get("metadata", {})),
    )


class HistoricalReplayEngine:
    def __init__(self, store: LocalResearchStore | None = None):
        self.store = store or LocalResearchStore()
        self.features = FeatureEngineer()
        self.ensemble = EventDrivenEnsemble(store=self.store)

    def rebuild_from_archived_snapshots(
        self,
        ticker: str | None = None,
        limit: int = 0,
    ) -> ReplaySummary:
        paths = self.store.list_raw_payload_paths("snapshots", f"{ticker}_" if ticker else None)
        snapshots: list[CompanySnapshot] = []
        for path in paths:
            snapshot = self._load_snapshot(path)
            if snapshot is None:
                continue
            snapshots.append(snapshot)

        snapshots.sort(key=lambda item: (item.as_of, item.ticker))
        if limit > 0:
            snapshots = snapshots[-limit:]

        feature_rows_written = 0
        prediction_rows_written = 0
        tickers_seen: set[str] = set()
        for snapshot in snapshots:
            tickers_seen.add(snapshot.ticker)
            self.store.write_snapshot(snapshot)
            vectors = self.features.build_all(snapshot)
            self.store.write_feature_vectors(vectors)
            feature_rows_written += len(vectors)
            program_vectors = [vector for vector in vectors if not vector.metadata.get("aggregate")]
            predictions = self.ensemble.score(program_vectors)
            prediction_rows_written += len(predictions)

        return ReplaySummary(
            replayed_snapshots=len(snapshots),
            replayed_tickers=len(tickers_seen),
            feature_rows_written=feature_rows_written,
            prediction_rows_written=prediction_rows_written,
            earliest_as_of=snapshots[0].as_of if snapshots else None,
            latest_as_of=snapshots[-1].as_of if snapshots else None,
            store_dir=str(self.store.base_dir),
        )

    @staticmethod
    def _load_snapshot(path: Path) -> CompanySnapshot | None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        return snapshot_from_dict(payload)
