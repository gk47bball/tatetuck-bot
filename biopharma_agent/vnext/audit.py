from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from typing import Any

import pandas as pd

from .entities import CompanySnapshot, ModelPrediction, PortfolioRecommendation, SignalArtifact
from .features import FeatureEngineer
from .models import EventDrivenEnsemble
from .portfolio import PortfolioConstructor, aggregate_signal
from .replay import snapshot_from_dict
from .storage import LocalResearchStore


def _coerce_json(value: Any) -> Any:
    if isinstance(value, str) and value and value[0] in "[{":
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _coerce_list(value: Any) -> list[Any]:
    value = _coerce_json(value)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return value
    return [value]


@dataclass(slots=True)
class ResearchAudit:
    generated_at: str
    store_dir: str
    latest_snapshot_at: str | None
    latest_ticker_count: int
    signal_ticker_count: int
    recommendation_ticker_count: int
    sec_enriched_pct: float
    catalyst_coverage_pct: float
    near_term_catalyst_pct: float
    approved_product_pct: float
    revenue_positive_pct: float
    financing_flagged_pct: float
    zero_market_cap_pct: float
    expected_return_std: float
    confidence_std: float
    target_weight_std: float
    actionable_count: int
    scenario_diversity: int
    archetype_diversity: int
    scenario_counts: dict[str, int]
    archetype_counts: dict[str, int]
    primary_event_type_counts: dict[str, int]
    top_ideas: list[dict[str, object]]
    blockers: list[str]
    warnings: list[str]

    def to_row(self) -> dict[str, object]:
        return asdict(self)


class ResearchAuditBuilder:
    def __init__(self, store: LocalResearchStore | None = None):
        self.store = store or LocalResearchStore()
        self.portfolio = PortfolioConstructor(store=self.store)
        self.features = FeatureEngineer()
        self.ensemble = EventDrivenEnsemble(store=self.store)

    def build(self, top_n: int = 10, refresh_company_views: bool = False) -> ResearchAudit:
        latest_snapshots = self._load_latest_snapshots()
        if not latest_snapshots:
            report = ResearchAudit(
                generated_at=pd.Timestamp.now(tz="UTC").isoformat(),
                store_dir=str(self.store.base_dir),
                latest_snapshot_at=None,
                latest_ticker_count=0,
                signal_ticker_count=0,
                recommendation_ticker_count=0,
                sec_enriched_pct=0.0,
                catalyst_coverage_pct=0.0,
                near_term_catalyst_pct=0.0,
                approved_product_pct=0.0,
                revenue_positive_pct=0.0,
                financing_flagged_pct=0.0,
                zero_market_cap_pct=0.0,
                expected_return_std=0.0,
                confidence_std=0.0,
                target_weight_std=0.0,
                actionable_count=0,
                scenario_diversity=0,
                archetype_diversity=0,
                scenario_counts={},
                archetype_counts={},
                primary_event_type_counts={},
                top_ideas=[],
                blockers=["No archived company snapshots are available for research auditing."],
                warnings=[],
            )
            self.store.append_records("research_audits", [report.to_row()])
            return report

        signals = self.store.read_table("signal_artifacts")
        recommendations = self.store.read_table("portfolio_recommendations")
        predictions = self.store.read_table("predictions")
        features = self.store.read_table("feature_vectors")

        rows: list[dict[str, object]] = []
        computed_signals: list[SignalArtifact] = []
        computed_recommendations: list[PortfolioRecommendation] = []
        latest_snapshot_at = max(snapshot.as_of for snapshot in latest_snapshots.values())

        for ticker, snapshot in sorted(latest_snapshots.items()):
            signal = self._resolve_signal(
                snapshot,
                signals=signals,
                predictions=predictions,
                features=features,
                refresh_company_views=refresh_company_views,
            )
            recommendation = self._resolve_recommendation(
                snapshot,
                signal=signal,
                recommendations=recommendations,
                refresh_company_views=refresh_company_views,
            )

            if refresh_company_views or not self._has_signal_row(signals, ticker, snapshot.as_of):
                computed_signals.append(signal)
            if refresh_company_views or not self._has_recommendation_row(recommendations, ticker, snapshot.as_of):
                computed_recommendations.append(recommendation)

            primary_event_type = self._primary_event_type(snapshot)
            near_term_catalyst = any(event.horizon_days <= 180 for event in snapshot.catalyst_events)
            metadata = snapshot.metadata or {}
            rows.append(
                {
                    "ticker": ticker,
                    "company_name": snapshot.company_name,
                    "as_of": snapshot.as_of,
                    "market_cap": float(snapshot.market_cap or 0.0),
                    "revenue": float(snapshot.revenue or 0.0),
                    "num_catalysts": len(snapshot.catalyst_events),
                    "num_approved_products": len(snapshot.approved_products),
                    "num_financing_events": len(snapshot.financing_events),
                    "sec_enriched": bool(metadata.get("sec_cik")),
                    "near_term_catalyst": near_term_catalyst,
                    "primary_event_type": primary_event_type or "none",
                    "best_phase": metadata.get("best_phase") or self._best_phase(snapshot),
                    "archetype": self._classify_archetype(snapshot),
                    "expected_return": float(signal.expected_return),
                    "confidence": float(signal.confidence),
                    "target_weight": float(recommendation.target_weight),
                    "scenario": recommendation.scenario,
                }
            )

        if computed_signals:
            self.store.write_signal_artifacts(computed_signals)
        if computed_recommendations:
            self.store.write_portfolio_recommendations(computed_recommendations)

        frame = pd.DataFrame(rows).sort_values(["target_weight", "confidence"], ascending=[False, False])
        scenario_counts = {str(key): int(value) for key, value in frame["scenario"].value_counts().to_dict().items()}
        archetype_counts = {str(key): int(value) for key, value in frame["archetype"].value_counts().to_dict().items()}

        actionable = frame[frame["target_weight"] >= 1.0].copy()
        event_source = actionable if not actionable.empty else frame
        primary_event_type_counts = {
            str(key): int(value)
            for key, value in event_source["primary_event_type"].value_counts().to_dict().items()
        }

        blockers: list[str] = []
        warnings: list[str] = []

        latest_ticker_count = len(frame)
        expected_return_std = float(frame["expected_return"].std(ddof=0) or 0.0)
        confidence_std = float(frame["confidence"].std(ddof=0) or 0.0)
        target_weight_std = float(frame["target_weight"].std(ddof=0) or 0.0)
        sec_enriched_pct = float(frame["sec_enriched"].mean()) if latest_ticker_count else 0.0
        catalyst_coverage_pct = float((frame["num_catalysts"] > 0).mean()) if latest_ticker_count else 0.0
        near_term_catalyst_pct = float(frame["near_term_catalyst"].mean()) if latest_ticker_count else 0.0
        approved_product_pct = float((frame["num_approved_products"] > 0).mean()) if latest_ticker_count else 0.0
        revenue_positive_pct = float((frame["revenue"] > 0).mean()) if latest_ticker_count else 0.0
        financing_flagged_pct = float((frame["num_financing_events"] > 0).mean()) if latest_ticker_count else 0.0
        zero_market_cap_pct = float((frame["market_cap"] <= 0).mean()) if latest_ticker_count else 0.0

        if latest_ticker_count < 20:
            blockers.append(f"Only {latest_ticker_count} latest snapshots are available; need broader universe coverage.")
        if sec_enriched_pct < 0.60:
            blockers.append(f"SEC enrichment coverage is only {sec_enriched_pct:.0%}; fundamental coverage is too thin.")
        if catalyst_coverage_pct < 0.70:
            blockers.append(f"Only {catalyst_coverage_pct:.0%} of the universe has catalysts; event coverage is too sparse.")
        if expected_return_std < 0.03:
            blockers.append("Expected-return dispersion is too low; the ranking engine is collapsing toward one score.")
        if target_weight_std < 0.75:
            blockers.append("Portfolio target-weight dispersion is too low; the PM layer is not differentiating ideas enough.")
        if len(actionable) < 2:
            blockers.append("Fewer than two actionable long ideas cleared the PM thresholds.")

        if confidence_std < 0.02:
            warnings.append("Confidence dispersion is low; model conviction is clustering tightly.")
        if near_term_catalyst_pct < 0.40:
            warnings.append("Less than 40% of the universe has a catalyst inside the next 180 days.")
        if zero_market_cap_pct > 0.10:
            warnings.append("More than 10% of the latest universe has zero or missing market cap.")
        if financing_flagged_pct > 0.40:
            warnings.append("Financing overhang is flagged on more than 40% of the latest universe.")
        if len(archetype_counts) < 3:
            warnings.append("Archetype coverage is narrow; the latest universe is not well balanced across biotech setups.")
        if len(scenario_counts) < 2:
            warnings.append("Scenario diversity is narrow; recommendations are clustering into too few PM playbooks.")
        if primary_event_type_counts:
            top_event_type, top_event_count = max(primary_event_type_counts.items(), key=lambda item: item[1])
            if top_event_count / max(len(event_source), 1) > 0.70:
                warnings.append(f"Primary event concentration is high: {top_event_type} drives most actionable names.")

        top_ideas = [
            {
                "ticker": row.ticker,
                "company_name": row.company_name,
                "scenario": row.scenario,
                "target_weight": round(float(row.target_weight), 2),
                "expected_return": round(float(row.expected_return), 4),
                "confidence": round(float(row.confidence), 4),
                "archetype": row.archetype,
                "primary_event_type": row.primary_event_type,
            }
            for row in frame.head(top_n).itertuples(index=False)
        ]

        report = ResearchAudit(
            generated_at=pd.Timestamp.now(tz="UTC").isoformat(),
            store_dir=str(self.store.base_dir),
            latest_snapshot_at=latest_snapshot_at,
            latest_ticker_count=latest_ticker_count,
            signal_ticker_count=int(frame["expected_return"].notna().sum()),
            recommendation_ticker_count=int(frame["target_weight"].notna().sum()),
            sec_enriched_pct=sec_enriched_pct,
            catalyst_coverage_pct=catalyst_coverage_pct,
            near_term_catalyst_pct=near_term_catalyst_pct,
            approved_product_pct=approved_product_pct,
            revenue_positive_pct=revenue_positive_pct,
            financing_flagged_pct=financing_flagged_pct,
            zero_market_cap_pct=zero_market_cap_pct,
            expected_return_std=expected_return_std,
            confidence_std=confidence_std,
            target_weight_std=target_weight_std,
            actionable_count=int(len(actionable)),
            scenario_diversity=len(scenario_counts),
            archetype_diversity=len(archetype_counts),
            scenario_counts=scenario_counts,
            archetype_counts=archetype_counts,
            primary_event_type_counts=primary_event_type_counts,
            top_ideas=top_ideas,
            blockers=blockers,
            warnings=warnings,
        )
        self.store.append_records("research_audits", [report.to_row()])
        return report

    def _load_latest_snapshots(self) -> dict[str, CompanySnapshot]:
        snapshots = self.store.read_table("company_snapshots")
        latest_by_ticker: dict[str, CompanySnapshot] = {}
        tickers: list[str]
        if not snapshots.empty and "ticker" in snapshots.columns:
            tickers = sorted(str(item) for item in snapshots["ticker"].dropna().unique().tolist())
        else:
            tickers = sorted({path.name.split("_", 1)[0] for path in self.store.list_raw_payload_paths("snapshots")})

        for ticker in tickers:
            payload = self.store.read_latest_raw_payload("snapshots", f"{ticker}_")
            if isinstance(payload, dict):
                latest_by_ticker[ticker] = snapshot_from_dict(payload)
        return latest_by_ticker

    def _resolve_signal(
        self,
        snapshot: CompanySnapshot,
        signals: pd.DataFrame,
        predictions: pd.DataFrame,
        features: pd.DataFrame,
        refresh_company_views: bool = False,
    ) -> SignalArtifact:
        signal_row = None if refresh_company_views else self._lookup_row(signals, snapshot.ticker, snapshot.as_of)
        if signal_row is not None:
            primary_event_type = signal_row.get("primary_event_type")
            if isinstance(primary_event_type, float) and pd.isna(primary_event_type):
                primary_event_type = None
            primary_event_bucket = signal_row.get("primary_event_bucket", "none")
            if isinstance(primary_event_bucket, float) and pd.isna(primary_event_bucket):
                primary_event_bucket = "none"
            return SignalArtifact(
                ticker=snapshot.ticker,
                as_of=snapshot.as_of,
                expected_return=float(signal_row["expected_return"]),
                catalyst_success_prob=float(signal_row["catalyst_success_prob"]),
                confidence=float(signal_row["confidence"]),
                crowding_risk=float(signal_row["crowding_risk"]),
                financing_risk=float(signal_row["financing_risk"]),
                thesis_horizon=str(signal_row.get("thesis_horizon", "90d")),
                rationale=_coerce_list(signal_row.get("rationale")),
                supporting_evidence=snapshot.evidence[:5],
                primary_event_type=None if primary_event_type is None else str(primary_event_type),
                primary_event_bucket=str(primary_event_bucket),
                program_predictions=[],
            )

        program_predictions = self._predictions_for_snapshot(predictions, snapshot.ticker, snapshot.as_of)
        if not program_predictions:
            program_predictions = self._score_snapshot(snapshot, features)
        return aggregate_signal(
            ticker=snapshot.ticker,
            as_of=snapshot.as_of,
            predictions=program_predictions,
            evidence_rationale=[
                f"{len(snapshot.programs)} programs and {len(snapshot.catalyst_events)} catalysts loaded from the archive.",
                f"Best phase is {snapshot.metadata.get('best_phase') or self._best_phase(snapshot)}.",
            ],
            evidence=snapshot.evidence[:5],
        )

    def _resolve_recommendation(
        self,
        snapshot: CompanySnapshot,
        signal: SignalArtifact,
        recommendations: pd.DataFrame,
        refresh_company_views: bool = False,
    ) -> PortfolioRecommendation:
        recommendation_row = None if refresh_company_views else self._lookup_row(recommendations, snapshot.ticker, snapshot.as_of)
        if recommendation_row is not None:
            primary_event_type = recommendation_row.get("primary_event_type")
            if isinstance(primary_event_type, float) and pd.isna(primary_event_type):
                primary_event_type = None
            return PortfolioRecommendation(
                ticker=snapshot.ticker,
                as_of=snapshot.as_of,
                stance=str(recommendation_row["stance"]),
                target_weight=float(recommendation_row["target_weight"]),
                max_weight=float(recommendation_row["max_weight"]),
                confidence=float(recommendation_row["confidence"]),
                scenario=str(recommendation_row["scenario"]),
                thesis_horizon=str(recommendation_row["thesis_horizon"]),
                primary_event_type=None if primary_event_type is None else str(primary_event_type),
                risk_flags=_coerce_list(recommendation_row.get("risk_flags")),
            )
        return self.portfolio.recommend(signal)

    @staticmethod
    def _lookup_row(frame: pd.DataFrame, ticker: str, as_of: str) -> pd.Series | None:
        if frame.empty:
            return None
        subset = frame[(frame["ticker"] == ticker) & (frame["as_of"] == as_of)]
        if subset.empty:
            return None
        return subset.iloc[-1]

    @staticmethod
    def _has_signal_row(signals: pd.DataFrame, ticker: str, as_of: str) -> bool:
        return ResearchAuditBuilder._lookup_row(signals, ticker, as_of) is not None

    @staticmethod
    def _has_recommendation_row(recommendations: pd.DataFrame, ticker: str, as_of: str) -> bool:
        return ResearchAuditBuilder._lookup_row(recommendations, ticker, as_of) is not None

    def _predictions_for_snapshot(self, predictions: pd.DataFrame, ticker: str, as_of: str) -> list[ModelPrediction]:
        if predictions.empty:
            return []
        subset = predictions[(predictions["ticker"] == ticker) & (predictions["as_of"] == as_of)].copy()
        if subset.empty:
            return []
        subset = subset.sort_values("entity_id").drop_duplicates(subset=["entity_id"], keep="last")
        return [
            ModelPrediction(
                entity_id=str(row["entity_id"]),
                ticker=str(row["ticker"]),
                as_of=str(row["as_of"]),
                expected_return=float(row["expected_return"]),
                catalyst_success_prob=float(row["catalyst_success_prob"]),
                confidence=float(row["confidence"]),
                crowding_risk=float(row["crowding_risk"]),
                financing_risk=float(row["financing_risk"]),
                thesis_horizon=str(row["thesis_horizon"]),
                model_name=str(row["model_name"]),
                model_version=str(row["model_version"]),
                metadata=_coerce_json(row.get("metadata")) if "metadata" in subset.columns else {},
            )
            for _, row in subset.iterrows()
        ]

    def _score_snapshot(self, snapshot: CompanySnapshot, features: pd.DataFrame) -> list[ModelPrediction]:
        if not features.empty:
            subset = features[(features["ticker"] == snapshot.ticker) & (features["as_of"] == snapshot.as_of)].copy()
            if not subset.empty:
                vectors = [self._feature_vector_from_row(row) for _, row in subset.iterrows()]
                program_vectors = [vector for vector in vectors if not vector.metadata.get("aggregate")]
                return self.ensemble.score(program_vectors)
        vectors = self.features.build_all(snapshot)
        program_vectors = [vector for vector in vectors if not vector.metadata.get("aggregate")]
        return self.ensemble.score(program_vectors)

    @staticmethod
    def _feature_vector_from_row(row: pd.Series):
        from .entities import FeatureVector

        feature_family = {
            key: float(row[key])
            for key in row.index
            if key
            not in {"entity_id", "ticker", "as_of", "thesis_horizon", "evaluation_date"}
            and not key.startswith("meta_")
            and not key.startswith("target_")
        }
        metadata = {key[5:]: row[key] for key in row.index if key.startswith("meta_")}
        return FeatureVector(
            entity_id=str(row["entity_id"]),
            ticker=str(row["ticker"]),
            as_of=str(row["as_of"]),
            thesis_horizon=str(row["thesis_horizon"]),
            feature_family=feature_family,
            metadata=metadata,
        )

    @staticmethod
    def _best_phase(snapshot: CompanySnapshot) -> str:
        if not snapshot.programs:
            return "UNKNOWN"
        order = {"APPROVED": 6, "NDA_BLA": 5, "PHASE3": 4, "PHASE2": 3, "PHASE1": 2, "EARLY_PHASE1": 1}
        return max(snapshot.programs, key=lambda program: order.get(program.phase, 0)).phase

    @classmethod
    def _classify_archetype(cls, snapshot: CompanySnapshot) -> str:
        best_phase = snapshot.metadata.get("best_phase") or cls._best_phase(snapshot)
        has_approved = bool(snapshot.approved_products)
        if snapshot.revenue >= 1_000_000_000 or (has_approved and snapshot.revenue >= 500_000_000):
            return "mature_commercial"
        if snapshot.revenue >= 25_000_000 or has_approved:
            return "early_commercial"
        if best_phase in {"PHASE3", "NDA_BLA"}:
            return "late_stage_clinical"
        if any(program.modality in {"gene editing", "gene therapy", "cell therapy", "platform"} for program in snapshot.programs):
            return "platform_clinical"
        return "early_clinical"

    @staticmethod
    def _primary_event_type(snapshot: CompanySnapshot) -> str | None:
        if not snapshot.catalyst_events:
            return None
        event = min(snapshot.catalyst_events, key=lambda item: (item.horizon_days, -item.importance))
        return event.event_type
