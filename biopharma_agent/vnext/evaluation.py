from __future__ import annotations

from dataclasses import dataclass, field
import json

import numpy as np
import pandas as pd

from .entities import ModelPrediction, SignalArtifact
from .features import FeatureEngineer
from .labels import PointInTimeLabeler
from .models import EventDrivenEnsemble
from .portfolio import PortfolioConstructor, aggregate_signal
from .replay import snapshot_from_dict
from .settings import VNextSettings
from .storage import LocalResearchStore
from .taxonomy import event_type_bucket


def _spearman(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 3:
        return 0.0
    corr = a.rank().corr(b.rank(), method="pearson")
    if corr is None or pd.isna(corr):
        return 0.0
    return float(corr)


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


@dataclass(slots=True)
class WalkForwardSummary:
    num_rows: int
    num_windows: int
    rank_ic: float
    hit_rate: float
    top_bottom_spread: float
    turnover: float
    max_drawdown: float
    beta_adjusted_return: float
    calibrated_brier: float
    leakage_passed: bool
    message: str = ""
    event_type_scorecards: dict[str, dict[str, float]] = field(default_factory=dict)


class WalkForwardEvaluator:
    def __init__(self, store: LocalResearchStore | None = None, settings: VNextSettings | None = None):
        self.store = store or LocalResearchStore()
        self.settings = settings or VNextSettings.from_env()
        self.labeler = PointInTimeLabeler(store=self.store)
        self.portfolio = PortfolioConstructor()
        self.features = FeatureEngineer()

    def build_training_frame(self, refresh_labels: bool = False) -> pd.DataFrame:
        snapshots = self.store.read_table("company_snapshots")
        if snapshots.empty:
            return pd.DataFrame()

        features = self._feature_frame_from_archived_snapshots()
        if features.empty:
            features = self.store.read_table("feature_vectors")
        if features.empty:
            return pd.DataFrame()

        labels = self.store.read_table("labels")
        if refresh_labels or labels.empty:
            self.labeler.materialize_labels(snapshots=snapshots, catalysts=self.store.read_table("catalysts"))
            labels = self.store.read_table("labels")
        if labels.empty:
            return pd.DataFrame()
        frame = features.merge(labels, on=["ticker", "as_of"], how="left")
        if frame.empty:
            return frame
        frame["as_of"] = pd.to_datetime(frame["as_of"], errors="coerce", utc=True, format="mixed").dt.tz_convert(None)
        frame["evaluation_date"] = frame["as_of"].dt.normalize()
        frame = frame.sort_values("as_of").drop_duplicates(
            subset=["ticker", "entity_id", "evaluation_date"],
            keep="last",
        )
        if "meta_aggregate" in frame.columns:
            aggregate_mask = frame["meta_aggregate"].fillna(False).astype(bool)
            non_aggregate = frame[~aggregate_mask].copy()
            aggregate = frame[aggregate_mask].copy()
            if not non_aggregate.empty:
                keep_aggregate_keys = set(
                    aggregate[["ticker", "as_of"]].apply(tuple, axis=1).tolist()
                ) - set(non_aggregate[["ticker", "as_of"]].apply(tuple, axis=1).tolist())
                if keep_aggregate_keys:
                    aggregate = aggregate[
                        aggregate[["ticker", "as_of"]].apply(tuple, axis=1).isin(keep_aggregate_keys)
                    ].copy()
                else:
                    aggregate = aggregate.iloc[0:0].copy()
                frame = pd.concat([non_aggregate, aggregate], ignore_index=True)
        return frame

    def _feature_frame_from_archived_snapshots(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        seen: set[tuple[str, str, str]] = set()
        for path in self.store.list_raw_payload_paths("snapshots"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            try:
                snapshot = snapshot_from_dict(payload)
            except Exception:
                continue
            for vector in self.features.build_all(snapshot):
                dedupe_key = (vector.ticker, vector.entity_id, vector.as_of)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                rows.append(vector.to_row())
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def leakage_audit(self, frame: pd.DataFrame) -> bool:
        if frame.empty:
            return False
        forbidden_prefixes = ("target_",)
        feature_columns = [
            col
            for col in frame.columns
            if col not in {"entity_id", "ticker", "as_of", "thesis_horizon", "evaluation_date"}
            and not col.startswith("target_")
            and not col.startswith("meta_")
        ]
        if any(col.startswith(forbidden_prefixes) and col in feature_columns for col in feature_columns):
            return False
        if "market_flow_return_6mo_legacy" in feature_columns:
            return False
        return True

    def evaluate(self, min_train_rows: int = 24) -> WalkForwardSummary:
        frame = self.build_training_frame()
        if frame.empty:
            return WalkForwardSummary(
                num_rows=0,
                num_windows=0,
                rank_ic=0.0,
                hit_rate=0.0,
                top_bottom_spread=0.0,
                turnover=0.0,
                max_drawdown=0.0,
                beta_adjusted_return=0.0,
                calibrated_brier=1.0,
                leakage_passed=False,
                message="No archived snapshots with forward labels yet. Collect snapshots over time, then rerun.",
                event_type_scorecards={},
            )

        frame = frame.sort_values(["evaluation_date", "as_of"])
        unique_dates = self._rebalance_dates(sorted(frame["evaluation_date"].drop_duplicates().tolist()))
        if len(unique_dates) < 3:
            return WalkForwardSummary(
                num_rows=len(frame),
                num_windows=0,
                rank_ic=0.0,
                hit_rate=0.0,
                top_bottom_spread=0.0,
                turnover=0.0,
                max_drawdown=0.0,
                beta_adjusted_return=0.0,
                calibrated_brier=1.0,
                leakage_passed=self.leakage_audit(frame),
                message="Need at least three distinct as-of dates for walk-forward evaluation.",
                event_type_scorecards={},
            )

        ensemble = EventDrivenEnsemble(store=self.store)
        rank_ics: list[float] = []
        hit_rates: list[float] = []
        spreads: list[float] = []
        turnovers: list[float] = []
        briers: list[float] = []
        cumulative_returns: list[float] = []
        previous_top: set[str] = set()
        previous_recommendations: dict[str, object] = {}
        previous_signals: dict[str, SignalArtifact] = {}
        num_windows = 0
        event_type_rows: dict[str, list[dict[str, float]]] = {}

        for split_idx in range(2, len(unique_dates)):
            test_date = unique_dates[split_idx]
            train = frame[frame["evaluation_date"] < test_date].copy()
            test = self._latest_test_frame(frame, test_date)
            if len(train) < min_train_rows or test.empty:
                continue

            ensemble.fit(train, persist_artifact=False, register_experiment=False)
            predictions = ensemble.score(
                [
                    self._row_to_feature_vector(row)
                    for _, row in test.iterrows()
                ]
            )
            company_frame, current_top, current_recommendations, current_signals = self._company_test_frame(
                test=test,
                predictions=predictions,
                previous_recommendations=previous_recommendations,
                previous_signals=previous_signals,
                previous_top=previous_top,
            )
            if company_frame.empty:
                continue
            company_frame = company_frame.dropna(subset=["target_return_90d"])
            if company_frame.empty:
                continue
            if company_frame["ticker"].nunique() < self.settings.evaluation_min_names_per_window:
                continue

            num_windows += 1
            rank_ics.append(_spearman(company_frame["expected_return"], company_frame["target_return_90d"]))
            hit_rates.append(float((np.sign(company_frame["expected_return"]) == np.sign(company_frame["target_return_90d"])).mean()))
            top = company_frame.nlargest(max(1, len(company_frame) // 5), "expected_return")
            bottom = company_frame.nsmallest(max(1, len(company_frame) // 5), "expected_return")
            spreads.append(float(top["target_return_90d"].mean() - bottom["target_return_90d"].mean()))
            turnovers.append(1.0 if not previous_top else 1.0 - (len(current_top & previous_top) / max(len(current_top | previous_top), 1)))
            previous_top = current_top
            briers.append(float(((company_frame["catalyst_success_prob"] - company_frame["target_catalyst_success"]) ** 2).mean()))
            cumulative_returns.append(float(top["target_return_90d"].mean()))
            previous_recommendations = current_recommendations
            previous_signals = current_signals

            for event_type, group in company_frame.groupby("target_primary_event_type", dropna=False):
                event_name = "none" if pd.isna(event_type) or event_type is None else str(event_type)
                event_type_rows.setdefault(event_name, []).append(
                    {
                        "rank_ic": _spearman(group["expected_return"], group["target_return_90d"]),
                        "hit_rate": float((np.sign(group["expected_return"]) == np.sign(group["target_return_90d"])).mean()),
                        "beta_adjusted_return": float(group["target_return_90d"].mean()),
                        "calibrated_brier": float(((group["catalyst_success_prob"] - group["target_catalyst_success"]) ** 2).mean()),
                        "top_bottom_spread": float(
                            group.nlargest(max(1, len(group) // 2), "expected_return")["target_return_90d"].mean()
                            - group.nsmallest(max(1, len(group) // 2), "expected_return")["target_return_90d"].mean()
                        )
                        if len(group) > 1
                        else 0.0,
                        "rows": float(len(group)),
                    }
                )

        if not num_windows:
            return WalkForwardSummary(
                num_rows=len(frame),
                num_windows=0,
                rank_ic=0.0,
                hit_rate=0.0,
                top_bottom_spread=0.0,
                turnover=0.0,
                max_drawdown=0.0,
                beta_adjusted_return=0.0,
                calibrated_brier=1.0,
                leakage_passed=self.leakage_audit(frame),
                message="Not enough labeled windows to fit and test the ensemble yet.",
                event_type_scorecards={},
            )

        curve = pd.Series(cumulative_returns).cumsum()
        peak = curve.cummax()
        drawdown = (curve - peak).min()
        event_type_scorecards = {
            event_type: {
                "rows": float(sum(item["rows"] for item in rows)),
                "windows": float(len(rows)),
                "rank_ic": _safe_mean([item["rank_ic"] for item in rows]),
                "hit_rate": _safe_mean([item["hit_rate"] for item in rows]),
                "top_bottom_spread": _safe_mean([item["top_bottom_spread"] for item in rows]),
                "beta_adjusted_return": _safe_mean([item["beta_adjusted_return"] for item in rows]),
                "calibrated_brier": _safe_mean([item["calibrated_brier"] for item in rows]),
                "event_bucket": event_type_bucket(event_type),
            }
            for event_type, rows in event_type_rows.items()
        }
        return WalkForwardSummary(
            num_rows=len(frame),
            num_windows=num_windows,
            rank_ic=_safe_mean(rank_ics),
            hit_rate=_safe_mean(hit_rates),
            top_bottom_spread=_safe_mean(spreads),
            turnover=_safe_mean(turnovers),
            max_drawdown=float(abs(drawdown)),
            beta_adjusted_return=_safe_mean(cumulative_returns),
            calibrated_brier=_safe_mean(briers),
            leakage_passed=self.leakage_audit(frame),
            message="ok",
            event_type_scorecards=event_type_scorecards,
        )

    def _rebalance_dates(self, unique_dates: list[pd.Timestamp]) -> list[pd.Timestamp]:
        spacing_days = max(int(self.settings.evaluation_rebalance_spacing_days), 1)
        selected: list[pd.Timestamp] = []
        for date in unique_dates:
            if not selected:
                selected.append(date)
                continue
            if (date - selected[-1]).days >= spacing_days:
                selected.append(date)
        if unique_dates and selected[-1] != unique_dates[-1]:
            selected.append(unique_dates[-1])
        return selected

    def _latest_test_frame(self, frame: pd.DataFrame, test_date: pd.Timestamp) -> pd.DataFrame:
        eligible = frame[frame["evaluation_date"] <= test_date].copy()
        if eligible.empty:
            return eligible
        eligible["staleness_days"] = (test_date - eligible["evaluation_date"]).dt.days.astype(float)
        eligible = eligible[eligible["staleness_days"] <= self.settings.evaluation_max_snapshot_staleness_days]
        if eligible.empty:
            return eligible
        latest_as_of = eligible.groupby("ticker")["as_of"].transform("max")
        test = eligible[eligible["as_of"] == latest_as_of].copy()
        return test.sort_values(["ticker", "entity_id"])

    def _company_test_frame(
        self,
        test: pd.DataFrame,
        predictions: list[ModelPrediction],
        previous_recommendations: dict[str, object],
        previous_signals: dict[str, SignalArtifact],
        previous_top: set[str],
    ) -> tuple[pd.DataFrame, set[str], dict[str, object], dict[str, SignalArtifact]]:
        test = test.copy()
        test["as_of"] = pd.to_datetime(test["as_of"], errors="coerce", utc=True, format="mixed").dt.tz_convert(None)

        predictions_by_key: dict[tuple[str, pd.Timestamp], list[ModelPrediction]] = {}
        for pred in predictions:
            pred_ts = pd.to_datetime(pred.as_of, errors="coerce", utc=True, format="mixed")
            if pd.isna(pred_ts):
                continue
            pred_ts = pred_ts.tz_convert(None)
            key = (pred.ticker, pred_ts)
            predictions_by_key.setdefault(key, []).append(pred)

        records: list[dict[str, float | str | None]] = []
        current_recommendations: dict[str, object] = {}
        current_signals: dict[str, SignalArtifact] = {}
        for (ticker, as_of), group in test.groupby(["ticker", "as_of"], dropna=True):
            program_predictions = predictions_by_key.get((ticker, as_of))
            if not program_predictions:
                continue
            label = group.sort_values("entity_id").iloc[0]
            signal = aggregate_signal(
                ticker=ticker,
                as_of=as_of.isoformat(),
                predictions=program_predictions,
                evidence_rationale=[],
                evidence=[],
            )
            recommendation = self.portfolio.recommend(
                signal,
                previous_recommendation=previous_recommendations.get(ticker),
                previous_signal=previous_signals.get(ticker),
            )
            current_recommendations[ticker] = recommendation
            current_signals[ticker] = signal
            records.append(
                {
                    "ticker": ticker,
                    "as_of": as_of,
                    "expected_return": signal.expected_return,
                    "catalyst_success_prob": signal.catalyst_success_prob,
                    "confidence": signal.confidence,
                    "target_weight": recommendation.target_weight,
                    "scenario": recommendation.scenario,
                    "primary_event_type": signal.primary_event_type,
                    "target_return_90d": float(label["target_return_90d"]) if pd.notna(label["target_return_90d"]) else np.nan,
                    "target_catalyst_success": float(label["target_catalyst_success"]) if pd.notna(label["target_catalyst_success"]) else np.nan,
                    "target_primary_event_type": label.get("target_primary_event_type"),
                    "target_primary_event_bucket": label.get("target_primary_event_bucket"),
                }
            )

        company_frame = pd.DataFrame(records)
        if company_frame.empty:
            return company_frame, set(), current_recommendations, current_signals
        actionable = company_frame[company_frame["target_weight"] > 0].copy()
        rank_source = actionable if not actionable.empty else company_frame
        current_top = set(
            rank_source[
                rank_source["target_weight"] >= self.settings.evaluation_turnover_book_weight_floor
            ]["ticker"].tolist()
        )
        if not current_top:
            top_count = max(1, len(rank_source) // 5)
            current_top = set(rank_source.nlargest(top_count, "target_weight")["ticker"].tolist())
        sticky_holds = set(
            company_frame[
                company_frame["ticker"].isin(previous_top)
                & (company_frame["target_weight"] >= 0.5)
            ]["ticker"].tolist()
        )
        current_top |= sticky_holds
        return company_frame, current_top, current_recommendations, current_signals

    @staticmethod
    def _row_to_feature_vector(row: pd.Series):
        from .entities import FeatureVector

        features = {
            key: float(row[key])
            for key in row.index
            if key
            not in {
                "entity_id",
                "ticker",
                "as_of",
                "thesis_horizon",
                "evaluation_date",
            }
            and not key.startswith("target_")
            and not key.startswith("meta_")
        }
        metadata = {key[5:]: row[key] for key in row.index if key.startswith("meta_")}
        return FeatureVector(
            entity_id=row["entity_id"],
            ticker=row["ticker"],
            as_of=str(row["as_of"]),
            thesis_horizon=row["thesis_horizon"],
            feature_family=features,
            metadata=metadata,
        )
