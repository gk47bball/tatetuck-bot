from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .attribution import FACTOR_FAMILY_PREFIXES, ablate_feature_family, ablate_momentum
from .entities import ModelPrediction, SignalArtifact
from .features import FeatureEngineer
from .labels import PointInTimeLabeler
from .market_profile import build_expectation_lens, classify_company_state
from .models import EventDrivenEnsemble
from .portfolio import PortfolioConstructor, aggregate_signal
from .replay import snapshot_from_dict
from .settings import VNextSettings
from .storage import LocalResearchStore
from .taxonomy import event_timing_priority, event_type_bucket, event_type_priority, is_exact_timing_event, is_synthetic_event


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
    strict_rank_ic: float = 0.0
    strict_hit_rate: float = 0.0
    strict_top_bottom_spread: float = 0.0
    alpha_rank_ic: float = 0.0
    pm_context_coverage: float = 0.0
    exact_primary_event_rate: float = 0.0
    synthetic_primary_event_rate: float = 0.0
    stale_catalyst_rate: float = 0.0
    institutional_blockers: list[str] = field(default_factory=list)
    latest_window_top_trades: list[dict[str, object]] = field(default_factory=list)
    event_type_scorecards: dict[str, dict[str, float]] = field(default_factory=dict)
    company_state_scorecards: dict[str, dict[str, float]] = field(default_factory=dict)
    setup_type_scorecards: dict[str, dict[str, float]] = field(default_factory=dict)
    state_setup_scorecards: dict[str, dict[str, float]] = field(default_factory=dict)
    factor_attribution: dict[str, dict[str, float]] = field(default_factory=dict)
    momentum_ablation: dict[str, float] = field(default_factory=dict)


class WalkForwardEvaluator:
    def __init__(self, store: LocalResearchStore | None = None, settings: VNextSettings | None = None):
        self.store = store or LocalResearchStore()
        self.settings = settings or VNextSettings.from_env()
        self.labeler = PointInTimeLabeler(store=self.store)
        self.portfolio = PortfolioConstructor(store=self.store, use_validation_priors=False)
        self.features = FeatureEngineer()
        self._snapshot_cache: dict[tuple[str, str], object] = {}
        self._from_failure_universe_rate: float = 0.0

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

        # Augment with failure universe to correct survivorship bias
        try:
            from .failure_universe import load_failure_frame
            failure_frame = load_failure_frame()
            if not failure_frame.empty:
                frame = pd.concat([frame, failure_frame], ignore_index=True, sort=False)
                frame = frame.fillna(0.0)  # fill missing feature columns with 0
        except ImportError:
            pass

        self._from_failure_universe_rate = float(
            frame["meta_from_failure_universe"].fillna(False).mean()
        ) if "meta_from_failure_universe" in frame.columns else 0.0

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

    def evaluate(self, min_train_rows: int = 50) -> WalkForwardSummary:
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
        alpha_rank_ics: list[float] = []
        hit_rates: list[float] = []
        spreads: list[float] = []
        strict_rank_ics: list[float] = []
        strict_hit_rates: list[float] = []
        strict_spreads: list[float] = []
        turnovers: list[float] = []
        briers: list[float] = []
        cumulative_returns: list[float] = []
        exact_event_rates: list[float] = []
        synthetic_event_rates: list[float] = []
        stale_event_rates: list[float] = []
        pm_context_coverages: list[float] = []
        previous_top: set[str] = set()
        previous_recommendations: dict[str, object] = {}
        previous_signals: dict[str, SignalArtifact] = {}
        num_windows = 0
        event_type_rows: dict[str, list[dict[str, float]]] = {}
        company_state_rows: dict[str, list[dict[str, float]]] = {}
        setup_type_rows: dict[str, list[dict[str, float]]] = {}
        state_setup_rows: dict[str, list[dict[str, float]]] = {}
        factor_rows: dict[str, list[dict[str, float]]] = {}
        momentum_baseline_ics: list[float] = []
        momentum_only_ics: list[float] = []
        momentum_ablated_ics: list[float] = []
        signal_momentum_corrs: list[float] = []
        latest_window_top_trades: list[dict[str, object]] = []

        for split_idx in range(2, len(unique_dates)):
            test_date = unique_dates[split_idx]
            train = frame[frame["evaluation_date"] < test_date].copy()
            test = self._latest_test_frame(frame, test_date)
            if len(train) < min_train_rows or test.empty:
                continue

            ensemble.fit(train, persist_artifact=False, register_experiment=False)
            test_vectors = [self._row_to_feature_vector(row) for _, row in test.iterrows()]
            try:
                predictions = ensemble.score(test_vectors, persist=False)
            except TypeError:
                predictions = ensemble.score(test_vectors)
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
            baseline_rank_ic = _spearman(company_frame["expected_return"], company_frame["target_return_90d"])
            rank_ics.append(baseline_rank_ic)
            if (
                "target_alpha_90d" in company_frame.columns
                and company_frame["target_alpha_90d"].notna().sum() >= 3
            ):
                alpha_rank_ic_window = _spearman(company_frame["expected_return"], company_frame["target_alpha_90d"])
                alpha_rank_ics.append(alpha_rank_ic_window)
            hit_rates.append(float((np.sign(company_frame["expected_return"]) == np.sign(company_frame["target_return_90d"])).mean()))
            top = company_frame.nlargest(max(1, len(company_frame) // 5), "expected_return")
            bottom = company_frame.nsmallest(max(1, len(company_frame) // 5), "expected_return")
            baseline_spread = float(top["target_return_90d"].mean() - bottom["target_return_90d"].mean())
            spreads.append(baseline_spread)
            exact_event_rates.append(float(company_frame["primary_event_exact"].fillna(False).mean()))
            synthetic_event_rates.append(float(company_frame["primary_event_synthetic"].fillna(False).mean()))
            stale_event_rates.append(
                float(
                    (company_frame["primary_event_status"].fillna("") == "stale_synthetic").mean()
                )
                if "primary_event_status" in company_frame.columns
                else 0.0
            )
            pm_context_coverages.append(
                float(
                    company_frame[
                        ["company_state", "setup_type", "internal_upside_pct", "floor_support_pct"]
                    ].notna().all(axis=1).mean()
                )
            )
            strict_frame = company_frame[
                company_frame["primary_event_exact"].fillna(False)
                & ~company_frame["primary_event_synthetic"].fillna(False)
            ].copy()
            strict_frame = strict_frame.dropna(subset=["target_return_90d"])
            if len(strict_frame) >= max(2, self.settings.evaluation_min_names_per_window):
                strict_rank_ics.append(_spearman(strict_frame["expected_return"], strict_frame["target_return_90d"]))
                strict_hit_rates.append(
                    float((np.sign(strict_frame["expected_return"]) == np.sign(strict_frame["target_return_90d"])).mean())
                )
                strict_top = strict_frame.nlargest(max(1, len(strict_frame) // 5), "expected_return")
                strict_bottom = strict_frame.nsmallest(max(1, len(strict_frame) // 5), "expected_return")
                strict_spreads.append(float(strict_top["target_return_90d"].mean() - strict_bottom["target_return_90d"].mean()))
            turnovers.append(1.0 if not previous_top else 1.0 - (len(current_top & previous_top) / max(len(current_top | previous_top), 1)))
            briers.append(float(((company_frame["catalyst_success_prob"] - company_frame["target_catalyst_success"]) ** 2).mean()))
            cumulative_returns.append(float(top["target_return_90d"].mean()))

            momentum_frame = self._company_momentum_frame(test=test, company_frame=company_frame)
            momentum_only_ics.append(_spearman(momentum_frame["momentum_3mo"], momentum_frame["target_return_90d"]))
            momentum_baseline_ics.append(baseline_rank_ic)
            signal_momentum_corrs.append(_spearman(momentum_frame["momentum_3mo"], momentum_frame["expected_return"]))
            no_momentum_frame = self._ablated_company_frame(
                test=test,
                ensemble=ensemble,
                previous_recommendations=previous_recommendations,
                previous_signals=previous_signals,
                previous_top=previous_top,
                ablation_kind="momentum",
            )
            if not no_momentum_frame.empty:
                momentum_ablated_ics.append(
                    _spearman(no_momentum_frame["expected_return"], no_momentum_frame["target_return_90d"])
                )

            for family in FACTOR_FAMILY_PREFIXES:
                ablated_frame = self._ablated_company_frame(
                    test=test,
                    ensemble=ensemble,
                    previous_recommendations=previous_recommendations,
                    previous_signals=previous_signals,
                    previous_top=previous_top,
                    ablation_kind=family,
                )
                if ablated_frame.empty:
                    continue
                ablated_rank_ic = _spearman(ablated_frame["expected_return"], ablated_frame["target_return_90d"])
                ablated_top = ablated_frame.nlargest(max(1, len(ablated_frame) // 5), "expected_return")
                ablated_bottom = ablated_frame.nsmallest(max(1, len(ablated_frame) // 5), "expected_return")
                ablated_spread = float(ablated_top["target_return_90d"].mean() - ablated_bottom["target_return_90d"].mean())
                aligned = company_frame[["ticker", "expected_return"]].merge(
                    ablated_frame[["ticker", "expected_return"]],
                    on="ticker",
                    suffixes=("_base", "_ablated"),
                )
                factor_rows.setdefault(family, []).append(
                    {
                        "rank_ic_delta": float(baseline_rank_ic - ablated_rank_ic),
                        "spread_delta": float(baseline_spread - ablated_spread),
                        "signal_correlation": _spearman(
                            aligned["expected_return_base"],
                            aligned["expected_return_ablated"],
                        )
                        if not aligned.empty
                        else 0.0,
                    }
                )

            previous_top = current_top
            previous_recommendations = current_recommendations
            previous_signals = current_signals
            latest_window_top_trades = (
                company_frame.sort_values(["target_weight", "confidence", "expected_return"], ascending=False)
                .head(10)
                .to_dict(orient="records")
            )
            latest_window_top_trades = [
                {
                    key: (value.isoformat() if isinstance(value, pd.Timestamp) else value)
                    for key, value in row.items()
                }
                for row in latest_window_top_trades
            ]

            for event_type, group in company_frame.groupby("target_primary_event_type", dropna=False):
                event_name = "none" if pd.isna(event_type) or event_type is None else str(event_type)
                event_type_rows.setdefault(event_name, []).append(self._group_metrics(group))
            for company_state, group in company_frame.groupby("company_state", dropna=False):
                state_name = "unknown" if pd.isna(company_state) or company_state is None else str(company_state)
                company_state_rows.setdefault(state_name, []).append(self._group_metrics(group))
            for setup_type, group in company_frame.groupby("setup_type", dropna=False):
                setup_name = "unknown" if pd.isna(setup_type) or setup_type is None else str(setup_type)
                setup_type_rows.setdefault(setup_name, []).append(self._group_metrics(group))
            combo_frame = company_frame.copy()
            combo_frame["state_setup_key"] = combo_frame.apply(
                lambda row: f"{row.get('company_state') or 'unknown'}|{row.get('setup_type') or 'unknown'}",
                axis=1,
            )
            for combo_key, group in combo_frame.groupby("state_setup_key", dropna=False):
                combo_name = "unknown|unknown" if pd.isna(combo_key) or combo_key is None else str(combo_key)
                state_setup_rows.setdefault(combo_name, []).append(self._group_metrics(group))

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

        # Compound returns properly: arithmetic cumsum understates drawdown for
        # biotech where window returns routinely exceed ±20–30%.
        curve = (1.0 + pd.Series(cumulative_returns)).cumprod() - 1.0
        peak = curve.cummax()
        drawdown = (curve - peak).min()
        event_type_scorecards = {
            event_type: {
                "rows": float(sum(item["rows"] for item in rows)),
                "windows": float(len(rows)),
                "rank_ic": _safe_mean([item["rank_ic"] for item in rows]),
                "hit_rate": _safe_mean([item["hit_rate"] for item in rows]),
                "top_bottom_spread": _safe_mean([item["top_bottom_spread"] for item in rows]),
                "mean_return_90d": _safe_mean([item["mean_return_90d"] for item in rows]),
                "calibrated_brier": _safe_mean([item["calibrated_brier"] for item in rows]),
                "event_bucket": event_type_bucket(event_type),
            }
            for event_type, rows in event_type_rows.items()
        }
        company_state_scorecards = {
            company_state: self._aggregate_group_rows(rows)
            for company_state, rows in company_state_rows.items()
        }
        setup_type_scorecards = {
            setup_type: self._aggregate_group_rows(rows)
            for setup_type, rows in setup_type_rows.items()
        }
        state_setup_scorecards = {
            combo_key: self._aggregate_group_rows(rows)
            for combo_key, rows in state_setup_rows.items()
        }
        factor_attribution = {
            family: {
                "rank_ic_delta": _safe_mean([item["rank_ic_delta"] for item in rows]),
                "spread_delta": _safe_mean([item["spread_delta"] for item in rows]),
                "signal_correlation": _safe_mean([item["signal_correlation"] for item in rows]),
                "windows": float(len(rows)),
            }
            for family, rows in factor_rows.items()
        }
        momentum_ablation = {
            "baseline_rank_ic": _safe_mean(momentum_baseline_ics),
            "momentum_only_rank_ic": _safe_mean(momentum_only_ics),
            "no_momentum_rank_ic": _safe_mean(momentum_ablated_ics),
            "signal_momentum_correlation": _safe_mean(signal_momentum_corrs),
        }
        institutional_blockers: list[str] = []
        pm_context_coverage = _safe_mean(pm_context_coverages)
        exact_primary_event_rate = _safe_mean(exact_event_rates)
        synthetic_primary_event_rate = _safe_mean(synthetic_event_rates)
        stale_catalyst_rate = _safe_mean(stale_event_rates)
        if pm_context_coverage < 0.95:
            institutional_blockers.append(
                f"Only {pm_context_coverage * 100:.1f}% of evaluated rows carry full PM context fields."
            )
        if exact_primary_event_rate < 0.60:
            institutional_blockers.append(
                f"Only {exact_primary_event_rate * 100:.1f}% of evaluated rows have exact primary event timing."
            )
        if synthetic_primary_event_rate > 0.25:
            institutional_blockers.append(
                f"{synthetic_primary_event_rate * 100:.1f}% of evaluated rows still rely on synthetic primary events."
            )
        if stale_catalyst_rate > 0.10:
            institutional_blockers.append(
                f"{stale_catalyst_rate * 100:.1f}% of evaluated rows have a stale synthetic primary event "
                "(CT.gov not updated after trial readout); these phantom catalysts corrupt near-term signal."
            )
        if momentum_ablation["no_momentum_rank_ic"] > 0.0 and (
            momentum_ablation["baseline_rank_ic"] - momentum_ablation["no_momentum_rank_ic"]
        ) > 0.12:
            institutional_blockers.append(
                "Momentum ablation removes too much of the rank IC; factor mix still looks too tape-dependent."
            )
        universe_membership = self.store.read_table("universe_membership")
        if universe_membership.empty:
            institutional_blockers.append("Historical universe membership is missing, so survivorship bias is not controlled.")
        elif "is_delisted" not in universe_membership.columns or not universe_membership["is_delisted"].fillna(False).astype(bool).any():
            institutional_blockers.append("Historical universe membership still has no delisted names, so survivorship bias is only partially controlled.")
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
            strict_rank_ic=_safe_mean(strict_rank_ics),
            strict_hit_rate=_safe_mean(strict_hit_rates),
            strict_top_bottom_spread=_safe_mean(strict_spreads),
            alpha_rank_ic=_safe_mean(alpha_rank_ics),
            pm_context_coverage=pm_context_coverage,
            exact_primary_event_rate=exact_primary_event_rate,
            synthetic_primary_event_rate=synthetic_primary_event_rate,
            stale_catalyst_rate=stale_catalyst_rate,
            institutional_blockers=institutional_blockers,
            latest_window_top_trades=latest_window_top_trades,
            event_type_scorecards=event_type_scorecards,
            company_state_scorecards=company_state_scorecards,
            setup_type_scorecards=setup_type_scorecards,
            state_setup_scorecards=state_setup_scorecards,
            factor_attribution=factor_attribution,
            momentum_ablation=momentum_ablation,
        )

    @staticmethod
    def _group_metrics(group: pd.DataFrame) -> dict[str, float]:
        return {
            "rank_ic": _spearman(group["expected_return"], group["target_return_90d"]),
            "hit_rate": float((np.sign(group["expected_return"]) == np.sign(group["target_return_90d"])).mean()),
            "mean_return_90d": float(group["target_return_90d"].mean()),
            "calibrated_brier": float(((group["catalyst_success_prob"] - group["target_catalyst_success"]) ** 2).mean()),
            "top_bottom_spread": float(
                group.nlargest(max(1, len(group) // 2), "expected_return")["target_return_90d"].mean()
                - group.nsmallest(max(1, len(group) // 2), "expected_return")["target_return_90d"].mean()
            )
            if len(group) > 1
            else 0.0,
            "rows": float(len(group)),
        }

    @staticmethod
    def _aggregate_group_rows(rows: list[dict[str, float]]) -> dict[str, float]:
        return {
            "rows": float(sum(item["rows"] for item in rows)),
            "windows": float(len(rows)),
            "rank_ic": _safe_mean([item["rank_ic"] for item in rows]),
            "hit_rate": _safe_mean([item["hit_rate"] for item in rows]),
            "top_bottom_spread": _safe_mean([item["top_bottom_spread"] for item in rows]),
            "mean_return_90d": _safe_mean([item["mean_return_90d"] for item in rows]),
            "calibrated_brier": _safe_mean([item["calibrated_brier"] for item in rows]),
        }

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

    def _ablated_company_frame(
        self,
        test: pd.DataFrame,
        ensemble: EventDrivenEnsemble,
        previous_recommendations: dict[str, object],
        previous_signals: dict[str, SignalArtifact],
        previous_top: set[str],
        ablation_kind: str,
    ) -> pd.DataFrame:
        vectors = []
        for _, row in test.iterrows():
            vector = self._row_to_feature_vector(row)
            if ablation_kind == "momentum":
                vector.feature_family = ablate_momentum(vector.feature_family)
            else:
                vector.feature_family = ablate_feature_family(vector.feature_family, ablation_kind)
            vectors.append(vector)
        try:
            predictions = ensemble.score(vectors, persist=False)
        except TypeError:
            predictions = ensemble.score(vectors)
        company_frame, _, _, _ = self._company_test_frame(
            test=test,
            predictions=predictions,
            previous_recommendations=previous_recommendations,
            previous_signals=previous_signals,
            previous_top=previous_top,
        )
        if company_frame.empty:
            return company_frame
        return company_frame.dropna(subset=["target_return_90d"])

    @staticmethod
    def _company_momentum_frame(test: pd.DataFrame, company_frame: pd.DataFrame) -> pd.DataFrame:
        if "market_flow_momentum_3mo" not in test.columns:
            merged = company_frame.copy()
            merged["momentum_3mo"] = 0.0
            return merged
        company_momentum = (
            test.groupby("ticker", dropna=True)["market_flow_momentum_3mo"]
            .mean()
            .reset_index(name="momentum_3mo")
        )
        merged = company_frame.merge(company_momentum, on="ticker", how="left")
        merged["momentum_3mo"] = pd.to_numeric(merged["momentum_3mo"], errors="coerce").fillna(0.0)
        return merged

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
            snapshot = self._load_snapshot_at(ticker, as_of)
            signal = aggregate_signal(
                ticker=ticker,
                as_of=as_of.isoformat(),
                predictions=program_predictions,
                evidence_rationale=[],
                evidence=[],
            )
            primary_event = None
            if snapshot is not None:
                signal, primary_event = self._enrich_signal_context(snapshot, signal, as_of)
            recommendation = self.portfolio.recommend(
                signal,
                previous_recommendation=previous_recommendations.get(ticker),
                previous_signal=previous_signals.get(ticker),
            )
            current_recommendations[ticker] = recommendation
            current_signals[ticker] = signal
            primary_event_quality = self._primary_event_quality(primary_event)
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
                    "company_state": signal.company_state,
                    "setup_type": signal.setup_type,
                    "internal_upside_pct": signal.internal_upside_pct,
                    "floor_support_pct": signal.floor_support_pct,
                    "primary_event_title": None if primary_event is None else primary_event.title,
                    "primary_event_date": None if primary_event is None else primary_event.expected_date,
                    "primary_event_status": None if primary_event is None else primary_event.status,
                    "primary_event_exact": primary_event_quality["exact"],
                    "primary_event_synthetic": primary_event_quality["synthetic"],
                    "target_return_90d": float(label["target_return_90d"]) if pd.notna(label["target_return_90d"]) else np.nan,
                    "target_alpha_90d": (
                        float(label["target_alpha_90d"])
                        if "target_alpha_90d" in label.index and pd.notna(label["target_alpha_90d"])
                        else np.nan
                    ),
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

    def _load_snapshot_at(self, ticker: str, as_of: pd.Timestamp):
        key = (ticker, as_of.isoformat())
        if key in self._snapshot_cache:
            return self._snapshot_cache[key]
        for path in self.store.list_raw_payload_paths("snapshots", f"{ticker}_"):
            snapshot = self._load_snapshot_payload(path)
            if snapshot is None:
                continue
            snapshot_ts = pd.to_datetime(snapshot.as_of, errors="coerce", utc=True, format="mixed")
            if pd.isna(snapshot_ts):
                continue
            snapshot_ts = snapshot_ts.tz_convert(None)
            if snapshot_ts == as_of:
                self._snapshot_cache[key] = snapshot
                return snapshot
        self._snapshot_cache[key] = None
        return None

    @staticmethod
    def _load_snapshot_payload(path: Path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        try:
            return snapshot_from_dict(payload)
        except Exception:
            return None

    def _enrich_signal_context(self, snapshot, signal: SignalArtifact, as_of: pd.Timestamp) -> tuple[SignalArtifact, object]:
        company_state = classify_company_state(snapshot)
        signal.company_state = company_state
        primary_event = self._primary_event(snapshot, preferred_event_type=signal.primary_event_type)
        peer_context = self._historical_peer_context(snapshot, as_of)
        expectation_lens = build_expectation_lens(snapshot, signal, primary_event, peer_context)
        signal.setup_type = str(expectation_lens["setup_type"])
        signal.internal_value = float(expectation_lens["internal_value"])
        signal.internal_price_target = (
            None if expectation_lens["internal_price_target"] is None else float(expectation_lens["internal_price_target"])
        )
        signal.internal_upside_pct = float(expectation_lens["internal_upside_pct"])
        signal.floor_support_pct = float(expectation_lens["floor_support_pct"])
        return signal, primary_event

    @staticmethod
    def _primary_event(snapshot, preferred_event_type: str | None = None):
        if not snapshot.catalyst_events:
            return None
        candidates = snapshot.catalyst_events
        best_overall = min(
            candidates,
            key=lambda item: (
                -event_timing_priority(item.status, item.expected_date, item.title),
                -event_type_priority(item.event_type),
                item.horizon_days,
                -item.importance,
            ),
        )
        if preferred_event_type:
            typed_candidates = [item for item in snapshot.catalyst_events if item.event_type == preferred_event_type]
            if typed_candidates:
                best_typed = min(
                    typed_candidates,
                    key=lambda item: (
                        -event_timing_priority(item.status, item.expected_date, item.title),
                        -event_type_priority(item.event_type),
                        item.horizon_days,
                        -item.importance,
                    ),
                )
                if event_timing_priority(best_typed.status, best_typed.expected_date, best_typed.title) >= event_timing_priority(
                    best_overall.status,
                    best_overall.expected_date,
                    best_overall.title,
                ):
                    return best_typed
        return best_overall

    def _historical_peer_context(self, snapshot, as_of: pd.Timestamp) -> dict[str, object]:
        companies = self.store.read_table("company_snapshots")
        programs = self.store.read_table("programs")
        if companies.empty or programs.empty:
            return {
                "summary": "Peer context unavailable until more archived snapshots are loaded.",
                "peer_tickers": [],
                "peer_stage": "unknown",
                "valuation_posture": "unknown",
                "current_multiple": None,
                "median_multiple": None,
                "metric_label": "value",
            }
        companies = companies.copy()
        programs = programs.copy()
        companies["as_of_ts"] = pd.to_datetime(companies["as_of"], errors="coerce", utc=True, format="mixed").dt.tz_convert(None)
        programs["as_of_ts"] = pd.to_datetime(programs["as_of"], errors="coerce", utc=True, format="mixed").dt.tz_convert(None)
        companies = companies[companies["as_of_ts"] <= as_of].sort_values("as_of_ts")
        programs = programs[programs["as_of_ts"] <= as_of].sort_values("as_of_ts")
        if companies.empty or programs.empty:
            return {
                "summary": "Peer context unavailable until more archived snapshots are loaded.",
                "peer_tickers": [],
                "peer_stage": "unknown",
                "valuation_posture": "unknown",
                "current_multiple": None,
                "median_multiple": None,
                "metric_label": "value",
            }
        latest_companies = companies.drop_duplicates(subset=["ticker"], keep="last").copy()
        latest_programs = programs.drop_duplicates(subset=["ticker", "program_id"], keep="last").copy()
        phase_rank_map = {
            "EARLY_PHASE1": 1,
            "PHASE1": 2,
            "PHASE2": 3,
            "PHASE3": 4,
            "NDA_BLA": 5,
            "APPROVED": 6,
        }
        latest_programs["phase_rank"] = latest_programs["phase"].map(phase_rank_map).fillna(0.0)
        top_programs = (
            latest_programs.sort_values(["ticker", "phase_rank", "tam_estimate"], ascending=[True, False, False])
            .drop_duplicates(subset=["ticker"], keep="first")
            [["ticker", "phase", "phase_rank", "tam_estimate", "name"]]
        )
        latest_companies = latest_companies.merge(top_programs, on="ticker", how="left")
        latest_companies = latest_companies[latest_companies["ticker"] != snapshot.ticker].copy()
        if latest_companies.empty:
            return {
                "summary": "Peer context unavailable until more archived snapshots are loaded.",
                "peer_tickers": [],
                "peer_stage": "unknown",
                "valuation_posture": "unknown",
                "current_multiple": None,
                "median_multiple": None,
                "metric_label": "value",
            }
        company_state = classify_company_state(snapshot)
        stage = (
            "commercial"
            if company_state in {"commercial_launch", "commercialized"}
            else "late_stage"
            if any(phase_rank_map.get(program.phase, 0) >= 4 for program in snapshot.programs)
            else "clinical"
        )
        if stage == "commercial":
            peers = latest_companies[latest_companies["revenue"].fillna(0.0) > 10_000_000].copy()
            current_multiple = snapshot.enterprise_value / max(snapshot.revenue, 1.0)
            if not peers.empty:
                peers["comparison_metric"] = peers["enterprise_value"].fillna(0.0) / peers["revenue"].clip(lower=1.0)
            metric_label = "EV/revenue"
        else:
            peers = latest_companies[latest_companies["revenue"].fillna(0.0) <= 10_000_000].copy()
            if stage == "late_stage":
                peers = peers[peers["phase_rank"].fillna(0.0) >= 4]
            else:
                peers = peers[peers["phase_rank"].fillna(0.0) < 4]
            top_tam = max((program.tam_estimate for program in snapshot.programs), default=0.0)
            current_multiple = snapshot.market_cap / max(top_tam, 1.0)
            if not peers.empty:
                peers["comparison_metric"] = peers["market_cap"].fillna(0.0) / peers["tam_estimate"].clip(lower=1.0)
            metric_label = "market-cap/TAM"
        if "comparison_metric" not in peers.columns:
            return {
                "summary": "Peer context is still sparse for this stage bucket.",
                "peer_tickers": [],
                "peer_stage": stage,
                "valuation_posture": "unknown",
                "current_multiple": current_multiple,
                "median_multiple": None,
                "metric_label": metric_label,
            }
        peers = peers.replace([pd.NA, float("inf"), float("-inf")], pd.NA).dropna(subset=["comparison_metric"])
        if peers.empty:
            return {
                "summary": "Peer context is still sparse for this stage bucket.",
                "peer_tickers": [],
                "peer_stage": stage,
                "valuation_posture": "unknown",
                "current_multiple": current_multiple,
                "median_multiple": None,
                "metric_label": metric_label,
            }
        median_multiple = float(peers["comparison_metric"].median())
        percentile = float((peers["comparison_metric"] <= current_multiple).mean() * 100.0)
        peers["distance"] = (peers["comparison_metric"] - current_multiple).abs()
        peer_tickers = peers.nsmallest(3, "distance")["ticker"].astype(str).tolist()
        valuation_posture = (
            "rich"
            if current_multiple > (median_multiple * 1.15)
            else "discounted"
            if current_multiple < (median_multiple * 0.85)
            else "near peer median"
        )
        summary = (
            f"{stage.replace('_', ' ')} peers trade around {median_multiple:.2f}x {metric_label}; "
            f"{snapshot.ticker} screens at {current_multiple:.2f}x ({valuation_posture}, {percentile:.0f}th percentile)."
        )
        return {
            "summary": summary,
            "peer_tickers": peer_tickers,
            "peer_stage": stage,
            "valuation_posture": valuation_posture,
            "current_multiple": current_multiple,
            "median_multiple": median_multiple,
            "metric_label": metric_label,
        }

    @staticmethod
    def _primary_event_quality(primary_event) -> dict[str, bool]:
        if primary_event is None:
            return {"exact": False, "synthetic": True}
        synthetic = is_synthetic_event(primary_event.status, primary_event.title)
        exact = is_exact_timing_event(primary_event.status, primary_event.expected_date, primary_event.title)
        return {"exact": exact, "synthetic": synthetic}

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
