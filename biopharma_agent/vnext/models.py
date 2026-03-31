from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .entities import ExperimentRecord, FeatureVector, ModelPrediction
from .storage import LocalResearchStore


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_term = math.exp(-min(value, 60.0))
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(min(-value, 60.0))
    return 1.0 / (1.0 + exp_term)


def _feature_value(features: dict[str, float] | pd.Series, key: str, default: float) -> float:
    value = features.get(key, default)
    if value is None:
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(value):
        return default
    return value


@dataclass(slots=True)
class EnsembleBundle:
    regressor_weights: np.ndarray | None = None
    regressor_bias: float = 0.0
    classifier_weights: np.ndarray | None = None
    classifier_bias: float = 0.0
    feature_mean: np.ndarray | None = None
    feature_scale: np.ndarray | None = None
    calibrator: IsotonicRegression | None = None
    feature_columns: list[str] | None = None


class EventDrivenEnsemble:
    def __init__(self, store: LocalResearchStore | None = None, model_name: str = "event_driven_ensemble", model_version: str = "v3"):
        self.store = store or LocalResearchStore()
        self.model_name = model_name
        self.model_version = model_version
        self.bundle = self._load()

    def _load(self) -> EnsembleBundle:
        path = self.store.model_path(self.model_name, self.model_version)
        if not path.exists():
            return EnsembleBundle()
        try:
            with open(path, "rb") as f:
                bundle = pickle.load(f)
        except (OSError, EOFError, pickle.PickleError):
            return EnsembleBundle()
        if not isinstance(bundle, EnsembleBundle):
            return EnsembleBundle()
        if not hasattr(bundle, "calibrator"):
            bundle.calibrator = None
        if not hasattr(bundle, "feature_columns"):
            bundle.feature_columns = None
        if not all(
            hasattr(bundle, attr)
            for attr in (
                "regressor_weights",
                "regressor_bias",
                "classifier_weights",
                "classifier_bias",
                "feature_mean",
                "feature_scale",
            )
        ):
            return EnsembleBundle()
        return bundle

    def _save(self) -> Path:
        path = self.store.model_path(self.model_name, self.model_version)
        with open(path, "wb") as f:
            pickle.dump(self.bundle, f)
        return path

    @staticmethod
    def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        feature_mean = np.mean(X, axis=0)
        feature_scale = np.std(X, axis=0)
        feature_scale = np.where(feature_scale < 1e-6, 1.0, feature_scale)
        normalized = (X - feature_mean) / feature_scale
        augmented = np.column_stack([normalized, np.ones(len(normalized))])
        penalty = np.eye(augmented.shape[1], dtype=float) * alpha
        penalty[-1, -1] = 0.0
        weights = np.linalg.pinv(augmented.T @ augmented + penalty) @ augmented.T @ y
        return feature_mean, feature_scale, weights[:-1], float(weights[-1])

    @staticmethod
    def _linear_predict(
        X: np.ndarray,
        feature_mean: np.ndarray | None,
        feature_scale: np.ndarray | None,
        weights: np.ndarray | None,
        bias: float,
    ) -> np.ndarray | None:
        if feature_mean is None or feature_scale is None or weights is None:
            return None
        normalized = (X - feature_mean) / np.where(feature_scale < 1e-6, 1.0, feature_scale)
        return normalized @ weights + bias

    def feature_columns(self, frame: pd.DataFrame) -> list[str]:
        return [
            column
            for column in frame.columns
            if column
            not in {
                "entity_id",
                "ticker",
                "as_of",
                "thesis_horizon",
                "target_return_30d",
                "target_return_90d",
                "target_return_180d",
                "target_alpha_30d",
                "target_alpha_90d",
                "target_alpha_180d",
                "target_catalyst_success",
            }
            and not column.startswith("target_")
            and not column.startswith("meta_")
            and column != "evaluation_date"
        ]

    def fit(
        self,
        frame: pd.DataFrame,
        persist_artifact: bool = True,
        register_experiment: bool = True,
    ) -> ExperimentRecord | None:
        required = {"target_return_90d", "target_catalyst_success"}
        if frame.empty or not required.issubset(frame.columns):
            return None
        usable = frame.dropna(subset=["target_return_90d", "target_catalyst_success"])
        if len(usable) < 30:
            return None

        # Prefer benchmark-relative alpha as the regression target when it is
        # present with at least 80% non-null coverage.  Training on alpha rather
        # than raw returns makes the signal cycle-stable: it captures genuine
        # stock-picking skill rather than riding broad XBI moves, so the model
        # generalises across bull and bear biotech regimes.
        used_alpha_target = False
        if (
            "target_alpha_90d" in usable.columns
            and usable["target_alpha_90d"].notna().sum() >= 0.80 * len(usable)
        ):
            usable = usable.dropna(subset=["target_alpha_90d"])
            regression_target_col = "target_alpha_90d"
            used_alpha_target = True
        else:
            regression_target_col = "target_return_90d"

        feature_columns = self.feature_columns(usable)
        X = usable[feature_columns].fillna(0.0).to_numpy(dtype=float)
        y_reg = usable[regression_target_col].to_numpy(dtype=float)
        y_clf = usable["target_catalyst_success"].to_numpy(dtype=int)

        # Volatility-normalise the regression target so that a 20% expected return
        # on a 150-vol name does not receive the same weight as 20% on a 50-vol name.
        # The fitted signal then ranks on IR-per-unit-vol, which is more honest for
        # cross-sectional ranking in biotech.  Fall back to raw y_reg if
        # market_flow_volatility is not present in the feature columns.
        if "market_flow_volatility" in feature_columns:
            vol_idx = feature_columns.index("market_flow_volatility")
            vol_col = X[:, vol_idx]
            y_reg_adj = y_reg / np.where(vol_col > 0.01, vol_col, 1.0)
        else:
            y_reg_adj = y_reg

        reg_mean, reg_scale, reg_weights, reg_bias = self._ridge_fit(X, y_reg_adj, alpha=1.5)
        clf_targets = np.where(y_clf > 0, 2.0, -2.0)
        clf_mean, clf_scale, clf_weights, clf_bias = self._ridge_fit(X, clf_targets, alpha=2.0)

        self.bundle = EnsembleBundle(
            regressor_weights=reg_weights,
            regressor_bias=reg_bias,
            classifier_weights=clf_weights,
            classifier_bias=clf_bias,
            feature_mean=reg_mean,
            feature_scale=reg_scale,
            calibrator=None,
            feature_columns=feature_columns,
        )
        clf_linear = self._linear_predict(X, clf_mean, clf_scale, clf_weights, clf_bias)
        if clf_linear is None:
            clf_scores = np.zeros(len(X), dtype=float)
        else:
            clf_scores = np.asarray([_sigmoid(value) for value in clf_linear], dtype=float)
        rule_probs = np.asarray([self._rule_success_prob_from_row(row) for _, row in usable.iterrows()], dtype=float)
        blended_probs = (0.40 * rule_probs) + (0.60 * clf_scores)
        finite_mask = np.isfinite(blended_probs) & np.isfinite(y_clf)
        if (
            finite_mask.any()
            and len(np.unique(y_clf[finite_mask])) > 1
            and len(np.unique(np.round(blended_probs[finite_mask], 5))) > 1
        ):
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(blended_probs[finite_mask], y_clf[finite_mask])
            self.bundle.calibrator = calibrator

        artifact_path: Path | None = None
        artifact_error = ""
        if persist_artifact:
            try:
                artifact_path = self._save()
            except OSError as exc:
                artifact_error = f"{type(exc).__name__}: {exc}"

        record = ExperimentRecord(
            experiment_id=f"{self.model_name}_{self.model_version}_{len(usable)}",
            created_at=pd.Timestamp.now(tz="UTC").isoformat(),
            model_name=self.model_name,
            model_version=self.model_version,
            train_window_start=str(usable["as_of"].min()),
            train_window_end=str(usable["as_of"].max()),
            holdout_window_start=None,
            holdout_window_end=None,
            metrics={
                "num_rows": float(len(usable)),
                "mean_target_return_90d": float(np.mean(usable["target_return_90d"].to_numpy(dtype=float))),
                "mean_regression_target": float(np.mean(y_reg)),
                "base_event_rate": float(np.mean(y_clf)),
                "used_alpha_target": used_alpha_target,
            },
            artifact_path=str(artifact_path) if artifact_path is not None else None,
        )
        if register_experiment:
            self.store.register_experiment(record)
        if artifact_error:
            self.store.write_pipeline_run(
                {
                    "job_name": "model_artifact_persist",
                    "status": "warning",
                    "started_at": record.created_at,
                    "finished_at": record.created_at,
                    "duration_seconds": 0.0,
                    "metrics": {
                        "model_name": self.model_name,
                        "model_version": self.model_version,
                        "num_rows": float(len(usable)),
                    },
                    "config": {"persist_artifact": persist_artifact},
                    "notes": artifact_error,
                }
            )
        return record

    def score(self, feature_vectors: Iterable[FeatureVector], persist: bool = True) -> list[ModelPrediction]:
        feature_vectors = list(feature_vectors)
        if not feature_vectors:
            return []
        frame = pd.DataFrame([vector.to_row() for vector in feature_vectors])
        return self._predict_frame(frame, feature_vectors, persist=persist)

    def _predict_frame(self, frame: pd.DataFrame, feature_vectors: list[FeatureVector], persist: bool = True) -> list[ModelPrediction]:
        predictions: list[ModelPrediction] = []
        feature_columns = self.bundle.feature_columns or self.feature_columns(frame)
        X = frame.reindex(columns=feature_columns, fill_value=0.0).fillna(0.0).to_numpy(dtype=float)

        reg_scores = self._linear_predict(
            X,
            self.bundle.feature_mean,
            self.bundle.feature_scale,
            self.bundle.regressor_weights,
            self.bundle.regressor_bias,
        )
        clf_linear = self._linear_predict(
            X,
            self.bundle.feature_mean,
            self.bundle.feature_scale,
            self.bundle.classifier_weights,
            self.bundle.classifier_bias,
        )
        clf_scores = None if clf_linear is None else np.asarray([_sigmoid(value) for value in clf_linear], dtype=float)

        for index, vector in enumerate(feature_vectors):
            rule_expected_return, rule_success_prob = self._rule_score(vector)
            expected_return = rule_expected_return
            catalyst_success_prob = rule_success_prob
            if reg_scores is not None and clf_scores is not None:
                expected_return = (0.40 * rule_expected_return) + (0.60 * float(reg_scores[index]))
                catalyst_success_prob = (0.40 * rule_success_prob) + (0.60 * float(clf_scores[index]))
                if self.bundle.calibrator is not None:
                    catalyst_success_prob = float(self.bundle.calibrator.predict([catalyst_success_prob])[0])

            # Crowding risk is primarily the analyst/investor crowdedness score.
            # Raw positive momentum only signals crowding when it is already elevated
            # (e.g. >15% 3-month run into a catalyst) — modest momentum driven by
            # fundamentals should not be penalised as crowding.
            momentum_3mo = vector.feature_family.get("market_flow_momentum_3mo", 0.0)
            momentum_crowding_signal = max(momentum_3mo - 0.15, 0.0) * 0.5
            crowding_risk = _sigmoid(
                vector.feature_family.get("catalyst_timing_crowdedness", 0.30) * 2.4
                + momentum_crowding_signal
            )
            financing_risk = _sigmoid(
                (0.9 * vector.feature_family.get("balance_sheet_financing_pressure", 0.0))
                - (0.25 * vector.feature_family.get("balance_sheet_runway_score", 0.0))
                + (0.35 * vector.feature_family.get("balance_sheet_debt_to_cap", 0.0))
            )
            confidence = _sigmoid(
                (1.2 * vector.feature_family.get("program_quality_pos_prior", 0.20))
                + (1.0 * vector.feature_family.get("catalyst_timing_probability", 0.30))
                + (0.8 * vector.feature_family.get("catalyst_timing_clinical_focus", 0.0))
                + (0.4 * vector.feature_family.get("catalyst_timing_expected_value", 0.0))
                + (0.4 * vector.feature_family.get("program_quality_trial_count", 0.0))
                - (0.35 * vector.feature_family.get("catalyst_timing_company_event_earnings", 0.0))
                - (0.5 * financing_risk)
            )
            predictions.append(
                ModelPrediction(
                    entity_id=vector.entity_id,
                    ticker=vector.ticker,
                    as_of=vector.as_of,
                    expected_return=float(expected_return),
                    catalyst_success_prob=float(max(0.01, min(0.99, catalyst_success_prob))),
                    confidence=float(max(0.01, min(0.99, confidence))),
                    crowding_risk=float(max(0.01, min(0.99, crowding_risk))),
                    financing_risk=float(max(0.01, min(0.99, financing_risk))),
                    thesis_horizon=vector.thesis_horizon,
                    model_name=self.model_name,
                    model_version=self.model_version,
                    metadata=vector.metadata.copy(),
                )
            )
        if persist:
            self.store.write_predictions(predictions)
        return predictions

    def _rule_score(self, vector: FeatureVector) -> tuple[float, float]:
        features = vector.feature_family
        return self._rule_score_from_features(features)

    @classmethod
    def _rule_success_prob_from_row(cls, row: pd.Series) -> float:
        _, success_prob = cls._rule_score_from_features(row.to_dict())
        return success_prob

    @staticmethod
    def _rule_score_from_features(features: dict[str, float] | pd.Series) -> tuple[float, float]:
        pre_commercial = _feature_value(features, "state_profile_pre_commercial", 0.0)
        commercial_launch = _feature_value(features, "state_profile_commercial_launch", 0.0)
        commercialized = _feature_value(features, "state_profile_commercialized", 0.0)
        competition_intensity = _feature_value(features, "state_profile_competition_intensity", 0.55)
        floor_support = _feature_value(features, "state_profile_floor_support_pct", 0.0)
        launch_progress = _feature_value(features, "state_profile_launch_progress_pct", _feature_value(features, "commercial_execution_launch_progress_pct", 0.0))
        lifecycle_score = _feature_value(features, "state_profile_lifecycle_management_score", _feature_value(features, "commercial_execution_lifecycle_management_score", 0.0))
        pipeline_optionality = _feature_value(features, "state_profile_pipeline_optionality_score", 0.0)
        capital_deployment = _feature_value(features, "state_profile_capital_deployment_score", _feature_value(features, "balance_sheet_capital_deployment_score", 0.0))
        hard_catalyst_presence = _feature_value(features, "state_profile_hard_catalyst_presence", 0.0)
        precommercial_value_gap = _feature_value(features, "state_profile_precommercial_value_gap", 0.0)

        expected_return = (
            0.18 * _feature_value(features, "program_quality_pos_prior", 0.15)
            + 0.12 * _feature_value(features, "program_quality_phase_score", 0.20)
            + 0.10 * _feature_value(features, "program_quality_endpoint_score", 0.25)
            + 0.08 * _feature_value(features, "program_quality_tam_to_cap", 0.0)
            + 0.15 * _feature_value(features, "catalyst_timing_expected_value", 0.10)
            + 0.06 * _feature_value(features, "catalyst_timing_event_phase3_readout", 0.0)
            + 0.04 * _feature_value(features, "catalyst_timing_event_phase2_readout", 0.0)
            + 0.05 * _feature_value(features, "catalyst_timing_event_pdufa", 0.0)
            + 0.04 * _feature_value(features, "commercial_execution_revenue_to_cap", -1.0)
            + 0.07 * _feature_value(features, "balance_sheet_cash_to_cap", 0.0)
            + 0.03 * _feature_value(features, "market_flow_momentum_3mo", 0.0)
            - 0.09 * _feature_value(features, "program_quality_modality_risk", 0.50)
            - 0.08 * _feature_value(features, "catalyst_timing_crowdedness", 0.30)
            - 0.07 * _feature_value(features, "balance_sheet_financing_pressure", 0.0)
            - 0.04 * _feature_value(features, "market_flow_volatility", 0.0)
            - 0.05 * _feature_value(features, "catalyst_timing_company_event_earnings", 0.0)
        )
        expected_return += (
            0.08 * pre_commercial * precommercial_value_gap
            + 0.06 * commercial_launch * launch_progress
            + 0.06 * commercial_launch * lifecycle_score
            + 0.05 * commercialized * pipeline_optionality
            + 0.05 * commercialized * capital_deployment
            + 0.04 * floor_support
            + 0.03 * hard_catalyst_presence
            - 0.08 * pre_commercial * competition_intensity * (1.0 - hard_catalyst_presence)
            - 0.03 * competition_intensity
        )
        success_prob = _sigmoid(
            (1.4 * _feature_value(features, "program_quality_pos_prior", 0.15))
            + (0.7 * _feature_value(features, "program_quality_phase_score", 0.20))
            + (0.4 * _feature_value(features, "program_quality_endpoint_score", 0.25))
            + (0.4 * _feature_value(features, "catalyst_timing_probability", 0.35))
            + (0.5 * _feature_value(features, "catalyst_timing_clinical_focus", 0.0))
            + (0.2 * _feature_value(features, "catalyst_timing_event_pdufa", 0.0))
            + (0.2 * _feature_value(features, "catalyst_timing_event_phase3_readout", 0.0))
            + (0.15 * hard_catalyst_presence)
            + (0.08 * commercial_launch * launch_progress)
            + (0.06 * commercialized * pipeline_optionality)
            - (0.6 * _feature_value(features, "program_quality_modality_risk", 0.50))
            - (0.3 * _feature_value(features, "catalyst_timing_company_event_earnings", 0.0))
            - (0.25 * pre_commercial * competition_intensity * (1.0 - hard_catalyst_presence))
        )
        return expected_return, success_prob
