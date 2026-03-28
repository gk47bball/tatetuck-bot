from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .entities import ExperimentRecord, FeatureVector, ModelPrediction
from .storage import LocalResearchStore


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


@dataclass(slots=True)
class EnsembleBundle:
    regressor: HistGradientBoostingRegressor | None = None
    classifier: HistGradientBoostingClassifier | None = None
    feature_columns: list[str] | None = None


class EventDrivenEnsemble:
    def __init__(self, store: LocalResearchStore | None = None, model_name: str = "event_driven_ensemble", model_version: str = "v1"):
        self.store = store or LocalResearchStore()
        self.model_name = model_name
        self.model_version = model_version
        self.bundle = self._load()

    def _load(self) -> EnsembleBundle:
        path = self.store.model_path(self.model_name, self.model_version)
        if not path.exists():
            return EnsembleBundle()
        with open(path, "rb") as f:
            return pickle.load(f)

    def _save(self) -> Path:
        path = self.store.model_path(self.model_name, self.model_version)
        with open(path, "wb") as f:
            pickle.dump(self.bundle, f)
        return path

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
                "target_catalyst_success",
            }
            and not column.startswith("target_")
            and not column.startswith("meta_")
            and column != "evaluation_date"
        ]

    def fit(self, frame: pd.DataFrame) -> ExperimentRecord | None:
        required = {"target_return_90d", "target_catalyst_success"}
        if frame.empty or not required.issubset(frame.columns):
            return None
        usable = frame.dropna(subset=["target_return_90d", "target_catalyst_success"])
        if len(usable) < 20:
            return None

        feature_columns = self.feature_columns(usable)
        X = usable[feature_columns].fillna(0.0).to_numpy(dtype=float)
        y_reg = usable["target_return_90d"].to_numpy(dtype=float)
        y_clf = usable["target_catalyst_success"].to_numpy(dtype=int)

        self.bundle = EnsembleBundle(
            regressor=HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, random_state=7),
            classifier=HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, random_state=7),
            feature_columns=feature_columns,
        )
        self.bundle.regressor.fit(X, y_reg)
        self.bundle.classifier.fit(X, y_clf)
        artifact_path = self._save()

        record = ExperimentRecord(
            experiment_id=f"{self.model_name}_{self.model_version}_{len(usable)}",
            created_at=pd.Timestamp.utcnow().isoformat(),
            model_name=self.model_name,
            model_version=self.model_version,
            train_window_start=str(usable["as_of"].min()),
            train_window_end=str(usable["as_of"].max()),
            holdout_window_start=None,
            holdout_window_end=None,
            metrics={"num_rows": float(len(usable))},
            artifact_path=str(artifact_path),
        )
        self.store.register_experiment(record)
        return record

    def score(self, feature_vectors: Iterable[FeatureVector]) -> list[ModelPrediction]:
        feature_vectors = list(feature_vectors)
        if not feature_vectors:
            return []
        frame = pd.DataFrame([vector.to_row() for vector in feature_vectors])
        return self._predict_frame(frame, feature_vectors)

    def _predict_frame(self, frame: pd.DataFrame, feature_vectors: list[FeatureVector]) -> list[ModelPrediction]:
        predictions: list[ModelPrediction] = []
        feature_columns = self.bundle.feature_columns or self.feature_columns(frame)
        X = frame.reindex(columns=feature_columns, fill_value=0.0).fillna(0.0).to_numpy(dtype=float)

        reg_scores = None
        clf_scores = None
        if self.bundle.regressor is not None and self.bundle.classifier is not None and self.bundle.feature_columns:
            reg_scores = self.bundle.regressor.predict(X)
            clf_scores = self.bundle.classifier.predict_proba(X)[:, 1]

        for index, vector in enumerate(feature_vectors):
            rule_expected_return, rule_success_prob = self._rule_score(vector)
            expected_return = rule_expected_return
            catalyst_success_prob = rule_success_prob
            if reg_scores is not None and clf_scores is not None:
                expected_return = (0.40 * rule_expected_return) + (0.60 * float(reg_scores[index]))
                catalyst_success_prob = (0.40 * rule_success_prob) + (0.60 * float(clf_scores[index]))

            crowding_risk = _sigmoid(
                vector.feature_family.get("catalyst_timing_crowdedness", 0.30) * 2.4
                + max(vector.feature_family.get("market_flow_momentum_3mo", 0.0), 0.0)
            )
            financing_risk = _sigmoid(
                (0.9 * vector.feature_family.get("balance_sheet_financing_pressure", 0.0))
                - (0.25 * vector.feature_family.get("balance_sheet_runway_score", 0.0))
                + (0.35 * vector.feature_family.get("balance_sheet_debt_to_cap", 0.0))
            )
            confidence = _sigmoid(
                (1.2 * vector.feature_family.get("program_quality_pos_prior", 0.20))
                + (0.8 * vector.feature_family.get("catalyst_timing_probability", 0.30))
                + (0.4 * vector.feature_family.get("program_quality_trial_count", 0.0))
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
        self.store.write_predictions(predictions)
        return predictions

    def _rule_score(self, vector: FeatureVector) -> tuple[float, float]:
        features = vector.feature_family
        expected_return = (
            0.16 * features.get("program_quality_pos_prior", 0.15)
            + 0.10 * features.get("program_quality_tam_to_cap", 0.0)
            + 0.14 * features.get("catalyst_timing_probability", 0.35)
            + 0.10 * features.get("catalyst_timing_importance", 0.40)
            + 0.08 * features.get("commercial_execution_revenue_to_cap", -1.0)
            + 0.08 * features.get("balance_sheet_cash_to_cap", 0.0)
            + 0.05 * features.get("market_flow_momentum_3mo", 0.0)
            - 0.09 * features.get("program_quality_modality_risk", 0.50)
            - 0.08 * features.get("catalyst_timing_crowdedness", 0.30)
            - 0.07 * features.get("balance_sheet_financing_pressure", 0.0)
            - 0.04 * features.get("market_flow_volatility", 0.0)
        )
        success_prob = _sigmoid(
            (1.4 * features.get("program_quality_pos_prior", 0.15))
            + (0.5 * features.get("program_quality_phase_score", 0.20))
            + (0.3 * features.get("program_quality_endpoint_score", 0.25))
            + (0.4 * features.get("catalyst_timing_probability", 0.35))
            - (0.6 * features.get("program_quality_modality_risk", 0.50))
        )
        return expected_return, success_prob
