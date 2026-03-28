from __future__ import annotations

from collections import Counter
import json

from .entities import ModelPrediction, PortfolioRecommendation, SignalArtifact
from .storage import LocalResearchStore
from .taxonomy import event_type_bucket


def _coerce_json(value):
    if isinstance(value, str) and value and value[0] in "[{":
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _coerce_list(value):
    value = _coerce_json(value)
    if value is None:
        return []
    if isinstance(value, float):
        try:
            if value != value:
                return []
        except Exception:
            return []
    if isinstance(value, list):
        return value
    return [value]


class PortfolioConstructor:
    def __init__(self, store: LocalResearchStore | None = None):
        self.store = store

    def recommend(
        self,
        signal: SignalArtifact,
        previous_recommendation: PortfolioRecommendation | None = None,
        previous_signal: SignalArtifact | None = None,
    ) -> PortfolioRecommendation:
        if previous_recommendation is None:
            previous_recommendation = self._lookup_previous_recommendation(signal.ticker, signal.as_of)
        if previous_signal is None:
            previous_signal = self._lookup_previous_signal(signal.ticker, signal.as_of)

        scenario = self._scenario(signal)
        risk_flags = self._risk_flags(signal)

        pm_edge = max(signal.expected_return, 0.0) * (0.70 + signal.confidence)
        clinical_bonus = 0.75 if signal.primary_event_bucket in {"clinical", "regulatory"} else 0.0
        persistence_bonus = 0.0
        if previous_signal is not None and previous_signal.expected_return > 0.08:
            if previous_signal.primary_event_bucket == signal.primary_event_bucket:
                persistence_bonus = 0.60
            else:
                persistence_bonus = 0.25
        base_weight = (pm_edge * 16.0) + (signal.confidence * 2.0) + clinical_bonus + persistence_bonus - 0.75
        scenario_bonus = {
            "pre-catalyst long": 2.0,
            "commercial compounder": 1.0,
            "pairs candidate": 0.25,
            "watchlist only": 0.0,
            "avoid due to financing": 0.0,
        }
        risk_discount = (signal.crowding_risk * 1.8) + (signal.financing_risk * 3.5)
        if signal.primary_event_bucket == "earnings":
            risk_discount += 0.75
        target_weight = max(0.0, base_weight + scenario_bonus.get(scenario, 0.0) - risk_discount)

        if scenario == "watchlist only":
            target_weight = min(target_weight, 1.0)
        elif scenario == "avoid due to financing":
            target_weight = 0.0
        elif scenario == "pairs candidate":
            target_weight = min(target_weight, 2.5)
        elif scenario == "commercial compounder":
            target_weight = min(target_weight, 5.0)
        elif scenario == "pre-catalyst long":
            target_weight = min(target_weight, 8.0)

        max_weight = 10.0
        if signal.crowding_risk > 0.75:
            max_weight = 5.0
        if signal.financing_risk > 0.75:
            max_weight = min(max_weight, 3.0)

        target_weight = self._smooth_weight(
            raw_weight=min(target_weight, max_weight),
            previous_recommendation=previous_recommendation,
            signal=signal,
            previous_signal=previous_signal,
        )
        target_weight = min(target_weight, max_weight)

        return PortfolioRecommendation(
            ticker=signal.ticker,
            as_of=signal.as_of,
            stance="long" if target_weight > 0 else "avoid",
            target_weight=round(target_weight, 2),
            max_weight=max_weight,
            confidence=round(signal.confidence, 3),
            scenario=scenario,
            thesis_horizon=signal.thesis_horizon,
            primary_event_type=signal.primary_event_type,
            risk_flags=risk_flags,
        )

    def top_ideas(self, signals: list[SignalArtifact]) -> list[PortfolioRecommendation]:
        recommendations = [self.recommend(signal) for signal in signals]
        return sorted(recommendations, key=lambda rec: (rec.target_weight, rec.confidence), reverse=True)

    @staticmethod
    def _scenario(signal: SignalArtifact) -> str:
        if signal.financing_risk > 0.80:
            return "avoid due to financing"
        if (
            signal.primary_event_bucket in {"clinical", "regulatory"}
            and signal.expected_return > 0.16
            and signal.catalyst_success_prob > 0.52
            and signal.thesis_horizon in {"30d", "90d"}
        ):
            return "pre-catalyst long"
        if (
            signal.primary_event_bucket == "commercial"
            and signal.expected_return > 0.10
            and signal.catalyst_success_prob > 0.55
        ):
            return "commercial compounder"
        if signal.expected_return > 0.08:
            return "pairs candidate"
        return "watchlist only"

    @staticmethod
    def _risk_flags(signal: SignalArtifact) -> list[str]:
        flags: list[str] = []
        if signal.crowding_risk > 0.65:
            flags.append("crowded catalyst")
        if signal.financing_risk > 0.65:
            flags.append("financing overhang")
        if signal.catalyst_success_prob < 0.40:
            flags.append("low catalyst probability")
        if signal.primary_event_bucket == "earnings":
            flags.append("earnings-driven setup")
        return flags

    @staticmethod
    def _smooth_weight(
        raw_weight: float,
        previous_recommendation: PortfolioRecommendation | None,
        signal: SignalArtifact,
        previous_signal: SignalArtifact | None,
    ) -> float:
        if previous_recommendation is None:
            return PortfolioConstructor._quantize_weight(raw_weight)

        prior_weight = max(previous_recommendation.target_weight, 0.0)
        if raw_weight <= 0.0:
            if previous_signal is not None and previous_signal.expected_return > 0.05 and signal.confidence > 0.45:
                return PortfolioConstructor._quantize_weight(min(prior_weight, 0.75))
            return 0.0

        if (
            prior_weight >= 1.0
            and raw_weight < 1.0
            and previous_signal is not None
            and previous_signal.expected_return > 0.06
            and signal.expected_return > 0.04
        ):
            floor_weight = max(1.0, prior_weight * 0.60)
            return PortfolioConstructor._quantize_weight(min(floor_weight, max(prior_weight, raw_weight)))

        delta = raw_weight - prior_weight
        abs_delta = abs(delta)
        if abs_delta < 0.75:
            smoothed = prior_weight + (delta * 0.35)
        elif abs_delta < 2.0:
            smoothed = prior_weight + (delta * 0.55)
        else:
            smoothed = prior_weight + (delta * 0.80)
        return PortfolioConstructor._quantize_weight(max(smoothed, 0.0))

    @staticmethod
    def _quantize_weight(weight: float) -> float:
        return round(max(weight, 0.0) * 4.0) / 4.0

    def _lookup_previous_recommendation(self, ticker: str, as_of: str) -> PortfolioRecommendation | None:
        if self.store is None:
            return None
        frame = self.store.read_table("portfolio_recommendations")
        if frame.empty:
            return None
        subset = frame[frame["ticker"] == ticker].copy()
        if subset.empty:
            return None
        subset = subset[subset["as_of"] < as_of].sort_values("as_of")
        if subset.empty:
            return None
        row = subset.iloc[-1]
        primary_event_type = row.get("primary_event_type")
        if isinstance(primary_event_type, float) and primary_event_type != primary_event_type:
            primary_event_type = None
        return PortfolioRecommendation(
            ticker=str(row["ticker"]),
            as_of=str(row["as_of"]),
            stance=str(row["stance"]),
            target_weight=float(row["target_weight"]),
            max_weight=float(row["max_weight"]),
            confidence=float(row["confidence"]),
            scenario=str(row["scenario"]),
            thesis_horizon=str(row["thesis_horizon"]),
            primary_event_type=None if primary_event_type is None else str(primary_event_type),
            risk_flags=_coerce_list(row.get("risk_flags")),
        )

    def _lookup_previous_signal(self, ticker: str, as_of: str) -> SignalArtifact | None:
        if self.store is None:
            return None
        frame = self.store.read_table("signal_artifacts")
        if frame.empty:
            return None
        subset = frame[frame["ticker"] == ticker].copy()
        if subset.empty:
            return None
        subset = subset[subset["as_of"] < as_of].sort_values("as_of")
        if subset.empty:
            return None
        row = subset.iloc[-1]
        primary_event_type = row.get("primary_event_type")
        if isinstance(primary_event_type, float) and primary_event_type != primary_event_type:
            primary_event_type = None
        if primary_event_type is not None and not isinstance(primary_event_type, str):
            primary_event_type = str(primary_event_type)
        primary_bucket = row.get("primary_event_bucket")
        if primary_bucket is None or (isinstance(primary_bucket, float) and primary_bucket != primary_bucket):
            primary_bucket = event_type_bucket(primary_event_type)
        return SignalArtifact(
            ticker=str(row["ticker"]),
            as_of=str(row["as_of"]),
            expected_return=float(row["expected_return"]),
            catalyst_success_prob=float(row["catalyst_success_prob"]),
            confidence=float(row["confidence"]),
            crowding_risk=float(row["crowding_risk"]),
            financing_risk=float(row["financing_risk"]),
            thesis_horizon=str(row["thesis_horizon"]),
            rationale=_coerce_list(row.get("rationale")),
            supporting_evidence=[],
            primary_event_type=primary_event_type,
            primary_event_bucket=str(primary_bucket),
            program_predictions=[],
        )


def aggregate_signal(ticker: str, as_of: str, predictions: list[ModelPrediction], evidence_rationale: list[str], evidence) -> SignalArtifact:
    if not predictions:
        return SignalArtifact(
            ticker=ticker,
            as_of=as_of,
            expected_return=0.0,
            catalyst_success_prob=0.0,
            confidence=0.0,
            crowding_risk=0.0,
            financing_risk=0.0,
            thesis_horizon="90d",
            rationale=evidence_rationale,
            supporting_evidence=list(evidence),
            primary_event_type=None,
            primary_event_bucket="none",
            program_predictions=[],
        )

    weights = [max(pred.confidence, 0.05) for pred in predictions]
    weight_sum = sum(weights)
    expected_return = sum(pred.expected_return * weight for pred, weight in zip(predictions, weights)) / weight_sum
    catalyst_success_prob = sum(pred.catalyst_success_prob * weight for pred, weight in zip(predictions, weights)) / weight_sum
    confidence = sum(pred.confidence * weight for pred, weight in zip(predictions, weights)) / weight_sum
    crowding_risk = sum(pred.crowding_risk * weight for pred, weight in zip(predictions, weights)) / weight_sum
    financing_risk = max(pred.financing_risk for pred in predictions)
    horizon = Counter(pred.thesis_horizon for pred in predictions).most_common(1)[0][0]
    event_type_weights = Counter()
    for pred, weight in zip(predictions, weights):
        event_type = pred.metadata.get("event_type")
        if event_type:
            event_type_weights[str(event_type)] += float(weight)
    primary_event_type = event_type_weights.most_common(1)[0][0] if event_type_weights else None

    return SignalArtifact(
        ticker=ticker,
        as_of=as_of,
        expected_return=float(expected_return),
        catalyst_success_prob=float(catalyst_success_prob),
        confidence=float(confidence),
        crowding_risk=float(crowding_risk),
        financing_risk=float(financing_risk),
        thesis_horizon=horizon,
        rationale=evidence_rationale,
        supporting_evidence=list(evidence),
        primary_event_type=primary_event_type,
        primary_event_bucket=event_type_bucket(primary_event_type),
        program_predictions=predictions,
    )
