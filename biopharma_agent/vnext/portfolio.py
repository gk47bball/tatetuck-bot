from __future__ import annotations

from collections import Counter

from .entities import ModelPrediction, PortfolioRecommendation, SignalArtifact


class PortfolioConstructor:
    def recommend(self, signal: SignalArtifact) -> PortfolioRecommendation:
        scenario = self._scenario(signal)
        risk_flags = self._risk_flags(signal)

        gross_score = signal.expected_return * signal.confidence
        risk_discount = (0.50 * signal.crowding_risk) + (0.75 * signal.financing_risk)
        target_weight = max(0.0, gross_score * 12.0 - risk_discount * 4.0)

        if scenario == "watchlist only":
            target_weight = min(target_weight, 1.0)
        elif scenario == "avoid due to financing":
            target_weight = 0.0
        elif scenario == "pre-catalyst long":
            target_weight = min(target_weight + 1.5, 8.0)

        max_weight = 10.0
        if signal.crowding_risk > 0.75:
            max_weight = 5.0
        if signal.financing_risk > 0.75:
            max_weight = min(max_weight, 3.0)

        return PortfolioRecommendation(
            ticker=signal.ticker,
            as_of=signal.as_of,
            stance="long" if target_weight > 0 else "avoid",
            target_weight=round(min(target_weight, max_weight), 2),
            max_weight=max_weight,
            confidence=round(signal.confidence, 3),
            scenario=scenario,
            thesis_horizon=signal.thesis_horizon,
            risk_flags=risk_flags,
        )

    def top_ideas(self, signals: list[SignalArtifact]) -> list[PortfolioRecommendation]:
        recommendations = [self.recommend(signal) for signal in signals]
        return sorted(recommendations, key=lambda rec: (rec.target_weight, rec.confidence), reverse=True)

    @staticmethod
    def _scenario(signal: SignalArtifact) -> str:
        if signal.financing_risk > 0.80:
            return "avoid due to financing"
        if signal.expected_return > 0.18 and signal.thesis_horizon in {"30d", "90d"}:
            return "pre-catalyst long"
        if signal.expected_return > 0.12 and signal.catalyst_success_prob > 0.55:
            return "commercial compounder"
        if signal.expected_return > 0.06:
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
        return flags


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
        program_predictions=predictions,
    )
