from __future__ import annotations

from collections import Counter
import json

import pandas as pd

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


def _coerce_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _spearman(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 3:
        return 0.0
    corr = a.rank().corr(b.rank(), method="pearson")
    if corr is None or pd.isna(corr):
        return 0.0
    return float(corr)


class PortfolioConstructor:
    def __init__(self, store: LocalResearchStore | None = None, use_validation_priors: bool = True):
        self.store = store
        self.use_validation_priors = bool(store is not None and use_validation_priors)

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

        empirical_edge = self._empirical_edge(signal)
        scenario = self._scenario(signal, empirical_edge)
        risk_flags = self._risk_flags(signal, empirical_edge)

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
        target_weight = max(
            0.0,
            (base_weight * empirical_edge["weight_multiplier"])
            + scenario_bonus.get(scenario, 0.0)
            + empirical_edge["weight_bonus"]
            - risk_discount,
        )

        # Per-scenario caps must not exceed the execution layer's hard limits
        # (settings.max_single_position_pct = 4.0, execution_max_hard_catalyst = 4.0).
        # Storing inflated target_weights (e.g. 8%) that can never execute creates
        # a false picture in portfolio_recommendations and misleads scorecard feedback.
        if scenario == "watchlist only":
            target_weight = min(target_weight, 1.0)
        elif scenario == "avoid due to financing":
            target_weight = 0.0
        elif scenario == "pairs candidate":
            target_weight = min(target_weight, 2.5)
        elif scenario == "commercial compounder":
            target_weight = min(target_weight, 4.0)
        elif scenario == "pre-catalyst long":
            target_weight = min(target_weight, 4.0)

        # max_weight is the ceiling stored on the recommendation record.
        # Align to execution-layer limits so downstream readers aren't misled.
        max_weight = 4.0
        if signal.crowding_risk > 0.75:
            max_weight = 2.5
        if signal.financing_risk > 0.75:
            max_weight = min(max_weight, 2.0)

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
            company_state=signal.company_state,
            setup_type=signal.setup_type,
            risk_flags=risk_flags,
        )

    def top_ideas(self, signals: list[SignalArtifact]) -> list[PortfolioRecommendation]:
        recommendations = [self.recommend(signal) for signal in signals]
        return sorted(recommendations, key=lambda rec: (rec.target_weight, rec.confidence), reverse=True)

    @staticmethod
    def _scenario(signal: SignalArtifact, empirical_edge: dict[str, float] | None = None) -> str:
        edge_score = _coerce_float((empirical_edge or {}).get("edge_score"), 0.0)
        edge_tilt = _coerce_float((empirical_edge or {}).get("edge_tilt"), 0.0)
        catalyst_setup = signal.setup_type in {"hard_catalyst", "soft_catalyst"} or (
            signal.setup_type is None and signal.primary_event_bucket in {"clinical", "regulatory"}
        )
        if signal.financing_risk > 0.80:
            return "avoid due to financing"
        if signal.company_state == "pre_commercial" and signal.setup_type == "asymmetry_without_near_term_catalyst":
            asymmetry_bar = max(0.10, 0.12 - max(edge_tilt, 0.0))
            floor_bar = max(0.14, 0.18 - max(edge_tilt, 0.0))
            if signal.expected_return > asymmetry_bar and (signal.floor_support_pct or 0.0) > floor_bar:
                return "pairs candidate"
            return "watchlist only"
        if (
            catalyst_setup
            and signal.expected_return > max(0.11, 0.16 - max(edge_tilt, 0.0))
            and signal.catalyst_success_prob > max(0.47, 0.52 - max(edge_score, 0.0) * 0.20)
            and signal.thesis_horizon in {"30d", "90d"}
        ):
            return "pre-catalyst long"
        if (
            signal.company_state in {"commercial_launch", "commercialized"}
            and signal.expected_return > (0.10 + max(-edge_tilt, 0.0))
            and signal.catalyst_success_prob > (0.55 + max(-edge_score, 0.0) * 0.20)
        ):
            return "commercial compounder"
        if signal.expected_return > (0.08 + max(-edge_tilt, 0.0) * 0.5):
            return "pairs candidate"
        return "watchlist only"

    @staticmethod
    def _risk_flags(signal: SignalArtifact, empirical_edge: dict[str, float] | None = None) -> list[str]:
        flags: list[str] = []
        if signal.crowding_risk > 0.65:
            flags.append("crowded catalyst")
        if signal.financing_risk > 0.65:
            flags.append("financing overhang")
        if signal.catalyst_success_prob < 0.40:
            flags.append("low catalyst probability")
        if signal.primary_event_bucket == "earnings":
            flags.append("earnings-driven setup")
        if signal.company_state == "pre_commercial" and signal.setup_type == "asymmetry_without_near_term_catalyst":
            flags.append("no hard catalyst yet")
        if (signal.internal_upside_pct or 0.0) < 0.0:
            flags.append("negative asymmetry")
        if empirical_edge is not None:
            if empirical_edge["reliability"] >= 0.35 and empirical_edge["edge_score"] <= 0.02:
                flags.append("weak historical archetype edge")
            if empirical_edge["reliability"] >= 0.55 and empirical_edge["edge_score"] >= 0.12:
                flags.append("strong historical archetype edge")
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

    def _empirical_edge(self, signal: SignalArtifact) -> dict[str, float]:
        if not self.use_validation_priors or self.store is None:
            return {
                "edge_score": 0.0,
                "reliability": 0.0,
                "weight_multiplier": 1.0,
                "weight_bonus": 0.0,
                "edge_tilt": 0.0,
            }
        priors = self._validation_priors()
        if not priors:
            return {
                "edge_score": 0.0,
                "reliability": 0.0,
                "weight_multiplier": 1.0,
                "weight_bonus": 0.0,
                "edge_tilt": 0.0,
            }
        state_key = signal.company_state or "unknown"
        setup_key = signal.setup_type or "unknown"
        combo_key = f"{state_key}|{setup_key}"
        score_inputs = [
            (priors.get("state_setup_scorecards", {}).get(combo_key), 0.55),
            (priors.get("setup_type_scorecards", {}).get(setup_key), 0.30),
            (priors.get("company_state_scorecards", {}).get(state_key), 0.15),
        ]
        weighted_scores: list[tuple[float, float]] = []
        for metrics, base_weight in score_inputs:
            edge_score, reliability = self._scorecard_edge(metrics)
            if reliability <= 0.0:
                continue
            weighted_scores.append((edge_score, base_weight * reliability))
        if not weighted_scores:
            return {
                "edge_score": 0.0,
                "reliability": 0.0,
                "weight_multiplier": 1.0,
                "weight_bonus": 0.0,
                "edge_tilt": 0.0,
            }
        total_weight = sum(weight for _, weight in weighted_scores)
        edge_score = sum(score * weight for score, weight in weighted_scores) / max(total_weight, 1e-9)
        reliability = _clamp(total_weight, 0.0, 1.0)
        edge_tilt = edge_score * reliability * 0.20
        return {
            "edge_score": edge_score,
            "reliability": reliability,
            "weight_multiplier": _clamp(1.0 + (edge_score * reliability * 1.35), 0.75, 1.35),
            "weight_bonus": _clamp(edge_score * reliability * 4.5, -0.75, 1.0),
            "edge_tilt": edge_tilt,
        }

    def _validation_priors(self) -> dict[str, object]:
        if self.store is None:
            return {}
        payload = self.store.read_latest_raw_payload("validation_audits", "latest_walkforward_audit")
        if isinstance(payload, dict) and (
            payload.get("setup_type_scorecards") or payload.get("company_state_scorecards") or payload.get("state_setup_scorecards")
        ):
            return payload
        return self._derive_validation_priors()

    @staticmethod
    def _scorecard_edge(metrics: dict[str, object] | None) -> tuple[float, float]:
        if not metrics:
            return 0.0, 0.0
        rows = _coerce_float(metrics.get("rows"), 0.0)
        windows = _coerce_float(metrics.get("windows"), 0.0)
        if rows <= 0.0 or windows <= 0.0:
            return 0.0, 0.0
        rank_ic = _clamp(_coerce_float(metrics.get("rank_ic"), 0.0), -0.50, 0.50)
        hit_adj = _clamp(_coerce_float(metrics.get("hit_rate"), 0.50) - 0.50, -0.35, 0.35)
        spread = _clamp(_coerce_float(metrics.get("top_bottom_spread"), 0.0), -0.60, 0.60)
        beta = _clamp(_coerce_float(metrics.get("mean_return_90d"), 0.0), -0.35, 0.35)
        edge_score = (rank_ic * 0.45) + (hit_adj * 0.35) + (spread * 0.15) + (beta * 0.05)
        reliability = min(1.0, rows / 30.0) * min(1.0, windows / 6.0)
        if rows < 8.0:
            reliability *= 0.5
        return edge_score, reliability

    def _derive_validation_priors(self) -> dict[str, object]:
        if self.store is None:
            return {}
        signals = self.store.read_table("signal_artifacts")
        labels = self.store.read_table("labels")
        if signals.empty or labels.empty:
            return {}
        frame = signals.merge(labels, on=["ticker", "as_of"], how="inner")
        if frame.empty:
            return {}
        frame = frame.dropna(subset=["target_return_90d"]).copy()
        if frame.empty:
            return {}
        frame["company_state"] = frame["company_state"].fillna("unknown").astype(str)
        frame["setup_type"] = frame["setup_type"].fillna("unknown").astype(str)
        frame["state_setup_key"] = frame["company_state"] + "|" + frame["setup_type"]
        return {
            "company_state_scorecards": self._group_scorecards(frame, "company_state"),
            "setup_type_scorecards": self._group_scorecards(frame, "setup_type"),
            "state_setup_scorecards": self._group_scorecards(frame, "state_setup_key"),
        }

    @staticmethod
    def _group_scorecards(frame: pd.DataFrame, group_column: str) -> dict[str, dict[str, float]]:
        scorecards: dict[str, dict[str, float]] = {}
        for group_name, group in frame.groupby(group_column, dropna=False):
            key = "unknown" if pd.isna(group_name) or group_name is None else str(group_name)
            group = group.copy()
            group["as_of_ts"] = pd.to_datetime(group["as_of"], errors="coerce", utc=True, format="mixed").dt.tz_convert(None)
            top = group.nlargest(max(1, len(group) // 2), "expected_return")
            bottom = group.nsmallest(max(1, len(group) // 2), "expected_return")
            scorecards[key] = {
                "rows": float(len(group)),
                "windows": float(group["as_of_ts"].dt.normalize().nunique()) if "as_of_ts" in group else 0.0,
                "rank_ic": _spearman(group["expected_return"], group["target_return_90d"]),
                "hit_rate": float((group["target_return_90d"] > 0.0).mean()),
                "top_bottom_spread": float(top["target_return_90d"].mean() - bottom["target_return_90d"].mean())
                if len(group) > 1
                else 0.0,
                "mean_return_90d": float(group["target_return_90d"].mean()),
                "calibrated_brier": float(((group["catalyst_success_prob"] - group["target_catalyst_success"]) ** 2).mean())
                if "target_catalyst_success" in group
                else 1.0,
            }
        return scorecards

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
        company_state = row.get("company_state")
        if isinstance(company_state, float) and company_state != company_state:
            company_state = None
        setup_type = row.get("setup_type")
        if isinstance(setup_type, float) and setup_type != setup_type:
            setup_type = None
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
            company_state=None if company_state is None else str(company_state),
            setup_type=None if setup_type is None else str(setup_type),
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
        company_state = row.get("company_state")
        if isinstance(company_state, float) and company_state != company_state:
            company_state = None
        setup_type = row.get("setup_type")
        if isinstance(setup_type, float) and setup_type != setup_type:
            setup_type = None
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
            company_state=None if company_state is None else str(company_state),
            setup_type=None if setup_type is None else str(setup_type),
            internal_value=float(row["internal_value"]) if row.get("internal_value") is not None else None,
            internal_price_target=float(row["internal_price_target"]) if row.get("internal_price_target") is not None else None,
            internal_upside_pct=float(row["internal_upside_pct"]) if row.get("internal_upside_pct") is not None else None,
            floor_support_pct=float(row["floor_support_pct"]) if row.get("floor_support_pct") is not None else None,
            program_predictions=[],
        )


def aggregate_signal(
    ticker: str,
    as_of: str,
    predictions: list[ModelPrediction],
    evidence_rationale: list[str],
    evidence,
    company_state: str | None = None,
) -> SignalArtifact:
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
            company_state=company_state,
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
        company_state=company_state,
        program_predictions=predictions,
    )
