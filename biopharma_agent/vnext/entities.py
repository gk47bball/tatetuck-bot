from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any


def _iso_or_none(value: date | datetime | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return datetime.combine(value, datetime.min.time()).isoformat()


@dataclass(slots=True)
class EvidenceSnippet:
    source: str
    source_id: str
    title: str
    excerpt: str
    url: str | None = None
    as_of: str | None = None
    confidence: float = 0.5


@dataclass(slots=True)
class Trial:
    trial_id: str
    title: str
    phase: str
    status: str
    conditions: list[str]
    interventions: list[str]
    enrollment: int
    primary_outcomes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ApprovedProduct:
    product_id: str
    name: str
    indication: str
    annual_revenue: float
    growth_signal: float


@dataclass(slots=True)
class CatalystEvent:
    event_id: str
    program_id: str | None
    event_type: str
    title: str
    expected_date: str | None
    horizon_days: int
    probability: float
    importance: float
    crowdedness: float
    status: str = "anticipated"


@dataclass(slots=True)
class FinancingEvent:
    event_id: str
    event_type: str
    probability: float
    horizon_days: int
    expected_dilution_pct: float
    summary: str


@dataclass(slots=True)
class Program:
    program_id: str
    name: str
    modality: str
    phase: str
    conditions: list[str]
    trials: list[Trial]
    pos_prior: float
    tam_estimate: float
    catalyst_events: list[CatalystEvent] = field(default_factory=list)
    evidence: list[EvidenceSnippet] = field(default_factory=list)


@dataclass(slots=True)
class CompanySnapshot:
    ticker: str
    company_name: str
    as_of: str
    market_cap: float
    enterprise_value: float
    revenue: float
    cash: float
    debt: float
    momentum_3mo: float | None
    trailing_6mo_return: float | None
    volatility: float | None
    programs: list[Program]
    approved_products: list[ApprovedProduct]
    catalyst_events: list[CatalystEvent]
    financing_events: list[FinancingEvent]
    evidence: list[EvidenceSnippet] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "as_of": self.as_of,
            "market_cap": self.market_cap,
            "enterprise_value": self.enterprise_value,
            "revenue": self.revenue,
            "cash": self.cash,
            "debt": self.debt,
            "momentum_3mo": self.momentum_3mo,
            "trailing_6mo_return": self.trailing_6mo_return,
            "volatility": self.volatility,
            "num_programs": len(self.programs),
            "num_approved_products": len(self.approved_products),
            "num_catalysts": len(self.catalyst_events),
            "num_financing_events": len(self.financing_events),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class FeatureVector:
    entity_id: str
    ticker: str
    as_of: str
    thesis_horizon: str
    feature_family: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row = {
            "entity_id": self.entity_id,
            "ticker": self.ticker,
            "as_of": self.as_of,
            "thesis_horizon": self.thesis_horizon,
        }
        row.update(self.feature_family)
        row.update({f"meta_{key}": value for key, value in self.metadata.items()})
        return row


@dataclass(slots=True)
class ModelPrediction:
    entity_id: str
    ticker: str
    as_of: str
    expected_return: float
    catalyst_success_prob: float
    confidence: float
    crowding_risk: float
    financing_risk: float
    thesis_horizon: str
    model_name: str
    model_version: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass(slots=True)
class SignalArtifact:
    ticker: str
    as_of: str
    expected_return: float
    catalyst_success_prob: float
    confidence: float
    crowding_risk: float
    financing_risk: float
    thesis_horizon: str
    rationale: list[str] = field(default_factory=list)
    supporting_evidence: list[EvidenceSnippet] = field(default_factory=list)
    primary_event_type: str | None = None
    primary_event_bucket: str | None = None
    program_predictions: list[ModelPrediction] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "as_of": self.as_of,
            "expected_return": self.expected_return,
            "catalyst_success_prob": self.catalyst_success_prob,
            "confidence": self.confidence,
            "crowding_risk": self.crowding_risk,
            "financing_risk": self.financing_risk,
            "thesis_horizon": self.thesis_horizon,
            "primary_event_type": self.primary_event_type,
            "primary_event_bucket": self.primary_event_bucket,
            "rationale": self.rationale,
            "supporting_evidence": [asdict(item) for item in self.supporting_evidence],
            "supporting_evidence_count": len(self.supporting_evidence),
            "program_prediction_count": len(self.program_predictions),
        }


@dataclass(slots=True)
class PortfolioRecommendation:
    ticker: str
    as_of: str
    stance: str
    target_weight: float
    max_weight: float
    confidence: float
    scenario: str
    thesis_horizon: str
    primary_event_type: str | None = None
    risk_flags: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExperimentRecord:
    experiment_id: str
    created_at: str
    model_name: str
    model_version: str
    train_window_start: str | None
    train_window_end: str | None
    holdout_window_start: str | None
    holdout_window_end: str | None
    metrics: dict[str, float]
    artifact_path: str | None = None


@dataclass(slots=True)
class CompanyAnalysis:
    snapshot: CompanySnapshot
    signal: SignalArtifact
    portfolio: PortfolioRecommendation
    feature_vectors: list[FeatureVector]
    program_predictions: list[ModelPrediction]
    literature_review: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def date_to_iso(value: date | datetime | None) -> str | None:
    return _iso_or_none(value)
