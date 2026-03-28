from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class FinanceData(TypedDict, total=False):
    ticker: str
    shortName: NotRequired[str | None]
    longName: NotRequired[str | None]
    sector: NotRequired[str | None]
    industry: NotRequired[str | None]
    description: NotRequired[str | None]
    marketCap: NotRequired[float | int | None]
    enterpriseValue: NotRequired[float | int | None]
    totalRevenue: NotRequired[float | int | None]
    grossMargins: NotRequired[float | None]
    operatingMargins: NotRequired[float | None]
    cash: NotRequired[float | int | None]
    debt: NotRequired[float | int | None]
    netIncome: NotRequired[float | int | None]
    trailing_6mo_return: NotRequired[float | None]
    momentum_3mo: NotRequired[float | None]
    volatility: NotRequired[float | None]
    price_now: NotRequired[float | None]
    _52WeekChange: NotRequired[float | None]


class TrialData(TypedDict, total=False):
    nct_id: NotRequired[str | None]
    title: NotRequired[str | None]
    overall_status: NotRequired[str | None]
    phase: NotRequired[list[str]]
    conditions: NotRequired[list[str]]
    interventions: NotRequired[list[str]]
    primary_outcomes: NotRequired[list[str]]
    enrollment: NotRequired[int | None]


class CompanyData(TypedDict, total=False):
    ticker: str
    company_name: str
    finance: FinanceData
    trials: list[TrialData]
    num_trials: int
    num_total_trials: int
    num_inactive_trials: int
    drug_names: list[str]
    best_phase: str
    base_pos: float
    total_enrollment: int
    max_single_enrollment: int
    phase_enrollment: dict[str, int]
    phase_trial_counts: dict[str, int]
    fda_adverse_events: int
    fda_serious_events: int
    fda_serious_ratio: float
    pubmed_papers: list[dict[str, Any]]
    num_papers: int
    conditions: list[str]


class AlphaBreakdown(TypedDict):
    value: float
    clinical: float
    safety: float
    finance: float
    momentum: float


class ScoreResult(TypedDict, total=False):
    signal: float
    pos: float
    rnpv: float
    alpha_breakdown: AlphaBreakdown
    conviction_weight: float
    recommended_allocation: float
    risk_parity_allocation: float
    error: NotRequired[str]
