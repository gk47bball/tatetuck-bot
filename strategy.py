"""
strategy.py — Alpha Stack v19 Fundamental Rebuild

Rebuilt to remove target leakage from the signal stack and to value biopharma
companies with indication-aware rNPV, portfolio POS, and a smooth
clinical-to-commercial transition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from biopharma_agent.types import AlphaBreakdown, CompanyData, ScoreResult, TrialData


PHASE_BASE = {
    "EARLY_PHASE1": 0.055,
    "PHASE1": 0.074,
    "PHASE2": 0.152,
    "PHASE3": 0.590,
    "NDA_BLA": 0.900,
    "APPROVED": 1.000,
}

PHASE_YEARS_TO_MARKET = {
    "EARLY_PHASE1": 10,
    "PHASE1": 8,
    "PHASE2": 5,
    "PHASE3": 2,
    "NDA_BLA": 1,
    "APPROVED": 0,
}

DISEASE_MULTIPLIERS = {
    "oncology": 1.10,
    "rare disease": 1.30,
    "hematology": 1.15,
    "neurology": 0.80,
    "immunology": 0.95,
    "infectious": 1.00,
    "cardiovascular": 0.85,
    "metabolic": 0.92,
    "gene therapy": 0.95,
    "cell therapy": 0.95,
}

THERAPEUTIC_TAMS = {
    "oncology": 8_000_000_000,
    "rare disease": 1_500_000_000,
    "hematology": 4_000_000_000,
    "neurology": 6_000_000_000,
    "immunology": 5_000_000_000,
    "infectious": 3_000_000_000,
    "cardiovascular": 7_000_000_000,
    "metabolic": 5_000_000_000,
    "gene therapy": 2_000_000_000,
    "cell therapy": 2_500_000_000,
}

INDICATION_TAMS = {
    "sickle cell": 2_500_000_000,
    "beta-thalassemia": 1_200_000_000,
    "thalassemia": 1_200_000_000,
    "non-hodgkin lymphoma": 6_000_000_000,
    "lymphoma": 6_000_000_000,
    "renal cell carcinoma": 4_500_000_000,
    "clear cell renal cell carcinoma": 4_500_000_000,
    "glycogen storage disease": 800_000_000,
    "alpha-1 antitrypsin": 3_000_000_000,
    "type 1 diabetes": 8_000_000_000,
    "muscular dystrophy": 3_500_000_000,
    "duchenne muscular dystrophy": 3_500_000_000,
    "nash": 12_000_000_000,
    "nonalcoholic steatohepatitis": 12_000_000_000,
    "hemophilia": 4_000_000_000,
    "cystic fibrosis": 6_000_000_000,
    "macular degeneration": 7_000_000_000,
    "retinal disease": 5_000_000_000,
    "solid tumor": 9_000_000_000,
    "multiple myeloma": 7_000_000_000,
    "leukemia": 6_500_000_000,
    "amyloidosis": 4_000_000_000,
    "parkinson": 5_000_000_000,
    "alzheimer": 10_000_000_000,
    "als": 2_500_000_000,
    "epilepsy": 5_000_000_000,
    "obesity": 20_000_000_000,
    "diabetes": 10_000_000_000,
    "hepatitis": 4_000_000_000,
    "pneumococcal": 9_000_000_000,
    "vaccine": 10_000_000_000,
    "colorectal": 8_000_000_000,
    "hepatocellular": 7_000_000_000,
    "craniopharyngioma": 1_200_000_000,
    "glioma": 2_500_000_000,
    "paroxysmal nocturnal hemoglobinuria": 3_500_000_000,
    "barrett": 2_000_000_000,
    "dense deposit disease": 800_000_000,
    "glomerulonephritis": 1_200_000_000,
}

CATEGORY_KEYWORDS = {
    "oncology": (
        "oncology",
        "carcinoma",
        "lymphoma",
        "leukemia",
        "myeloma",
        "tumor",
        "tumour",
        "cancer",
        "sarcoma",
        "melanoma",
    ),
    "hematology": (
        "hematology",
        "sickle cell",
        "thalassemia",
        "hemophilia",
        "hemoglobin",
        "anemia",
    ),
    "neurology": (
        "neurology",
        "alzheimer",
        "parkinson",
        "als",
        "amyotrophic",
        "huntington",
        "epilepsy",
        "neuro",
    ),
    "metabolic": (
        "diabetes",
        "obesity",
        "nash",
        "nonalcoholic steatohepatitis",
        "glycogen",
        "metabolic",
    ),
    "immunology": (
        "lupus",
        "colitis",
        "crohn",
        "arthritis",
        "dermatitis",
        "asthma",
        "psoriasis",
        "immunology",
        "immuno",
    ),
    "infectious": (
        "infectious",
        "infection",
        "viral",
        "vaccine",
        "vaccines",
        "pneumococcal",
        "covid",
        "influenza",
        "hepatitis",
        "hiv",
    ),
    "cardiovascular": (
        "cardio",
        "heart",
        "vascular",
        "hypertension",
        "cardiomyopathy",
    ),
    "rare disease": (
        "rare disease",
        "dystrophy",
        "glycogen storage disease",
        "cystic fibrosis",
        "alpha-1 antitrypsin",
        "hemophilia",
        "sickle cell",
        "thalassemia",
        "amyloidosis",
    ),
    "gene therapy": ("gene therapy", "crispr", "editing"),
    "cell therapy": ("cell therapy", "car-t", "car t", "stem cell"),
}

PENETRATION_BY_CATEGORY = {
    "oncology": 0.08,
    "rare disease": 0.25,
    "hematology": 0.16,
    "neurology": 0.12,
    "immunology": 0.12,
    "infectious": 0.15,
    "cardiovascular": 0.08,
    "metabolic": 0.10,
    "gene therapy": 0.18,
    "cell therapy": 0.14,
}

APPROVED_REVENUE_THRESHOLD = 10_000_000


@dataclass(frozen=True)
class AssetProfile:
    name: str
    phase: str
    conditions: list[str]
    enrollment: int
    pos: float


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _sigmoid_unit(score: float, slope: float = 1.0) -> float:
    return 2.0 / (1.0 + math.exp(-score * slope)) - 1.0


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _conditions_text(conditions: list[str]) -> str:
    return " ".join(conditions).lower()


def _matched_categories(conditions: list[str]) -> set[str]:
    cond_text = _conditions_text(conditions)
    matched = set()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if category in cond_text or any(keyword in cond_text for keyword in keywords):
            matched.add(category)
    return matched


def _disease_multiplier(conditions: list[str]) -> float:
    matched = _matched_categories(conditions)
    if not matched:
        return 1.0
    return max(DISEASE_MULTIPLIERS.get(category, 1.0) for category in matched)


def _estimate_indication_tam(conditions: list[str]) -> float:
    matched_tam = 0.0
    for condition in conditions:
        condition_lc = condition.lower()
        for keyword, tam in INDICATION_TAMS.items():
            if keyword in condition_lc:
                matched_tam = max(matched_tam, float(tam))

    if matched_tam > 0:
        return matched_tam

    category_tam = 0.0
    for category in _matched_categories(conditions):
        category_tam = max(category_tam, float(THERAPEUTIC_TAMS.get(category, 0.0)))
    return category_tam or 1_000_000_000.0


def _penetration_rate(conditions: list[str]) -> float:
    matched = _matched_categories(conditions)
    if not matched:
        return 0.15
    return min(PENETRATION_BY_CATEGORY.get(category, 0.15) for category in matched)


def _asset_name(trial: TrialData) -> str:
    interventions = [item for item in trial.get("interventions", []) if item]
    if interventions:
        return interventions[0].strip().lower()
    title = (trial.get("title") or "").strip().lower()
    return title or f"asset:{trial.get('nct_id') or 'unknown'}"


def _trial_phase(trial: TrialData) -> str:
    joined = " ".join(trial.get("phase", [])).upper()
    if "APPROVED" in joined:
        return "APPROVED"
    if "NDA" in joined or "BLA" in joined:
        return "NDA_BLA"
    if "PHASE3" in joined or "PHASE 3" in joined:
        return "PHASE3"
    if "PHASE2" in joined or "PHASE 2" in joined:
        return "PHASE2"
    if "EARLY" in joined:
        return "EARLY_PHASE1"
    return "PHASE1"


def _estimate_asset_pos(phase: str, enrollment: int, conditions: list[str]) -> float:
    base = PHASE_BASE.get(phase, PHASE_BASE["PHASE1"])
    enrollment_boost = min(math.log10(enrollment + 1) / 20.0, 0.06)
    pos = (base + enrollment_boost) * _disease_multiplier(conditions)
    return _clamp(pos, 0.03, 0.99)


def _build_asset_profiles(data: CompanyData) -> list[AssetProfile]:
    assets: dict[str, AssetProfile] = {}
    for trial in data.get("trials", []):
        name = _asset_name(trial)
        conditions = [c for c in trial.get("conditions", []) if c]
        phase = _trial_phase(trial)
        enrollment = _safe_int(trial.get("enrollment"), 0)
        candidate = AssetProfile(
            name=name,
            phase=phase,
            conditions=conditions,
            enrollment=enrollment,
            pos=_estimate_asset_pos(phase, enrollment, conditions),
        )
        current = assets.get(name)
        if current is None or candidate.pos > current.pos or candidate.enrollment > current.enrollment:
            assets[name] = candidate
    return sorted(assets.values(), key=lambda asset: (asset.pos, asset.enrollment), reverse=True)


def estimate_dynamic_pos(data: CompanyData) -> float:
    """Portfolio POS based on the top three unique assets, not a single lead program."""
    assets = _build_asset_profiles(data)
    if not assets:
        phase = data.get("best_phase", "PHASE1")
        return _clamp(PHASE_BASE.get(phase, PHASE_BASE["PHASE1"]), 0.03, 0.99)

    combined_failure = 1.0
    for asset in assets[:3]:
        combined_failure *= (1.0 - asset.pos)

    portfolio_pos = 1.0 - combined_failure
    literature_lift = min(math.log10(data.get("num_papers", 0) + 1) / 50.0, 0.03)
    return _clamp(portfolio_pos + literature_lift, 0.03, 0.99)


def _years_to_market(assets: list[AssetProfile], fallback_phase: str) -> float:
    if not assets:
        return float(PHASE_YEARS_TO_MARKET.get(fallback_phase, 8))
    weighted_years = 0.0
    total_weight = 0.0
    for asset in assets[:3]:
        weight = max(asset.pos, 0.05)
        weighted_years += PHASE_YEARS_TO_MARKET.get(asset.phase, 8) * weight
        total_weight += weight
    return weighted_years / max(total_weight, 0.1)


def _revenue_growth_implied(revenue: float, pos: float, momentum_3mo: float | None) -> float:
    commercial_weight = _clamp((revenue - 10_000_000) / 500_000_000, 0.0, 1.0)
    trend = momentum_3mo or 0.0
    growth = 0.08 + (0.10 * pos) + (0.20 * trend) + (0.08 * (1.0 - commercial_weight))
    return _clamp(growth, 0.05, 0.30)


def _has_approved_product(data: CompanyData) -> bool:
    revenue = _safe_float(data.get("finance", {}).get("totalRevenue"))
    best_phase = data.get("best_phase", "PHASE1")
    return revenue >= APPROVED_REVENUE_THRESHOLD or best_phase in {"APPROVED", "NDA_BLA"}


def estimate_advanced_rnpv(data: CompanyData, pos: float) -> float:
    """Indication-aware rNPV with portfolio POS and smooth commercial blending."""
    finance = data.get("finance", {})
    revenue = _safe_float(finance.get("totalRevenue"))
    cash = _safe_float(finance.get("cash"))
    debt = _safe_float(finance.get("debt"))
    net_cash = cash - debt
    momentum_3mo = finance.get("momentum_3mo")

    conditions = data.get("conditions", [])
    tam = _estimate_indication_tam(conditions)
    penetration = _penetration_rate(conditions)
    peak_revenue = tam * penetration

    assets = _build_asset_profiles(data)
    years_to_market = _years_to_market(assets, data.get("best_phase", "PHASE1"))
    discount_rate = 0.12
    operating_margin = 0.55
    clinical_burn = 25_000_000 + (data.get("num_trials", 0) * 6_000_000)
    clinical_burn = min(max(clinical_burn, 25_000_000), 180_000_000)

    clinical_rnpv = 0.0
    for year in range(1, 13):
        if year <= years_to_market:
            clinical_rnpv -= clinical_burn / ((1 + discount_rate) ** year)
            continue
        ramp = min(1.0, (year - years_to_market) / 4.0)
        cashflow = peak_revenue * operating_margin * ramp
        clinical_rnpv += (cashflow * pos) / ((1 + discount_rate) ** year)

    commercial_weight = _clamp((revenue - 10_000_000) / 500_000_000, 0.0, 1.0)
    clinical_weight = 1.0 - commercial_weight
    growth_rate = _revenue_growth_implied(revenue, pos, momentum_3mo)
    rev_multiple = 3.0 + (growth_rate * 15.0)
    commercial_rnpv = revenue * rev_multiple

    blended_rnpv = (clinical_rnpv * clinical_weight) + (commercial_rnpv * commercial_weight)
    cash_weight = 0.80 if commercial_weight < 0.5 else 0.55
    balance_sheet_adjustment = net_cash * cash_weight
    return max(blended_rnpv + balance_sheet_adjustment, 10_000_000.0)


def score_company(data: CompanyData) -> ScoreResult:
    """
    Evaluate a company with a five-factor signal where fundamentals dominate and
    momentum is explicitly a minority input.
    """
    zero_breakdown: AlphaBreakdown = {
        "value": 0.0,
        "clinical": 0.0,
        "safety": 0.0,
        "finance": 0.0,
        "momentum": 0.0,
    }

    finance = data.get("finance", {})
    market_cap = _safe_float(finance.get("marketCap"))
    if market_cap <= 0:
        return {
            "signal": 0.0,
            "pos": 0.0,
            "rnpv": 0.0,
            "alpha_breakdown": zero_breakdown,
            "conviction_weight": 0.0,
            "recommended_allocation": 0.0,
            "risk_parity_allocation": 0.0,
            "error": "No market cap",
        }

    revenue = _safe_float(finance.get("totalRevenue"))
    cash = _safe_float(finance.get("cash"))
    debt = _safe_float(finance.get("debt"))
    enterprise_value = _safe_float(finance.get("enterpriseValue"), market_cap) or market_cap
    momentum_3mo = finance.get("momentum_3mo")
    volatility = finance.get("volatility")

    pos = estimate_dynamic_pos(data)
    rnpv = estimate_advanced_rnpv(data, pos)
    indication_tam = _estimate_indication_tam(data.get("conditions", []))
    approved_product = revenue > 0 and _has_approved_product(data)
    assets = _build_asset_profiles(data)

    ratio = max(rnpv / market_cap, 1e-6)
    tam_cap_ratio = max(indication_tam / market_cap, 1e-6)
    revenue_cap_ratio = max((revenue * 6.0) / market_cap, 1e-6)
    commercial_weight = _clamp((revenue - 10_000_000) / 500_000_000, 0.0, 1.0)
    commercial_scale_bonus = min(math.log10(1.0 + (revenue / 100_000_000)), 1.2) if revenue > 0 else 0.0
    trend_value_bonus = 0.20 * _clamp(_safe_float(momentum_3mo), -0.5, 0.5)
    commercial_premium = 0.10 if approved_product else 0.0
    commercial_premium += commercial_weight * 0.20
    value_score = (
        (0.75 * math.log10(ratio))
        + (0.35 * math.log10(tam_cap_ratio))
        + (0.20 * math.log10(revenue_cap_ratio))
        + (0.20 * commercial_scale_bonus)
        + trend_value_bonus
        + commercial_premium
    )
    alpha_val = _sigmoid_unit(value_score, slope=1.1) * 0.35

    late_stage_assets = sum(1 for asset in assets[:3] if asset.phase in {"PHASE3", "NDA_BLA", "APPROVED"})
    mid_stage_assets = sum(1 for asset in assets[:3] if asset.phase == "PHASE2")
    total_enrollment = _safe_float(data.get("total_enrollment"))
    clinical_score = (
        (0.45 * math.log10(total_enrollment + 1))
        + (0.20 * math.log10(data.get("num_papers", 0) + 1))
        + (0.70 * late_stage_assets)
        + (0.25 * mid_stage_assets)
        + (0.20 * min(len(assets), 3))
        - 1.5
    )
    concentration_ratio = (
        _safe_float(data.get("max_single_enrollment")) / total_enrollment if total_enrollment > 0 else 1.0
    )
    if concentration_ratio > 0.85 and late_stage_assets == 0:
        clinical_score -= (concentration_ratio - 0.85) * 1.5
    if revenue < 25_000_000 and total_enrollment < 250:
        clinical_score -= 0.35
    elif revenue < 25_000_000 and total_enrollment < 500:
        clinical_score -= 0.15
    if revenue < 25_000_000 and market_cap > 2_000_000_000 and total_enrollment < 400:
        clinical_score -= 0.20
    alpha_clin = _sigmoid_unit(clinical_score, slope=1.0) * 0.20
    if revenue > 250_000_000 or (revenue > 10_000_000 and cash > 2_000_000_000):
        alpha_clin = max(alpha_clin, 0.0)

    serious_events = _safe_float(data.get("fda_serious_events"))
    serious_ratio = serious_events / max(total_enrollment, 50.0)
    fda_serious_ratio = _safe_float(data.get("fda_serious_ratio"))
    safety_penalty = _clamp((serious_ratio * 12.0) + (fda_serious_ratio * 1.5), 0.0, 1.0)
    alpha_safety = -safety_penalty * 0.15

    net_cash = cash - debt
    ev_cash_ratio = net_cash / max(enterprise_value, 1.0)
    revenue_support = math.log10(1.0 + (revenue / 50_000_000)) if revenue > 0 else 0.0
    runway_years = net_cash / max((25_000_000 + data.get("num_trials", 0) * 6_000_000), 1.0)
    runway_score = _clamp(runway_years / 4.0, -1.5, 1.5)
    finance_score = (0.70 * ev_cash_ratio) + (0.35 * revenue_support) + (0.15 * runway_score) - 0.10
    alpha_fin = _sigmoid_unit(finance_score, slope=1.3) * 0.15

    alpha_mom = 0.0
    if momentum_3mo is not None:
        trend = _clamp(_safe_float(momentum_3mo), -0.6, 0.6)
        if abs(trend) > 0.40:
            trend -= math.copysign((abs(trend) - 0.40) * 0.35, trend)
        vol_discount = _clamp(1.0 - (_safe_float(volatility) * 4.0), 0.35, 1.0)
        alpha_mom = _clamp(trend * 2.4 * vol_discount, -1.0, 1.0) * 0.20

    alpha_breakdown: AlphaBreakdown = {
        "value": alpha_val,
        "clinical": alpha_clin,
        "safety": alpha_safety,
        "finance": alpha_fin,
        "momentum": alpha_mom,
    }

    total_signal = alpha_val + alpha_clin + alpha_safety + alpha_fin + alpha_mom
    if volatility is not None and _safe_float(volatility) > 0.08:
        total_signal *= 0.80
    if data.get("num_trials", 0) > 25 and total_enrollment < 500:
        total_signal *= 0.75
    final_signal = _clamp(total_signal, -1.0, 1.0)

    raw_conviction = abs(final_signal)
    vol_penalty = 1.0 - min(_safe_float(volatility) * 2.0, 0.5) if volatility is not None else 0.8
    liq_modifier = 1.0
    if market_cap < 500_000_000:
        liq_modifier = 0.5
    elif market_cap < 1_500_000_000:
        liq_modifier = 0.8

    safety_modifier = max(0.2, 1.0 + alpha_safety)
    conviction_weight = _clamp(raw_conviction * vol_penalty * liq_modifier * safety_modifier, 0.0, 1.0)

    base_alloc = conviction_weight * 10.0
    if volatility is not None:
        risk_parity_allocation = base_alloc * (0.03 / max(_safe_float(volatility), 0.01))
    else:
        risk_parity_allocation = base_alloc * 0.5
    risk_parity_allocation = _clamp(risk_parity_allocation, 0.0, 15.0)

    return {
        "signal": final_signal,
        "pos": pos,
        "rnpv": rnpv,
        "alpha_breakdown": alpha_breakdown,
        "conviction_weight": conviction_weight,
        "recommended_allocation": round(base_alloc, 2),
        "risk_parity_allocation": round(risk_parity_allocation, 2),
    }
