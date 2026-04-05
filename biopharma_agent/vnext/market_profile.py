from __future__ import annotations

from typing import Any

from .entities import CatalystEvent, CompanySnapshot, SignalArtifact
from .taxonomy import event_type_bucket, event_type_priority, is_synthetic_event

PHASE_RANK = {
    "EARLY_PHASE1": 1,
    "PHASE1": 2,
    "PHASE2": 3,
    "PHASE3": 4,
    "NDA_BLA": 5,
    "APPROVED": 6,
}

INDICATION_LANDSCAPES: list[tuple[tuple[str, ...], dict[str, Any]]] = [
    (
        ("obesity", "weight loss", "glp-1", "mash"),
        {
            "label": "obesity / cardiometabolic",
            "competition_intensity": 0.95,
            "leaders": ["LLY", "NVO"],
            "differentiation_focus": "must beat current leaders on efficacy, tolerability, convenience, supply, or cardio-metabolic breadth",
            "market_style": "category enthusiasm can create a sentiment floor, but differentiation bar is very high",
        },
    ),
    (
        ("diabetes", "type 1 diabetes", "type 2 diabetes"),
        {
            "label": "diabetes",
            "competition_intensity": 0.85,
            "leaders": ["LLY", "NVO", "SNY"],
            "differentiation_focus": "needs durable efficacy plus real differentiation on convenience, safety, or disease modification",
            "market_style": "large TAM, but investors discount incremental products heavily",
        },
    ),
    (
        ("sickle cell", "thalassemia", "hemophilia", "hemoglobin"),
        {
            "label": "hematology",
            "competition_intensity": 0.65,
            "leaders": ["VRTX", "CRSP", "BMRN"],
            "differentiation_focus": "durability, safety, and access matter as much as raw efficacy",
            "market_style": "clear catalysts matter, but commercial scale depends on treatment-center adoption",
        },
    ),
    (
        ("renal cell carcinoma", "carcinoma", "lymphoma", "leukemia", "myeloma", "tumor", "cancer"),
        {
            "label": "oncology",
            "competition_intensity": 0.82,
            "leaders": ["MRK", "BMY", "ROG"],
            "differentiation_focus": "needs clear efficacy, biomarker positioning, or combinability versus entrenched standards of care",
            "market_style": "large markets exist, but the Street discounts crowded oncology assets aggressively",
        },
    ),
    (
        ("nash", "mash", "steatohepatitis", "alpha-1 antitrypsin", "glycogen storage disease"),
        {
            "label": "metabolic / liver",
            "competition_intensity": 0.75,
            "leaders": ["MDGL", "AKRO"],
            "differentiation_focus": "needs meaningful histology or fibrosis advantage plus tolerability and payer viability",
            "market_style": "big opportunity, but investors fade assets that look merely 'good enough'",
        },
    ),
    (
        ("alzheimer", "parkinson", "als", "huntington", "epilepsy"),
        {
            "label": "neurology",
            "competition_intensity": 0.70,
            "leaders": ["BIIB", "LLY", "PTCT"],
            "differentiation_focus": "trial design and effect size credibility matter more than TAM alone",
            "market_style": "binary readouts can rerate names sharply, but skepticism stays high",
        },
    ),
    (
        ("cystic fibrosis", "macular degeneration", "retina"),
        {
            "label": "specialty franchise",
            "competition_intensity": 0.72,
            "leaders": ["VRTX", "REGN", "APLS"],
            "differentiation_focus": "must show tangible convenience, durability, or access advantage against established franchises",
            "market_style": "good floor if commercial base exists, but upside needs lifecycle proof",
        },
    ),
]

HARD_CATALYST_TYPES = {"phase1_readout", "phase2_readout", "phase3_readout", "pdufa", "adcom", "clinical_readout"}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _all_conditions(snapshot: CompanySnapshot) -> list[str]:
    conditions: list[str] = []
    for program in snapshot.programs:
        conditions.extend(program.conditions)
    for product in snapshot.approved_products:
        conditions.append(product.indication)
    return [item for item in conditions if item]


def primary_indication(snapshot: CompanySnapshot) -> str:
    override = str(snapshot.metadata.get("primary_indication_override") or "").strip()
    if override:
        return override
    if snapshot.approved_products and classify_company_state(snapshot) in {"commercial_launch", "commercialized"}:
        return str(snapshot.approved_products[0].indication or "unspecified")
    conditions = _all_conditions(snapshot)
    if conditions:
        return max(set(conditions), key=lambda item: (conditions.count(item), len(item)))
    description = str(snapshot.metadata.get("description") or "").lower()
    for keywords, payload in INDICATION_LANDSCAPES:
        if any(keyword in description for keyword in keywords):
            return str(payload["label"])
    return "unspecified"


def competitive_landscape(indication: str) -> dict[str, Any]:
    indication_lc = indication.lower()
    for keywords, payload in INDICATION_LANDSCAPES:
        if any(keyword in indication_lc for keyword in keywords):
            return payload.copy()
    return {
        "label": "general biotech",
        "competition_intensity": 0.55,
        "leaders": [],
        "differentiation_focus": "needs cleaner proof of differentiation versus the existing treatment set",
        "market_style": "market where valuation depends heavily on catalyst clarity and balance-sheet durability",
    }


def classify_company_state(snapshot: CompanySnapshot) -> str:
    override = str(snapshot.metadata.get("company_state_override") or "").strip()
    if override in {"pre_commercial", "commercial_launch", "commercialized"}:
        return override
    approved_products = len(snapshot.approved_products)
    revenue = float(snapshot.revenue or 0.0)
    if approved_products <= 0 and revenue < 75_000_000:
        return "pre_commercial"
    if approved_products > 0 and revenue < 500_000_000:
        return "commercial_launch"
    if approved_products > 0 and revenue >= 500_000_000:
        return "commercialized"
    if revenue >= 500_000_000:
        return "commercialized"
    if revenue >= 75_000_000:
        return "commercial_launch"
    return "pre_commercial"


def build_snapshot_profile(snapshot: CompanySnapshot) -> dict[str, Any]:
    state = classify_company_state(snapshot)
    indication = primary_indication(snapshot)
    landscape = competitive_landscape(indication)
    market_cap = max(float(snapshot.market_cap or 0.0), 1.0)
    net_cash = float(snapshot.cash or 0.0) - float(snapshot.debt or 0.0)
    lead_tam = max((float(program.tam_estimate or 0.0) for program in snapshot.programs), default=0.0)
    late_stage_noncommercial = [
        program
        for program in snapshot.programs
        if PHASE_RANK.get(program.phase, 0) >= 4 and program.phase != "APPROVED"
    ]
    follow_on_programs = [
        program
        for program in snapshot.programs
        if program.phase != "APPROVED"
    ]
    hard_catalysts = [
        event
        for event in snapshot.catalyst_events
        if event.event_type in HARD_CATALYST_TYPES and event.horizon_days <= 180
    ]
    floor_support_pct = _clamp(
        (max(net_cash, 0.0) + (0.35 * float(snapshot.revenue or 0.0))) / market_cap,
        0.0,
        1.25,
    )
    launch_progress_pct = 0.0
    if state != "pre_commercial" and lead_tam > 0:
        launch_progress_pct = _clamp(float(snapshot.revenue or 0.0) / max(lead_tam * 0.10, 1.0), 0.0, 1.25)
    lifecycle_management_score = _clamp(len(follow_on_programs) / 4.0, 0.0, 1.5)
    pipeline_optionality_score = _clamp(
        sum(PHASE_RANK.get(program.phase, 0) for program in late_stage_noncommercial) / 8.0,
        0.0,
        2.0,
    )
    capital_deployment_score = _clamp((max(net_cash, 0.0) / market_cap) + (0.35 * pipeline_optionality_score), 0.0, 1.5)
    state_focus = {
        "pre_commercial": "The core question is whether the lead drug is differentiated enough to work and matter versus the current standard of care.",
        "commercial_launch": "The core question is whether launch velocity, lifecycle expansion, and competitive positioning can beat expectations.",
        "commercialized": "The core question is franchise durability plus pipeline, BD/M&A, and capital deployment optionality.",
    }[state]
    special_situation = str(snapshot.metadata.get("special_situation") or "")
    if special_situation == "pending_transaction":
        state_focus = "The core question is closing certainty and deal-spread capture rather than standalone product execution."
    elif special_situation == "partnered_royalty_transition":
        state_focus = "The core question is how much value now comes from partner-led milestones and royalties rather than a standalone commercial franchise."
    elif special_situation == "partner_search_overhang":
        state_focus = "The core question is whether the catalyst still matters once partner appetite, commercial ownership, and subgroup breadth are stress-tested."
    elif special_situation == "regulatory_overhang":
        state_focus = "The core question is whether the franchise can absorb an active regulatory dispute without a lasting de-rating."

    return {
        "company_state": state,
        "primary_indication": indication,
        "competition_intensity": float(landscape["competition_intensity"]),
        "market_leaders": list(landscape["leaders"]),
        "differentiation_focus": str(landscape["differentiation_focus"]),
        "market_style": str(landscape["market_style"]),
        "floor_support_pct": float(floor_support_pct),
        "launch_progress_pct": float(launch_progress_pct),
        "lifecycle_management_score": float(lifecycle_management_score),
        "pipeline_optionality_score": float(pipeline_optionality_score),
        "capital_deployment_score": float(capital_deployment_score),
        "hard_catalyst_count": len(hard_catalysts),
        "has_near_term_hard_catalyst": bool(hard_catalysts),
        "state_focus": state_focus,
    }


def update_snapshot_profile(snapshot: CompanySnapshot) -> CompanySnapshot:
    snapshot.metadata.update(build_snapshot_profile(snapshot))
    return snapshot


def classify_setup_type(
    snapshot: CompanySnapshot,
    signal: SignalArtifact,
    primary_event: CatalystEvent | None,
    profile: dict[str, Any],
) -> str:
    state = str(profile.get("company_state") or "pre_commercial")
    competition_intensity = float(profile.get("competition_intensity", 0.55) or 0.55)
    floor_support_pct = float(profile.get("floor_support_pct", 0.0) or 0.0)
    has_hard_catalyst = bool(profile.get("has_near_term_hard_catalyst"))
    event_bucket = event_type_bucket(primary_event.event_type if primary_event is not None else signal.primary_event_type)
    event_status = primary_event.status if primary_event is not None else None

    if (
        event_bucket in {"clinical", "regulatory"}
        and primary_event is not None
        and primary_event.horizon_days <= 180
        and not is_synthetic_event(event_status, primary_event.title)
    ):
        return "hard_catalyst"
    if (
        state == "pre_commercial"
        and event_bucket in {"clinical", "regulatory"}
        and primary_event is not None
        and primary_event.horizon_days <= 180
        and is_synthetic_event(event_status, primary_event.title if primary_event is not None else None)
        and competition_intensity >= 0.80
    ):
        return "asymmetry_without_near_term_catalyst"
    if (
        event_bucket in {"clinical", "regulatory"}
        and primary_event is not None
        and primary_event.horizon_days <= 180
    ):
        return "soft_catalyst"
    if (
        event_bucket == "strategic"
        and primary_event is not None
        and primary_event.horizon_days <= 180
        and not is_synthetic_event(event_status, primary_event.title if primary_event is not None else None)
        and state in {"commercial_launch", "commercialized"}
    ):
        return "capital_allocation"
    if state == "commercial_launch":
        return "launch_asymmetry"
    if state == "commercialized" and float(profile.get("capital_deployment_score", 0.0) or 0.0) >= 0.55:
        return "capital_allocation"
    if state == "commercialized" and float(profile.get("pipeline_optionality_score", 0.0) or 0.0) >= 0.35:
        return "pipeline_optionality"
    if float(snapshot.momentum_3mo or 0.0) <= -0.12 and floor_support_pct >= 0.20:
        return "sentiment_floor"
    if state == "pre_commercial" and (not has_hard_catalyst or competition_intensity >= 0.85):
        return "asymmetry_without_near_term_catalyst"
    return "watchful"


def _phase_share_cap(phase_rank: int) -> float:
    return {
        6: 0.16,
        5: 0.12,
        4: 0.08,
        3: 0.05,
        2: 0.03,
        1: 0.02,
        0: 0.015,
    }.get(phase_rank, 0.02)


def _risk_adjusted_program_sales(snapshot: CompanySnapshot, competition_intensity: float) -> float:
    top_programs = sorted(snapshot.programs, key=lambda program: float(program.tam_estimate or 0.0), reverse=True)[:3]
    competition_adjustment = _clamp(1.0 - (0.40 * competition_intensity), 0.35, 0.90)
    risk_adjusted_sales = 0.0
    for program in top_programs:
        phase_rank = PHASE_RANK.get(program.phase, 0)
        share_cap = _phase_share_cap(phase_rank)
        risk_adjusted_sales += (
            float(program.tam_estimate or 0.0)
            * share_cap
            * float(program.pos_prior or 0.0)
            * competition_adjustment
        )
    return float(risk_adjusted_sales)


def _peer_anchor_value(
    snapshot: CompanySnapshot,
    state: str,
    peer_context: dict[str, Any],
) -> float:
    metric_label = str(peer_context.get("metric_label") or "")
    median_multiple = float(peer_context.get("median_multiple") or 0.0)
    if median_multiple <= 0.0:
        return 0.0
    if state == "pre_commercial" and metric_label == "market-cap/TAM":
        top_tam = max((float(program.tam_estimate or 0.0) for program in snapshot.programs), default=0.0)
        return float(top_tam * median_multiple)
    revenue = float(snapshot.revenue or 0.0)
    if revenue <= 0.0 or metric_label != "EV/revenue":
        return 0.0
    return float(revenue * median_multiple)


def build_expectation_lens(
    snapshot: CompanySnapshot,
    signal: SignalArtifact,
    primary_event: CatalystEvent | None,
    peer_context: dict[str, Any],
) -> dict[str, Any]:
    profile = build_snapshot_profile(snapshot)
    state = str(profile["company_state"])
    competition_intensity = float(profile["competition_intensity"])
    net_cash = float(snapshot.cash or 0.0) - float(snapshot.debt or 0.0)
    market_cap = max(float(snapshot.market_cap or 0.0), 1.0)
    current_price = float(snapshot.metadata.get("price_now") or 0.0)
    peer_anchor_value = _peer_anchor_value(snapshot, state, peer_context)
    risk_adjusted_sales = _risk_adjusted_program_sales(snapshot, competition_intensity)
    cash_floor_value = max(net_cash, 0.0) + (0.15 * float(snapshot.revenue or 0.0))
    launch_progress = float(profile.get("launch_progress_pct", 0.0) or 0.0)
    lifecycle_score = float(profile.get("lifecycle_management_score", 0.0) or 0.0)
    pipeline_optionality = float(profile.get("pipeline_optionality_score", 0.0) or 0.0)
    capital_deployment = float(profile.get("capital_deployment_score", 0.0) or 0.0)
    exact_event_bonus = 0.15 if primary_event is not None and primary_event.status.startswith("exact_") else 0.0

    if state == "pre_commercial":
        empirical_value = risk_adjusted_sales * (2.1 + exact_event_bonus)
        internal_value = (
            (0.45 * peer_anchor_value if peer_anchor_value > 0 else 0.0)
            + (0.55 * empirical_value)
            + cash_floor_value
        )
        value_method = "stage peer market-cap/TAM blended with risk-adjusted lead-program opportunity"
    elif state == "commercial_launch":
        peer_multiple = float(peer_context.get("median_multiple") or 5.0)
        commercial_anchor = float(snapshot.revenue or 0.0) * _clamp(peer_multiple * (0.80 + (0.30 * launch_progress)), 2.5, 9.0)
        pipeline_anchor = risk_adjusted_sales * (1.9 + (0.2 * lifecycle_score))
        # Additive weights must sum to ≤ 1.0 across the main components to avoid
        # double-counting: the peer multiple already reflects pipeline optionality
        # for most commercial biotechs.  Old weights (0.55 + 0.45 + 0.35 = 1.35)
        # inflated internal_value by ~20-30% for launch-stage names, causing the
        # portfolio constructor to systematically overweight them.
        if peer_anchor_value > 0:
            internal_value = (
                (0.50 * peer_anchor_value)
                + (0.35 * commercial_anchor)
                + (0.15 * pipeline_anchor)
                + cash_floor_value
            )
        else:
            internal_value = (
                (0.65 * commercial_anchor)
                + (0.35 * pipeline_anchor)
                + cash_floor_value
            )
        value_method = "peer EV/revenue anchored launch value with lifecycle and pipeline support"
    else:
        peer_multiple = float(peer_context.get("median_multiple") or 4.0)
        franchise_anchor = float(snapshot.revenue or 0.0) * _clamp(peer_multiple * (0.90 + (0.15 * capital_deployment)), 2.0, 8.0)
        pipeline_anchor = risk_adjusted_sales * (1.4 + (0.25 * pipeline_optionality))
        # Same principle: peer anchor for commercialized names already prices in
        # pipeline.  Old weights (0.70 + 0.30 + 0.25 = 1.25) overstated value.
        if peer_anchor_value > 0:
            internal_value = (
                (0.65 * peer_anchor_value)
                + (0.20 * franchise_anchor)
                + (0.15 * pipeline_anchor)
                + (0.75 * max(net_cash, 0.0))
            )
        else:
            internal_value = (
                (0.70 * franchise_anchor)
                + (0.30 * pipeline_anchor)
                + (0.75 * max(net_cash, 0.0))
            )
        value_method = "franchise peer EV/revenue with pipeline and capital deployment optionality"

    internal_upside_pct = _clamp((internal_value / market_cap) - 1.0, -0.85, 2.50)
    internal_price_target = current_price * (internal_value / market_cap) if current_price > 0 else None
    setup_type = classify_setup_type(snapshot, signal, primary_event, profile)
    valuation_posture = peer_context.get("valuation_posture", "neutral")
    peer_gap_pct = (
        ((peer_anchor_value / market_cap) - 1.0)
        if peer_anchor_value > 0
        else internal_upside_pct
    )
    asymmetry_label = (
        "high positive asymmetry"
        if internal_upside_pct >= 0.35
        else "moderate positive asymmetry"
        if internal_upside_pct >= 0.15
        else "balanced / fairly priced"
        if internal_upside_pct > -0.10
        else "negative asymmetry"
    )
    market_view = {
        "pre_commercial": "The market is mostly debating probability and differentiation, not just whether the mechanism works.",
        "commercial_launch": "The market is mostly debating launch speed, lifecycle expansion, and competitive staying power.",
        "commercialized": "The market is mostly debating franchise durability, optionality, and capital deployment.",
    }[state]
    if setup_type == "asymmetry_without_near_term_catalyst":
        market_view += " This is more of an asymmetry / expectation setup than a clean dated catalyst trade."
    if setup_type == "sentiment_floor":
        market_view += " The current downside is partly cushioned by balance-sheet and franchise support."
    if setup_type in {"hard_catalyst", "soft_catalyst"}:
        market_view += " A meaningful event window still matters to closing the gap."
    if setup_type == "capital_allocation" and event_type_bucket(primary_event.event_type if primary_event is not None else signal.primary_event_type) == "strategic":
        market_view += " A meaningful strategic event window still matters to closing the gap."

    asymmetry_summary = (
        f"Peer-anchored value view is {internal_upside_pct * 100:+.1f}% versus current market cap; "
        f"peer gap screens at {peer_gap_pct * 100:+.1f}%, floor support at {float(profile['floor_support_pct']) * 100:.1f}%, "
        f"and peer posture looks {valuation_posture}."
    )
    competitive_summary = (
        f"{profile['primary_indication']} sits in a {profile['market_style']} market led by "
        f"{', '.join(profile['market_leaders']) if profile['market_leaders'] else 'mixed peers'}; "
        f"the differentiation bar is {profile['differentiation_focus']}."
    )
    return {
        **profile,
        "setup_type": setup_type,
        "internal_value": float(internal_value),
        "internal_price_target": None if internal_price_target is None else float(internal_price_target),
        "internal_upside_pct": float(internal_upside_pct),
        "peer_anchor_value": float(peer_anchor_value),
        "peer_gap_pct": float(peer_gap_pct),
        "risk_adjusted_sales": float(risk_adjusted_sales),
        "cash_floor_value": float(cash_floor_value),
        "value_method": value_method,
        "asymmetry_label": asymmetry_label,
        "market_view": market_view,
        "asymmetry_summary": asymmetry_summary,
        "competitive_summary": competitive_summary,
    }
