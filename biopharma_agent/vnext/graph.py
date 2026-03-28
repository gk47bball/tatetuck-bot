from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from prepare import classify_phase, gather_company_data
from strategy import PHASE_BASE, _estimate_indication_tam

from .entities import (
    ApprovedProduct,
    CatalystEvent,
    CompanySnapshot,
    EvidenceSnippet,
    FinancingEvent,
    Program,
    Trial,
)


MODALITY_KEYWORDS = {
    "gene editing": ("crispr", "editing", "base edit", "prime edit"),
    "gene therapy": ("aav", "gene therapy", "transgene"),
    "cell therapy": ("car-t", "car t", "nk cell", "cell therapy"),
    "antibody": ("antibody", "mab", "bispecific"),
    "small molecule": ("small molecule", "kinase", "oral"),
    "vaccine": ("vaccine", "mrna", "messenger rna"),
    "rna": ("rna", "sirna", "oligo", "as0", "antisense"),
}

PHASE_HORIZONS = {
    "APPROVED": 45,
    "NDA_BLA": 120,
    "PHASE3": 120,
    "PHASE2": 180,
    "PHASE1": 270,
    "EARLY_PHASE1": 360,
}


def infer_modality(text: str) -> str:
    text_lc = text.lower()
    for modality, keywords in MODALITY_KEYWORDS.items():
        if any(keyword in text_lc for keyword in keywords):
            return modality
    return "platform"


def infer_runway_months(revenue: float, cash: float, debt: float, num_trials: int) -> float:
    gross_burn = 25_000_000 + (num_trials * 6_000_000)
    net_burn = max(10_000_000, gross_burn - max(revenue * 0.15, 0.0))
    net_cash = cash - debt
    if net_cash <= 0:
        return 0.0
    return (net_cash / net_burn) * 12.0


def _build_catalyst(program_id: str | None, event_type: str, title: str, horizon_days: int, probability: float, importance: float, crowdedness: float, as_of: datetime) -> CatalystEvent:
    expected_date = as_of + timedelta(days=horizon_days)
    event_id = f"{program_id or 'company'}:{event_type}:{horizon_days}"
    return CatalystEvent(
        event_id=event_id,
        program_id=program_id,
        event_type=event_type,
        title=title,
        expected_date=expected_date.date().isoformat(),
        horizon_days=horizon_days,
        probability=probability,
        importance=importance,
        crowdedness=crowdedness,
    )


def build_company_snapshot(raw: dict[str, Any], as_of: datetime | None = None) -> CompanySnapshot:
    as_of = as_of or datetime.now(timezone.utc)
    finance = raw.get("finance", {})
    trials_by_program: dict[str, list[Trial]] = defaultdict(list)
    conditions_by_program: dict[str, set[str]] = defaultdict(set)
    evidence: list[EvidenceSnippet] = []

    for paper in raw.get("pubmed_papers", []):
        evidence.append(
            EvidenceSnippet(
                source="pubmed",
                source_id=paper.get("pmid") or "unknown",
                title=paper.get("title") or "PubMed abstract",
                excerpt=(paper.get("abstract") or "")[:280],
                confidence=0.65,
            )
        )

    for trial in raw.get("trials", []):
        interventions = [item for item in trial.get("interventions", []) if item]
        program_name = interventions[0] if interventions else (trial.get("title") or "unmapped-program")
        phase = classify_phase(trial.get("phase", []))
        trial_entity = Trial(
            trial_id=trial.get("nct_id") or program_name,
            title=trial.get("title") or program_name,
            phase=phase,
            status=trial.get("overall_status") or "UNKNOWN",
            conditions=list(trial.get("conditions", [])),
            interventions=interventions,
            enrollment=int(trial.get("enrollment") or 0),
            primary_outcomes=list(trial.get("primary_outcomes", [])),
        )
        trials_by_program[program_name].append(trial_entity)
        conditions_by_program[program_name].update(trial_entity.conditions)

    programs: list[Program] = []
    company_catalysts: list[CatalystEvent] = []
    for idx, (program_name, program_trials) in enumerate(trials_by_program.items(), start=1):
        text = f"{program_name} {' '.join(conditions_by_program[program_name])} {finance.get('description') or ''}"
        modality = infer_modality(text)
        phase = max(program_trials, key=lambda item: PHASE_BASE.get(item.phase, 0.01)).phase
        pos_prior = min(0.99, PHASE_BASE.get(phase, 0.07) + min(len(program_trials) * 0.03, 0.12))
        conditions = sorted(conditions_by_program[program_name])
        tam_estimate = float(_estimate_indication_tam(conditions))

        program_catalysts = [
            _build_catalyst(
                program_id=f"{raw.get('ticker')}:{idx}",
                event_type="clinical_readout" if phase != "APPROVED" else "commercial_update",
                title=f"{program_name} next milestone",
                horizon_days=PHASE_HORIZONS.get(phase, 180),
                probability=0.45 + min(pos_prior * 0.4, 0.35),
                importance=0.55 + min(tam_estimate / 10_000_000_000, 0.35),
                crowdedness=0.20 if modality in {"rare disease", "gene editing"} else 0.35,
                as_of=as_of,
            )
        ]
        company_catalysts.extend(program_catalysts)
        programs.append(
            Program(
                program_id=f"{raw.get('ticker')}:{idx}",
                name=program_name,
                modality=modality,
                phase=phase,
                conditions=conditions,
                trials=sorted(program_trials, key=lambda item: item.enrollment, reverse=True),
                pos_prior=pos_prior,
                tam_estimate=tam_estimate,
                catalyst_events=program_catalysts,
                evidence=evidence[:3],
            )
        )

    revenue = float(finance.get("totalRevenue") or 0.0)
    approved_products: list[ApprovedProduct] = []
    if revenue > 10_000_000:
        approved_products.append(
            ApprovedProduct(
                product_id=f"{raw.get('ticker')}:commercial",
                name=f"{raw.get('ticker')} commercial franchise",
                indication=(raw.get("conditions") or ["Commercial portfolio"])[0],
                annual_revenue=revenue,
                growth_signal=float(finance.get("momentum_3mo") or 0.0),
            )
        )
        company_catalysts.append(
            _build_catalyst(
                program_id=None,
                event_type="earnings",
                title=f"{raw.get('ticker')} quarterly commercial update",
                horizon_days=45,
                probability=0.95,
                importance=0.60,
                crowdedness=0.50,
                as_of=as_of,
            )
        )

    cash = float(finance.get("cash") or 0.0)
    debt = float(finance.get("debt") or 0.0)
    runway_months = infer_runway_months(revenue, cash, debt, raw.get("num_trials", 0))
    financing_events: list[FinancingEvent] = []
    if runway_months < 18:
        financing_events.append(
            FinancingEvent(
                event_id=f"{raw.get('ticker')}:financing",
                event_type="expected_financing",
                probability=0.80 if runway_months < 12 else 0.45,
                horizon_days=90 if runway_months < 12 else 180,
                expected_dilution_pct=0.18 if runway_months < 12 else 0.08,
                summary=f"Estimated runway is {runway_months:.1f} months.",
            )
        )

    snapshot_metadata = {
        "best_phase": raw.get("best_phase"),
        "num_trials": raw.get("num_trials", 0),
        "num_papers": raw.get("num_papers", 0),
        "runway_months": runway_months,
        "data_source": "prepare_compatibility_layer",
        "description": finance.get("description"),
    }

    return CompanySnapshot(
        ticker=raw.get("ticker") or "UNKNOWN",
        company_name=raw.get("company_name") or raw.get("ticker") or "Unknown",
        as_of=as_of.isoformat(),
        market_cap=float(finance.get("marketCap") or 0.0),
        enterprise_value=float(finance.get("enterpriseValue") or finance.get("marketCap") or 0.0),
        revenue=revenue,
        cash=cash,
        debt=debt,
        momentum_3mo=finance.get("momentum_3mo"),
        trailing_6mo_return=finance.get("trailing_6mo_return"),
        volatility=finance.get("volatility"),
        programs=programs,
        approved_products=approved_products,
        catalyst_events=company_catalysts,
        financing_events=financing_events,
        evidence=evidence,
        metadata=snapshot_metadata,
    )


def fetch_legacy_snapshot(ticker: str, company_name: str | None = None) -> dict[str, Any]:
    return gather_company_data(ticker, company_name or ticker)
