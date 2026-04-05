from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import math
import re
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
from .market_profile import update_snapshot_profile
from .taxonomy import event_timing_priority, program_event_type_for_phase


READOUT_KEYWORDS = (
    "results",
    "data",
    "met",
    "failed",
    "discontinued",
    "topline",
    "top-line",
    "interim",
    "readout",
)

EVIDENCE_STOPWORDS = {
    "biopharma",
    "biotech",
    "biosciences",
    "company",
    "corp",
    "corporation",
    "inc",
    "medicine",
    "medicines",
    "pharma",
    "pharmaceutical",
    "pharmaceuticals",
    "therapeutic",
    "therapeutics",
}

GENERIC_INTERVENTION_NAMES = {
    "no intervention",
    "observation",
    "observational",
    "placebo",
    "screening",
    "standard of care",
    "usual care",
}

GENERIC_PROGRAM_NAMES = {
    "screening",
    "unmapped-program",
    *GENERIC_INTERVENTION_NAMES,
}

ASSET_CODE_RE = re.compile(r"\b[A-Z]{1,6}-[A-Z0-9]*\d[A-Z0-9]*\b")
DOSAGE_FORM_RE = re.compile(r"\b\d+\s*mg\b|\boral tablet\b|\bintravenous\b|\bsubcutaneous\b", re.IGNORECASE)

# Stale synthetic: a catalyst the model believes is upcoming, but an 8-K
# suggests the trial read out recently.  We keep the event but push horizon
# out to 365 days so it does not pollute near-term catalyst signal.
STALE_SYNTHETIC_HORIZON_DAYS = 365
# An 8-K filed within this many days of as_of is considered "recent enough"
# to indicate the trial has already read out.
STALE_RECENCY_WINDOW_DAYS = 180
# Only flag a catalyst as stale if the model currently thinks it is upcoming
# (horizon_days > this threshold).  Events already placed far out are not
# phantom catalysts.
STALE_MIN_HORIZON_DAYS = 30


MODALITY_KEYWORDS = {
    "gene editing": ("crispr", "editing", "base edit", "prime edit"),
    "gene therapy": ("aav", "gene therapy", "transgene"),
    "cell therapy": ("car-t", "car t", "nk cell", "cell therapy"),
    "antibody": ("antibody", "mab", "bispecific"),
    "small molecule": ("small molecule", "kinase", "oral"),
    "vaccine": ("vaccine", "mrna", "messenger rna"),
    "rna": ("rna", "sirna", "oligo", "as0", "antisense"),
}

LOW_SIGNAL_TRIAL_KEYWORDS = (
    "awareness",
    "education",
    "educational",
    "implementation",
    "non interventional",
    "observational",
    "natural history",
    "long-term follow-up",
    "long term follow-up",
    "screening study",
    "screening",
    "specimen collection",
    "expanded access",
    "registry",
    "follow-up study",
    "healthy volunteer",
    "healthy volunteers",
    "patient survey",
    "patient preferences",
    "quality of life",
    "quality-of-life",
    "imaging",
    "de-escalation",
    "de escalation",
    "pre-clinical stage",
    "preclinical stage",
    "questionnaire",
    "peer educator",
    "peer educators",
    "school",
)

OBSERVATIONAL_OUTCOME_KEYWORDS = (
    "acceptability",
    "adherence",
    "awareness",
    "feasibility",
    "implementation",
    "number of participants",
    "number of patients",
    "prevalence",
    "questionnaire",
    "recruitment",
    "screening",
    "survey",
    "uptake",
)

APPROVED_PRODUCT_REGISTRY = {
    "AMGN": [
        ("TAVNEOS (avacopan)", "ANCA-associated vasculitis"),
    ],
    "APLS": [
        ("Syfovre", "geographic atrophy"),
        ("Empaveli", "PNH / C3 glomerulopathy"),
    ],
    "ABBV": [
        ("Skyrizi (risankizumab)", "Crohn's disease / plaque psoriasis"),
        ("Venclexta (venetoclax)", "hematologic malignancies"),
    ],
    "BMRN": [
        ("VOXZOGO", "achondroplasia"),
        ("VIMIZIM", "Morquio A syndrome"),
        ("NAGLAZYME", "MPS VI"),
        ("PALYNZIQ", "phenylketonuria"),
        ("ALDURAZYME", "MPS I"),
        ("BRINEURA", "CLN2 disease"),
        ("KUVAN", "phenylketonuria"),
    ],
    "CRSP": [
        ("CASGEVY", "sickle cell disease / transfusion-dependent beta-thalassemia"),
    ],
    "EXAS": [
        ("Cologuard", "colorectal cancer screening"),
        ("Oncotype DX", "breast cancer recurrence testing"),
    ],
    "GILD": [
        ("Sunlenca (lenacapavir)", "HIV-1 infection"),
        ("Epclusa (sofosbuvir/velpatasvir)", "chronic hepatitis C"),
        ("Tecartus (KTE-X19)", "B-cell malignancies"),
    ],
    "IOVA": [
        ("Amtagvi", "advanced melanoma"),
    ],
    "MDGL": [
        ("Rezdiffra", "MASH / NASH"),
    ],
    "MRNA": [
        ("Spikevax", "COVID-19"),
    ],
    "NVAX": [
        ("Nuvaxovid", "COVID-19"),
    ],
    "PTCT": [
        ("Sephience", "phenylketonuria"),
    ],
    "SRPT": [
        ("ELEVIDYS", "Duchenne muscular dystrophy"),
        ("EXONDYS 51", "Duchenne muscular dystrophy"),
        ("VYONDYS 53", "Duchenne muscular dystrophy"),
        ("AMONDYS 45", "Duchenne muscular dystrophy"),
    ],
}

DRIVER_OVERRIDE_REGISTRY: dict[str, dict[str, Any]] = {
    "APLS": {
        "label": "Pending Biogen acquisition",
        "indication": "$41 cash + CVR",
        "effective_on": "2026-04-01",
    },
    "ARVN": {
        "label": "vepdegestrant",
        "indication": "ER+/HER2- advanced breast cancer",
        "effective_on": "2026-02-01",
    },
    "NVAX": {
        "label": "Matrix-M platform",
        "indication": "partnered vaccines / Sanofi royalties",
        "effective_on": "2025-10-07",
    },
    "PTCT": {
        "label": "Sephience + votoplam",
        "indication": "PKU / Huntington disease optionality",
        "effective_on": "2025-07-29",
    },
}

PRIMARY_INDICATION_OVERRIDES: dict[str, dict[str, Any]] = {
    "ARVN": {"value": "Advanced Breast Cancer", "effective_on": "2026-02-01"},
    "NVAX": {"value": "partnered vaccine platform", "effective_on": "2025-10-07"},
    "PTCT": {"value": "Phenylketonuria", "effective_on": "2025-07-29"},
}

COMPANY_STATE_OVERRIDES: dict[str, dict[str, Any]] = {
    "ARVN": {"value": "pre_commercial", "effective_on": "2026-02-01"},
}

SPECIAL_SITUATION_OVERRIDES: dict[str, dict[str, Any]] = {
    "APLS": {
        "value": "pending_transaction",
        "label": "pending transaction",
        "reason": "Announced Biogen acquisition caps standalone upside largely to the cash consideration, CVR, and deal spread.",
        "effective_on": "2026-04-01",
        "expires_on": "2026-09-22",
    },
    "NVAX": {
        "value": "partnered_royalty_transition",
        "label": "partnered royalty transition",
        "reason": "Commercial economics are shifting toward Sanofi-led milestones and Matrix-M royalties rather than a standalone vaccine launch.",
        "effective_on": "2025-10-07",
    },
    "ARVN": {
        "value": "partner_search_overhang",
        "label": "partner search overhang",
        "reason": (
            "Arvinas and Pfizer are seeking a third-party commercialization partner for vepdegestrant after the "
            "VERITAC-2 program supported a narrower ESR1-mutant opportunity and commercialization roles were reduced."
        ),
        "effective_on": "2025-09-17",
        "risk_flags": [
            "seeking third-party commercialization partner for vepdegestrant",
            "VERITAC-2 missed PFS significance in the overall intent-to-treat population",
            "commercial workforce was reduced after pipeline reprioritization",
        ],
        "confidence_haircut": 0.78,
        "expected_return_haircut": 0.65,
        "catalyst_success_prob_haircut": 0.88,
    },
    "AMGN": {
        "value": "regulatory_overhang",
        "label": "TAVNEOS regulatory overhang",
        "reason": (
            "FDA requested voluntary withdrawal of TAVNEOS (avacopan) in January 2026 and Amgen has refused, "
            "leaving a live regulatory overhang on the franchise."
        ),
        "effective_on": "2026-01-16",
        "risk_flags": [
            "FDA requested voluntary TAVNEOS withdrawal in January 2026",
            "avacopan now carries regulatory dispute risk, not just routine commercial execution risk",
        ],
    },
}

CURATED_EVENT_OVERRIDES: dict[str, list[dict[str, Any]]] = {
    "APLS": [
        {
            "event_type": "strategic_transaction",
            "title": "Biogen acquisition agreement pending under merger terms",
            "expected_date": "2026-09-22",
            "probability": 0.90,
            "importance": 0.98,
            "crowdedness": 0.18,
            "status": "guided_company_event",
            "source": "company_curated",
            "effective_on": "2026-04-01",
            "expires_on": "2026-09-22",
        }
    ],
    "ARVN": [
        {
            "event_type": "pdufa",
            "title": "Vepdegestrant PDUFA decision in ER+/HER2- metastatic breast cancer",
            "expected_date": "2026-06-05",
            "probability": 0.88,
            "importance": 0.98,
            "crowdedness": 0.42,
            "status": "exact_company_calendar",
            "source": "company_curated",
            "effective_on": "2026-02-01",
        }
    ],
}

PROGRAM_CURATION_RULES: dict[str, list[dict[str, Any]]] = {
    "ARVN": [
        {
            "match_any": ("vepdegestrant", "arv-471", "pf-07850327"),
            "name": "vepdegestrant",
            "conditions": ["ER+/HER2- advanced or metastatic breast cancer with ESR1 mutation"],
        },
    ],
    "AMGN": [
        {
            "match_any": ("familial hypercholesterolemia canada", "hypercholesterolemie familiale canada"),
            "exclude": True,
        },
        {
            "match_any": ("rocatinlimab",),
            "name": "Rocatinlimab",
            "phase": "PHASE3",
            "conditions": ["Moderate-to-severe atopic dermatitis"],
        },
        {
            "match_any": ("avacopan", "tavneos"),
            "name": "Avacopan",
            "phase": "APPROVED",
            "conditions": ["Antineutrophil Cytoplasmic Antibody-associated Vasculitis"],
        },
    ],
    "GILD": [
        {"match_any": ("dolutegravir", "tivicay"), "exclude": True},
        {"match_any": ("digital health coaching program", "hiv team implementation", "prep awareness and uptake educational program"), "exclude": True},
        {
            "match_any": ("sofosbuvir/velpatasvir", "epclusa"),
            "name": "Sofosbuvir/Velpatasvir",
            "phase": "APPROVED",
            "conditions": ["Chronic hepatitis C"],
        },
        {
            "match_any": ("lenacapavir", "sunlenca"),
            "name": "Lenacapavir",
            "phase": "APPROVED",
            "conditions": ["HIV-1 infection"],
        },
        {
            "match_any": ("kte-x19", "tecartus"),
            "name": "KTE-X19",
            "conditions": [
                "Relapsed/Refractory Mantle Cell Lymphoma",
                "Relapsed/Refractory B-precursor Acute Lymphoblastic Leukemia",
            ],
        },
    ],
    "EXAS": [
        {"match_any": ("study ct/mri imaging", "patient survey", "de-escalation"), "exclude": True},
        {
            "match_any": ("cologuard",),
            "name": "Cologuard",
            "phase": "APPROVED",
            "conditions": ["Colorectal cancer screening"],
        },
        {
            "match_any": ("mrd", "ctdna", "b-64"),
            "name": "ctDNA MRD platform",
            "phase": "APPROVED",
            "conditions": ["Molecular residual disease monitoring"],
        },
    ],
    "ABBV": [
        {
            "match_any": (
                "the ccp study: coordinated programme to prevent arthritis",
                "quality of life of risankizumab",
            ),
            "exclude": True,
        },
        {"match_any": ("risankizumab", "skyrizi"), "name": "Risankizumab", "phase": "APPROVED"},
    ],
    "EDIT": [
        {"match_any": ("safety and efficacy assessments",), "exclude": True},
    ],
}

PHASE_HORIZONS = {
    "APPROVED": 45,
    "NDA_BLA": 120,
    "PHASE3": 120,
    "PHASE2": 180,
    "PHASE1": 270,
    "EARLY_PHASE1": 360,
}

TRIAL_STATUS_PRIORITY = {
    "ACTIVE_NOT_RECRUITING": 1.00,
    "RECRUITING": 0.96,
    "ENROLLING_BY_INVITATION": 0.90,
    "NOT_YET_RECRUITING": 0.82,
    "COMPLETED": 0.62,
    "AVAILABLE": 0.58,
    "SUSPENDED": 0.18,
    "WITHDRAWN": 0.05,
    "TERMINATED": 0.0,
}


def infer_modality(text: str) -> str:
    text_lc = text.lower()
    for modality, keywords in MODALITY_KEYWORDS.items():
        if any(keyword in text_lc for keyword in keywords):
            return modality
    return "platform"


def infer_runway_months(revenue: float, cash: float, debt: float, num_trials: int) -> float:
    gross_burn = 18_000_000 + (num_trials * 8_000_000)
    revenue_offset = min(max(revenue * 0.04, 0.0), gross_burn * 0.55)
    net_burn = gross_burn - revenue_offset
    if revenue > 500_000_000:
        net_burn = max(net_burn, 120_000_000)
    elif revenue > 100_000_000:
        net_burn = max(net_burn, 75_000_000)
    elif revenue > 25_000_000:
        net_burn = max(net_burn, 40_000_000)
    else:
        net_burn = max(net_burn, 15_000_000)
    net_cash = cash - debt
    if net_cash <= 0:
        return 0.0
    return min((net_cash / net_burn) * 12.0, 120.0)


def _trial_status_priority(status: str) -> float:
    return TRIAL_STATUS_PRIORITY.get(str(status or "").upper(), 0.45)


def _trial_endpoint_priority(primary_outcomes: list[str]) -> float:
    outcome_text = " ".join(str(item or "") for item in primary_outcomes).lower()
    if any(marker in outcome_text for marker in ("overall survival", "death from any")):
        return 0.95
    if any(marker in outcome_text for marker in ("progression-free survival", "event-free survival", "relapse-free")):
        return 0.82
    if any(marker in outcome_text for marker in ("hemoglobin", "vaso-occlusive", "transfusion independence", "bleed")):
        return 0.84
    if any(marker in outcome_text for marker in ("forced expiratory", "lung function", "forced vital capacity", "fev1")):
        return 0.78
    if any(marker in outcome_text for marker in ("functional independence", "activities of daily living", "disability")):
        return 0.65
    if any(marker in outcome_text for marker in ("objective response", "response rate", "complete response", "partial response")):
        return 0.52
    if any(marker in outcome_text for marker in ("pharmacokinetic", "pharmacodynamic", "biomarker", "dose-limiting")):
        return 0.30
    if any(marker in outcome_text for marker in ("safety", "adverse", "tolerability", "maximum tolerated")):
        return 0.25
    return 0.42


def _trial_scientific_priority(trial: Trial) -> tuple[float, float, float, float, float]:
    title = str(trial.title or "").lower()
    phase_score = PHASE_BASE.get(trial.phase, 0.01)
    endpoint_score = _trial_endpoint_priority(trial.primary_outcomes)
    status_score = _trial_status_priority(trial.status)
    pivotal_bonus = 0.08 if any(marker in title for marker in ("pivotal", "registrational")) else 0.0
    extension_penalty = -0.12 if any(marker in title for marker in ("extension", "follow-up", "follow up")) else 0.0
    enrollment_score = math.log10(max(int(trial.enrollment or 0), 0) + 1.0)
    return (
        phase_score + pivotal_bonus + extension_penalty,
        status_score,
        endpoint_score,
        enrollment_score,
        float(len(trial.primary_outcomes)),
    )


def select_lead_trial(program: Program) -> Trial | None:
    if not program.trials:
        return None
    program_phase_score = PHASE_BASE.get(program.phase, 0.01)
    return max(
        program.trials,
        key=lambda trial: (
            -abs(PHASE_BASE.get(trial.phase, 0.01) - program_phase_score),
            *_trial_scientific_priority(trial),
        ),
    )


def _evidence_tokens(*values: str) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        for token in re.findall(r"[a-z0-9][a-z0-9-]{1,}", str(value or "").lower()):
            normalized = token.strip("-")
            if normalized in EVIDENCE_STOPWORDS:
                continue
            if len(normalized) >= 4 or any(ch.isdigit() for ch in normalized):
                tokens.add(normalized)
    return tokens


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _clean_intervention_name(name: str) -> str:
    cleaned = re.sub(r"^(placebo|standard of care)\s+(for|plus)\s+", "", str(name or "").strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -")
    return cleaned or str(name or "").strip()


def _curation_rule_for(ticker: str, *values: Any) -> dict[str, Any] | None:
    text = " ".join(_normalize_text(value) for value in values if value).strip()
    if not text:
        return None
    for rule in PROGRAM_CURATION_RULES.get(ticker.upper(), []):
        if any(str(token).lower() in text for token in rule.get("match_any", ())):
            return rule
    return None


def _extract_asset_code(*values: str) -> str | None:
    for value in values:
        match = ASSET_CODE_RE.search(str(value or "").upper())
        if match:
            return match.group(0)
    return None


def _is_generic_intervention(name: str) -> bool:
    normalized = " ".join(str(name or "").strip().lower().split())
    if not normalized:
        return True
    if normalized in GENERIC_INTERVENTION_NAMES:
        return True
    return any(marker in normalized for marker in ("placebo", "standard of care"))


def _normalize_trial_phase(phase_list: list[str]) -> str:
    joined = " ".join(str(item or "") for item in phase_list).upper()
    if "PHASE4" in joined or "PHASE 4" in joined:
        return "APPROVED"
    return classify_phase(phase_list)


def _derive_program_name(trial: dict[str, Any]) -> str:
    title = str(trial.get("title") or "").strip()
    interventions = [_clean_intervention_name(str(item).strip()) for item in trial.get("interventions", []) if str(item or "").strip()]

    asset_code = _extract_asset_code(*interventions, title)
    if asset_code:
        return asset_code

    for candidate in interventions:
        if _is_generic_intervention(candidate):
            continue
        if len(candidate) <= 80:
            return candidate

    return title or "unmapped-program"


def _program_match_tokens(program: Program) -> tuple[set[str], set[str], set[str]]:
    program_tokens = _evidence_tokens(program.name)
    condition_tokens = _evidence_tokens(" ".join(program.conditions))
    trial_tokens = _evidence_tokens(
        " ".join(trial.title for trial in program.trials),
        " ".join(" ".join(trial.interventions) for trial in program.trials),
        " ".join(" ".join(trial.primary_outcomes) for trial in program.trials),
    )
    return program_tokens, condition_tokens, trial_tokens


def select_program_evidence(
    program: Program,
    snapshot: CompanySnapshot | None = None,
    *,
    evidence_pool: list[EvidenceSnippet] | None = None,
    limit: int = 3,
) -> list[EvidenceSnippet]:
    candidates = list(evidence_pool if evidence_pool is not None else (snapshot.evidence if snapshot is not None else []))
    if not candidates:
        return []
    program_tokens, condition_tokens, trial_tokens = _program_match_tokens(program)
    normalized_program_name = re.sub(r"[^a-z0-9]+", "", str(program.name or "").lower())
    scored: list[tuple[float, float, EvidenceSnippet]] = []
    seen: set[tuple[str, str, str]] = set()
    for snippet in candidates:
        text = " ".join(
            [
                str(snippet.title or ""),
                str(snippet.excerpt or ""),
                str(snippet.source_id or ""),
                str(snippet.url or ""),
            ]
        ).lower()
        text_tokens = _evidence_tokens(text)
        normalized_text = re.sub(r"[^a-z0-9]+", "", text)
        shared_program = len(program_tokens & text_tokens)
        shared_condition = len(condition_tokens & text_tokens)
        shared_trial = len(trial_tokens & text_tokens)
        score = (shared_program * 3.0) + (shared_condition * 2.0) + (shared_trial * 1.5)
        if normalized_program_name and normalized_program_name in normalized_text:
            score += 4.0
        if str(snippet.source or "") == "pubmed" and shared_program <= 0 and normalized_program_name not in normalized_text:
            continue
        if score <= 0.0:
            continue
        dedupe_key = (str(snippet.source), str(snippet.source_id), str(snippet.title))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        scored.append((score, float(snippet.confidence or 0.0), snippet))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored[:limit]]


def canonical_program_name(program: Program) -> str:
    overlay = curated_program_overlay(program)
    if overlay.get("exclude"):
        return str(program.name or "").strip() or "unmapped-program"
    if overlay.get("name"):
        return str(overlay["name"])
    asset_code = _extract_asset_code(program.name)
    if asset_code:
        return asset_code
    lead_trial = select_lead_trial(program)
    if lead_trial is not None:
        trial_code = _extract_asset_code(
            lead_trial.title,
            " ".join(lead_trial.interventions),
        )
        if trial_code:
            return trial_code
    name = str(program.name or "").strip()
    if name:
        return name
    if lead_trial is not None and lead_trial.title:
        return str(lead_trial.title)
    return "unmapped-program"


def select_company_evidence(
    snapshot: CompanySnapshot,
    *,
    evidence_pool: list[EvidenceSnippet] | None = None,
    limit: int = 5,
) -> list[EvidenceSnippet]:
    candidates = list(evidence_pool if evidence_pool is not None else snapshot.evidence)
    if not candidates:
        return []

    company_tokens = _evidence_tokens(snapshot.company_name, str(snapshot.metadata.get("driver_label") or ""))
    ticker_token = str(snapshot.ticker or "").lower().strip()
    if len(ticker_token) >= 3:
        company_tokens.add(ticker_token)
    program_tokens: set[str] = set()
    condition_tokens: set[str] = set()
    for program in snapshot.programs:
        program_tokens.update(_evidence_tokens(canonical_program_name(program), program.name))
        condition_tokens.update(_evidence_tokens(" ".join(program.conditions)))
    for product in snapshot.approved_products:
        program_tokens.update(_evidence_tokens(product.name, product.indication))

    scored: list[tuple[float, float, EvidenceSnippet]] = []
    seen: set[tuple[str, str, str]] = set()
    for snippet in candidates:
        text = " ".join(
            [
                str(snippet.title or ""),
                str(snippet.excerpt or ""),
                str(snippet.source_id or ""),
                str(snippet.url or ""),
            ]
        ).lower()
        text_tokens = _evidence_tokens(text)
        shared_company = len(company_tokens & text_tokens)
        shared_program = len(program_tokens & text_tokens)
        shared_condition = len(condition_tokens & text_tokens)
        score = (shared_company * 2.5) + (shared_program * 3.0) + (shared_condition * 1.5)
        source = str(snippet.source or "")
        if source == "pubmed" and shared_company <= 0 and shared_program <= 0:
            continue
        if source == "sec":
            score += 5.0
        elif source in {"eodhd_news", "external_event", "company_curated", "press_release"}:
            score += 2.5
        elif source == "pubmed" and score <= 0.0:
            continue
        if score <= 0.0:
            continue
        dedupe_key = (str(snippet.source), str(snippet.source_id), str(snippet.title))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        scored.append((score, float(snippet.confidence or 0.0), snippet))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored[:limit]]


def refresh_program_evidence(snapshot: CompanySnapshot, limit: int = 3) -> CompanySnapshot:
    if not snapshot.programs or not snapshot.evidence:
        return snapshot
    for program in snapshot.programs:
        matched = select_program_evidence(program, snapshot, evidence_pool=snapshot.evidence, limit=limit)
        if matched:
            program.evidence = matched
    return snapshot


def refresh_snapshot_evidence(
    snapshot: CompanySnapshot,
    *,
    company_limit: int = 5,
    program_limit: int = 3,
) -> CompanySnapshot:
    if snapshot.evidence:
        snapshot.evidence = select_company_evidence(snapshot, evidence_pool=snapshot.evidence, limit=company_limit)
    return refresh_program_evidence(snapshot, limit=program_limit)


def _trial_text(trial: Trial) -> str:
    return " ".join(
        item
        for item in [
            trial.title,
            " ".join(trial.conditions),
            " ".join(trial.interventions),
            " ".join(trial.primary_outcomes),
        ]
        if item
    ).lower()


def _is_low_signal_trial(trial: Trial) -> bool:
    text = _trial_text(trial)
    if any(keyword in text for keyword in LOW_SIGNAL_TRIAL_KEYWORDS):
        return True

    has_asset_code = bool(_extract_asset_code(trial.title, " ".join(trial.interventions)))
    has_specific_intervention = any(not _is_generic_intervention(item) for item in trial.interventions)
    outcome_text = " ".join(str(item or "") for item in trial.primary_outcomes).lower()
    title_text = str(trial.title or "").lower()

    if not has_specific_intervention and not has_asset_code:
        if any(keyword in outcome_text for keyword in OBSERVATIONAL_OUTCOME_KEYWORDS):
            return True
        if any(keyword in title_text for keyword in ("natural history", "registry", "screening", "awareness", "implementation")):
            return True
    return False


def is_low_signal_program(program: Program) -> bool:
    overlay = curated_program_overlay(program)
    if overlay.get("exclude"):
        return True
    name_text = str(program.name or "").strip().lower()
    if not name_text:
        return True
    if name_text in GENERIC_PROGRAM_NAMES:
        return True
    if name_text.startswith("study "):
        return True
    if DOSAGE_FORM_RE.search(name_text) and not _extract_asset_code(program.name):
        return True
    if any(keyword in name_text for keyword in ("awareness", "education", "educational", "implementation", "screening")):
        return True
    if any(
        keyword in name_text
        for keyword in (
            "patient survey",
            "patient preferences",
            "quality of life",
            "questionnaire",
            "imaging",
            "de-escalation",
            "de escalation",
            "pre-clinical stage",
            "preclinical stage",
        )
    ):
        return True
    lead_trial = select_lead_trial(program)
    if lead_trial is None:
        return True
    return _is_low_signal_trial(lead_trial)


def curated_program_overlay(program: Program) -> dict[str, Any]:
    ticker = str(program.program_id or "").split(":", 1)[0].upper()
    lead_trial = select_lead_trial(program)
    return _curation_rule_for(
        ticker,
        program.name,
        " ".join(program.conditions),
        lead_trial.title if lead_trial is not None else "",
        " ".join(lead_trial.interventions) if lead_trial is not None else "",
    ) or {}


def _program_catalyst_title(program_name: str, phase: str, conditions: list[str]) -> str:
    phase_labels = {
        "APPROVED": "commercial update",
        "NDA_BLA": "regulatory decision",
        "PHASE3": "phase 3 readout",
        "PHASE2": "phase 2 readout",
        "PHASE1": "phase 1 update",
        "EARLY_PHASE1": "early-phase update",
    }
    label = phase_labels.get(phase, "clinical update")
    if conditions:
        return f"{program_name} {label} in {conditions[0]}"
    return f"{program_name} {label}"


def _approved_products_for_company(ticker: str, revenue: float, growth_signal: float) -> list[ApprovedProduct]:
    registered = APPROVED_PRODUCT_REGISTRY.get(ticker.upper(), [])
    if not registered:
        return []
    revenue_per_product = (revenue / len(registered)) if registered and revenue > 0 else 0.0
    return [
        ApprovedProduct(
            product_id=f"{ticker}:approved:{index}",
            name=name,
            indication=indication,
            annual_revenue=revenue_per_product,
            growth_signal=growth_signal,
        )
        for index, (name, indication) in enumerate(registered, start=1)
    ]


def _parse_override_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    if "T" not in normalized:
        normalized = f"{normalized}T00:00:00+00:00"
    elif normalized.endswith("Z"):
        normalized = normalized.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _override_is_active(config: dict[str, Any] | None, as_of: datetime) -> bool:
    if not config:
        return False
    reference = as_of.astimezone(timezone.utc) if as_of.tzinfo is not None else as_of.replace(tzinfo=timezone.utc)
    start = _parse_override_datetime(str(config.get("effective_on") or "")) if config.get("effective_on") else None
    end = _parse_override_datetime(str(config.get("expires_on") or "")) if config.get("expires_on") else None
    if start is not None and reference < start:
        return False
    if end is not None and reference > end:
        return False
    return True


def _sort_catalyst_events(events: list[CatalystEvent]) -> list[CatalystEvent]:
    return sorted(
        events,
        key=lambda event: (
            -event_timing_priority(event.status, event.expected_date, event.title),
            -event.importance,
            event.horizon_days,
            event.crowdedness,
        ),
    )


def apply_curated_company_overrides(snapshot: CompanySnapshot, as_of: datetime | None = None) -> CompanySnapshot:
    as_of = as_of or _parse_override_datetime(snapshot.as_of) or datetime.now(timezone.utc)
    ticker = snapshot.ticker.upper()

    if not snapshot.approved_products:
        snapshot.approved_products = _approved_products_for_company(ticker, snapshot.revenue, float(snapshot.momentum_3mo or 0.0))
    snapshot.metadata["approved_product_registry_hit"] = bool(snapshot.approved_products)

    state_override = COMPANY_STATE_OVERRIDES.get(ticker)
    if _override_is_active(state_override, as_of):
        snapshot.metadata["company_state_override"] = str(state_override["value"])

    indication_override = PRIMARY_INDICATION_OVERRIDES.get(ticker)
    if _override_is_active(indication_override, as_of):
        snapshot.metadata["primary_indication_override"] = str(indication_override["value"])

    driver_override = DRIVER_OVERRIDE_REGISTRY.get(ticker)
    if _override_is_active(driver_override, as_of):
        snapshot.metadata["driver_label"] = str(driver_override["label"])
        snapshot.metadata["driver_indication"] = str(driver_override.get("indication") or "")

    special_override = SPECIAL_SITUATION_OVERRIDES.get(ticker)
    if _override_is_active(special_override, as_of):
        snapshot.metadata["special_situation"] = str(special_override["value"])
        snapshot.metadata["special_situation_label"] = str(special_override.get("label") or str(special_override["value"]).replace("_", " "))
        snapshot.metadata["special_situation_reason"] = str(special_override.get("reason") or "")
        snapshot.metadata["bear_case_flags"] = list(special_override.get("risk_flags") or [])
        if special_override.get("confidence_haircut") is not None:
            snapshot.metadata["confidence_haircut"] = float(special_override["confidence_haircut"])
        if special_override.get("expected_return_haircut") is not None:
            snapshot.metadata["expected_return_haircut"] = float(special_override["expected_return_haircut"])
        if special_override.get("catalyst_success_prob_haircut") is not None:
            snapshot.metadata["catalyst_success_prob_haircut"] = float(special_override["catalyst_success_prob_haircut"])
        special_title = f"{ticker} special situation: {snapshot.metadata['special_situation_label']}"
        if not any(str(item.source_id or "") == f"{ticker}:special:{special_override['value']}" for item in snapshot.evidence):
            snapshot.evidence.append(
                EvidenceSnippet(
                    source="company_curated",
                    source_id=f"{ticker}:special:{special_override['value']}",
                    title=special_title,
                    excerpt=str(special_override.get("reason") or ""),
                    as_of=str(special_override.get("effective_on") or snapshot.as_of),
                    confidence=0.82,
                )
            )

    existing_keys = {
        (event.event_type, event.expected_date, event.title)
        for event in snapshot.catalyst_events
    }
    for override in CURATED_EVENT_OVERRIDES.get(ticker, []):
        if not _override_is_active(override, as_of):
            continue
        key = (str(override["event_type"]), str(override["expected_date"]), str(override["title"]))
        if key in existing_keys:
            continue
        expected_at = _parse_override_datetime(str(override["expected_date"]))
        if expected_at is None:
            continue
        reference = as_of.astimezone(timezone.utc) if as_of.tzinfo is not None else as_of.replace(tzinfo=timezone.utc)
        horizon_days = max((expected_at.date() - reference.date()).days, 0)
        snapshot.catalyst_events.append(
            CatalystEvent(
                event_id=f"{ticker}:curated:{override['event_type']}:{override['expected_date']}",
                program_id=None,
                event_type=str(override["event_type"]),
                title=str(override["title"]),
                expected_date=str(override["expected_date"]),
                horizon_days=horizon_days,
                probability=float(override.get("probability", 0.8) or 0.8),
                importance=float(override.get("importance", 0.75) or 0.75),
                crowdedness=float(override.get("crowdedness", 0.3) or 0.3),
                status=str(override.get("status") or "guided_company_event"),
                source=str(override.get("source") or "company_curated"),
                timing_exact=str(override.get("status") or "").startswith("exact"),
                timing_synthetic=False,
            )
        )
        existing_keys.add(key)

    if snapshot.catalyst_events:
        snapshot.catalyst_events = _sort_catalyst_events(snapshot.catalyst_events)
    return snapshot


def _build_catalyst(
    program_id: str | None,
    event_type: str,
    title: str,
    horizon_days: int,
    probability: float,
    importance: float,
    crowdedness: float,
    as_of: datetime,
    status: str = "anticipated",
) -> CatalystEvent:
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
        status=status,
    )


def _flag_stale_catalysts(
    catalyst_events: list[CatalystEvent],
    sec_events: list[dict],  # raw SEC event payloads; each dict must contain at least:
    #   "title"    : str  — filing title (e.g. "Form 8-K: Results of Operations")
    #   "summary"  : str  — short description / subject line
    #   "filed_at" : str  — ISO-8601 date/datetime of the filing
    as_of: str,
    program_names: list[str],
    conditions: list[str],
) -> list[CatalystEvent]:
    """Flag synthetic catalyst events that appear to have already occurred.

    ClinicalTrials.gov registrants are notoriously slow to update trial status.
    We cross-reference SEC 8-K filings: if a filing within the past
    ``STALE_RECENCY_WINDOW_DAYS`` days contains readout-style keywords that
    match a program name or condition, and the model still believes the catalyst
    is upcoming (horizon_days > ``STALE_MIN_HORIZON_DAYS``), we mark the event
    as ``status="stale_synthetic"`` and push ``horizon_days`` to
    ``STALE_SYNTHETIC_HORIZON_DAYS``.

    Stale events are *not* deleted so downstream diagnostics can inspect them.

    Args:
        catalyst_events: List of CatalystEvent objects to inspect.  Only events
            whose status is not an exact-timing status are candidates.
        sec_events: List of raw SEC filing dicts.  If empty or unavailable the
            function returns the list unchanged (graceful skip).
        as_of: ISO-8601 string representing the snapshot date.
        program_names: Drug / program names for the company (used for keyword
            matching against filing text).
        conditions: Disease / indication strings for the company (also used for
            keyword matching).

    Returns:
        The same list of CatalystEvent objects, potentially with some events
        mutated to status="stale_synthetic" and horizon_days adjusted.
    """
    if not sec_events:
        return catalyst_events

    try:
        as_of_dt = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return catalyst_events

    recency_cutoff = as_of_dt - timedelta(days=STALE_RECENCY_WINDOW_DAYS)

    # Build a normalised keyword set from program names + conditions.
    # We use short, distinctive tokens (length >= 4) to avoid spurious matches
    # on common English words.
    match_tokens: set[str] = set()
    for phrase in list(program_names) + list(conditions):
        for token in phrase.lower().split():
            token = token.strip("(),.-")
            if len(token) >= 4:
                match_tokens.add(token)

    # Pre-filter: identify SEC filings that (a) are recent and (b) contain at
    # least one readout keyword — these are the only filings that can trigger
    # the staleness flag.
    readout_filings: list[str] = []  # concatenated text of qualifying filings
    for filing in sec_events:
        filed_at_raw = filing.get("filed_at") or ""
        try:
            filed_dt = datetime.fromisoformat(str(filed_at_raw).replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue
        if filed_dt < recency_cutoff or filed_dt > as_of_dt:
            continue
        filing_text = " ".join(
            str(filing.get(key) or "")
            for key in ("title", "summary")
        ).lower()
        if any(kw in filing_text for kw in READOUT_KEYWORDS):
            readout_filings.append(filing_text)

    if not readout_filings:
        return catalyst_events

    # For each synthetic catalyst that the model believes is still upcoming,
    # check whether any recent readout filing mentions a matching token.
    exact_statuses = {"exact_sec_filing", "exact_company_calendar", "exact_press_release"}
    for event in catalyst_events:
        if event.status in exact_statuses:
            continue
        if event.horizon_days <= STALE_MIN_HORIZON_DAYS:
            continue
        # Build tokens for this specific event (title + program-level conditions
        # already covered by match_tokens, but also check the event title itself).
        event_tokens = set(match_tokens)
        for token in (event.title or "").lower().split():
            token = token.strip("(),.-")
            if len(token) >= 4:
                event_tokens.add(token)
        for filing_text in readout_filings:
            if any(tok in filing_text for tok in event_tokens):
                # Mutate in place — CatalystEvent uses dataclass(slots=True)
                # but slots does not prevent attribute assignment.
                object.__setattr__(event, "status", "stale_synthetic")
                object.__setattr__(event, "horizon_days", STALE_SYNTHETIC_HORIZON_DAYS)
                break  # one matching filing is enough

    return catalyst_events


def build_company_snapshot(raw: dict[str, Any], as_of: datetime | None = None) -> CompanySnapshot:
    as_of = as_of or datetime.now(timezone.utc)
    ticker = str(raw.get("ticker") or "").upper()
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
        interventions = [_clean_intervention_name(item) for item in trial.get("interventions", []) if item]
        program_name = _derive_program_name(trial)
        rule = _curation_rule_for(
            ticker,
            program_name,
            trial.get("title"),
            " ".join(interventions),
            " ".join(trial.get("conditions", [])),
        )
        if rule and rule.get("exclude"):
            continue
        if rule and rule.get("name"):
            program_name = str(rule["name"])
        phase = _normalize_trial_phase(list(trial.get("phase", [])))
        if rule and rule.get("phase"):
            phase = str(rule["phase"])
        conditions = list(rule.get("conditions") or trial.get("conditions", [])) if rule else list(trial.get("conditions", []))
        trial_entity = Trial(
            trial_id=trial.get("nct_id") or program_name,
            title=trial.get("title") or program_name,
            phase=phase,
            status=trial.get("overall_status") or "UNKNOWN",
            conditions=conditions,
            interventions=interventions,
            enrollment=int(trial.get("enrollment") or 0),
            primary_outcomes=list(trial.get("primary_outcomes", [])),
            locations=list(trial.get("locations", [])),
        )
        if _is_low_signal_trial(trial_entity):
            continue
        trials_by_program[program_name].append(trial_entity)
        conditions_by_program[program_name].update(trial_entity.conditions)

    programs: list[Program] = []
    company_catalysts: list[CatalystEvent] = []
    for idx, (program_name, program_trials) in enumerate(trials_by_program.items(), start=1):
        text = f"{program_name} {' '.join(conditions_by_program[program_name])} {finance.get('description') or ''}"
        modality = infer_modality(text)
        phase = max(program_trials, key=lambda item: PHASE_BASE.get(item.phase, 0.01)).phase
        rule = _curation_rule_for(ticker, program_name, " ".join(conditions_by_program[program_name]))
        if rule and rule.get("exclude"):
            continue
        if rule and rule.get("phase"):
            phase = str(rule["phase"])
        pos_prior = min(0.99, PHASE_BASE.get(phase, 0.07) + min(len(program_trials) * 0.03, 0.12))
        conditions = sorted(rule.get("conditions") or conditions_by_program[program_name]) if rule else sorted(conditions_by_program[program_name])
        tam_estimate = float(_estimate_indication_tam(conditions))

        program_catalysts = [
            _build_catalyst(
                program_id=f"{raw.get('ticker')}:{idx}",
                event_type=program_event_type_for_phase(phase),
                title=_program_catalyst_title(program_name, phase, conditions),
                horizon_days=PHASE_HORIZONS.get(phase, 180),
                probability=0.45 + min(pos_prior * 0.4, 0.35),
                importance=0.55 + min(tam_estimate / 10_000_000_000, 0.35),
                crowdedness=0.20 if modality in {"rare disease", "gene editing"} else 0.35,
                as_of=as_of,
                status="phase_timing_estimate",
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
                trials=sorted(program_trials, key=_trial_scientific_priority, reverse=True),
                pos_prior=pos_prior,
                tam_estimate=tam_estimate,
                catalyst_events=program_catalysts,
                evidence=evidence[:3],
            )
        )

    programs.sort(
        key=lambda program: (
            -PHASE_BASE.get(program.phase, 0.0),
            -program.tam_estimate,
            -program.pos_prior,
            -len(program.trials),
            -sum(max(trial.enrollment or 0, 0) for trial in program.trials),
            program.name,
        )
    )

    revenue = float(finance.get("totalRevenue") or 0.0)
    growth_signal = float(finance.get("momentum_3mo") or 0.0)
    approved_products = _approved_products_for_company(str(raw.get("ticker") or ""), revenue, growth_signal)
    if revenue > 10_000_000:
        commercial_title = (
            f"{approved_products[0].name} estimated commercial update"
            if approved_products
            else f"{raw.get('ticker')} estimated commercial update"
        )
        company_catalysts.append(
            _build_catalyst(
                program_id=None,
                event_type="commercial_update",
                title=commercial_title,
                horizon_days=45,
                probability=0.78,
                importance=0.60,
                crowdedness=0.45,
                as_of=as_of,
                status="estimated_from_revenue",
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

    # Staleness check: cross-reference synthetic catalyst events against SEC
    # 8-K filings to detect phantom catalysts (trial read out but CT.gov not
    # updated).  ``event_tape`` is an optional list of raw filing dicts in the
    # raw payload.  If absent we skip gracefully.
    sec_tape: list[dict] = list(raw.get("event_tape") or [])
    all_program_names = [p.name for p in programs]
    all_conditions: list[str] = []
    for p in programs:
        all_conditions.extend(p.conditions)
    company_catalysts = _flag_stale_catalysts(
        catalyst_events=company_catalysts,
        sec_events=sec_tape,
        as_of=as_of.isoformat(),
        program_names=all_program_names,
        conditions=all_conditions,
    )

    snapshot_metadata = {
        "best_phase": raw.get("best_phase"),
        "num_trials": raw.get("num_trials", 0),
        "num_papers": raw.get("num_papers", 0),
        "runway_months": runway_months,
        "runway_months_capped": runway_months >= 120.0,
        "data_source": "prepare_compatibility_layer",
        "description": finance.get("description"),
        "commercial_revenue_present": revenue > 10_000_000,
        "approved_product_registry_hit": bool(approved_products),
        "price_now": finance.get("price_now"),
    }

    snapshot = CompanySnapshot(
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
    snapshot = refresh_snapshot_evidence(snapshot)
    snapshot = apply_curated_company_overrides(snapshot, as_of=as_of)
    return update_snapshot_profile(snapshot)


def fetch_legacy_snapshot(ticker: str, company_name: str | None = None) -> dict[str, Any]:
    return gather_company_data(ticker, company_name or ticker)
