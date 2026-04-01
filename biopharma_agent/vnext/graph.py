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
from .market_profile import update_snapshot_profile
from .taxonomy import program_event_type_for_phase


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
    "non interventional",
    "observational",
    "long-term follow-up",
    "long term follow-up",
    "specimen collection",
    "expanded access",
    "registry",
    "follow-up study",
    "healthy volunteer",
    "healthy volunteers",
)

APPROVED_PRODUCT_REGISTRY = {
    "APLS": [
        ("Syfovre", "geographic atrophy"),
        ("Empaveli", "PNH / C3 glomerulopathy"),
    ],
    "CRSP": [
        ("CASGEVY", "sickle cell disease / transfusion-dependent beta-thalassemia"),
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


def _trial_text(trial: Trial) -> str:
    return " ".join(
        item
        for item in [
            trial.title,
            " ".join(trial.conditions),
            " ".join(trial.interventions),
        ]
        if item
    ).lower()


def _is_low_signal_trial(trial: Trial) -> bool:
    text = _trial_text(trial)
    return any(keyword in text for keyword in LOW_SIGNAL_TRIAL_KEYWORDS)


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
        pos_prior = min(0.99, PHASE_BASE.get(phase, 0.07) + min(len(program_trials) * 0.03, 0.12))
        conditions = sorted(conditions_by_program[program_name])
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
                trials=sorted(program_trials, key=lambda item: item.enrollment, reverse=True),
                pos_prior=pos_prior,
                tam_estimate=tam_estimate,
                catalyst_events=program_catalysts,
                evidence=evidence[:3],
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
    return update_snapshot_profile(snapshot)


def fetch_legacy_snapshot(ticker: str, company_name: str | None = None) -> dict[str, Any]:
    return gather_company_data(ticker, company_name or ticker)
