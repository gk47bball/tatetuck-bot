"""
strategy.py — The file the AI agent modifies.
This is the equivalent of Karpathy's train.py.

Contains the scoring strategy that the agent iterates on.
Everything here is fair game: weights, logic, formulas, heuristics.
The agent's goal is to minimize valuation_error by making this
strategy's predictions match real market outcomes.
"""

import math

# ─── Tunable Parameters ─────────────────────────────────────────────────────────
# The agent can (and should) modify these.

PHASE_WEIGHTS = {
    "EARLY_PHASE1": 0.055,
    "PHASE1":       0.074,
    "PHASE2":       0.152,
    "PHASE3":       0.700,
    "NDA_BLA":      0.900,
    "APPROVED":     1.000,
}

DISEASE_MULTIPLIERS = {
    "oncology":       0.85,
    "rare disease":   1.25,
    "hematology":     1.10,
    "neurology":      0.70,
    "immunology":     0.95,
    "infectious":     1.05,
    "cardiovascular": 0.80,
    "metabolic":      0.90,
    "gene therapy":   1.15,
    "cell therapy":   1.10,
}

# Disease-specific TAMs
DISEASE_TAMS = {
    "oncology":       8_000_000_000,
    "rare disease":   1_500_000_000,
    "hematology":     4_000_000_000,
    "neurology":      6_000_000_000,
    "immunology":     5_000_000_000,
    "infectious":     3_000_000_000,
    "cardiovascular": 7_000_000_000,
    "metabolic":      5_000_000_000,
    "gene therapy":   2_000_000_000,
    "cell therapy":   2_500_000_000,
}
DEFAULT_TAM = 3_000_000_000

DISCOUNT_RATE = 0.12
PENETRATION_RATE = 0.15
CASH_RUNWAY_WEIGHT = 0.10
PIPELINE_BREADTH_WEIGHT = 0.05
LITERATURE_WEIGHT = 0.05
MOMENTUM_WEIGHT = 0.40        # 3-month price momentum signal
ENROLLMENT_WEIGHT = 0.08      # Large enrollment = management conviction
FDA_SAFETY_PENALTY = 0.10     # Penalty for high serious adverse event ratio

# ─── Strategy Logic ──────────────────────────────────────────────────────────────

def get_disease_multiplier(conditions: list) -> float:
    """Return the highest-matching disease area multiplier."""
    if not conditions:
        return 1.0
    cond_str = " ".join(conditions).lower()
    best = 1.0
    for area, mult in DISEASE_MULTIPLIERS.items():
        if area in cond_str:
            best = max(best, mult)
    return best


def get_disease_tam(conditions: list) -> float:
    """Return the best-matching disease-specific TAM."""
    if not conditions:
        return DEFAULT_TAM
    cond_str = " ".join(conditions).lower()
    best_tam = DEFAULT_TAM
    for area, tam in DISEASE_TAMS.items():
        if area in cond_str:
            best_tam = max(best_tam, tam)
    return best_tam


def estimate_pos(data: dict) -> float:
    """Estimate probability of success purely from our own PHASE_WEIGHTS."""
    phase = data.get("best_phase", "PHASE1")
    pos = PHASE_WEIGHTS.get(phase, 0.074)

    # Disease area adjustment
    disease_mult = get_disease_multiplier(data.get("conditions", []))
    pos *= disease_mult

    return min(pos, 1.0)


def estimate_rnpv(pos: float, tam: float, years_to_market: int = 5) -> float:
    """Calculate risk-adjusted NPV."""
    peak_revenue = tam * PENETRATION_RATE

    total_npv = 0.0
    total_years = years_to_market + 8
    for year in range(1, total_years + 1):
        if year <= years_to_market:
            rev = peak_revenue * (year / years_to_market)
        elif year <= years_to_market + 5:
            rev = peak_revenue
        else:
            rev = peak_revenue * (0.80 ** (year - years_to_market - 5))
        total_npv += rev / ((1 + DISCOUNT_RATE) ** year)

    return total_npv * pos


def score_company(data: dict) -> dict:
    """
    Score a single company. This is the function that evaluate.py calls.

    Must return a dict with at least:
        "signal": float between -1.0 (strong sell) and +1.0 (strong buy)

    The agent's goal is to make "signal" correlate with actual 6-month returns.
    """
    pos = estimate_pos(data)
    tam = get_disease_tam(data.get("conditions", []))

    phase_years = {
        "EARLY_PHASE1": 12, "PHASE1": 10, "PHASE2": 7,
        "PHASE3": 3, "NDA_BLA": 1, "APPROVED": 0,
    }
    years = phase_years.get(data.get("best_phase", "PHASE1"), 10)

    rnpv = estimate_rnpv(pos, tam, years)

    market_cap = data.get("finance", {}).get("marketCap")

    if not market_cap or market_cap <= 0:
        return {
            "signal": 0.0,
            "pos": pos,
            "rnpv": rnpv,
            "reason": "No market cap data available",
        }

    # ── Core signal: rNPV vs market cap (log-scaled) ──
    ratio = rnpv / market_cap
    if ratio > 0:
        base_signal = math.log10(ratio)
    else:
        base_signal = -1.0

    # ── Cash runway adjustment ──
    cash = data.get("finance", {}).get("cash") or 0
    debt = data.get("finance", {}).get("debt") or 0
    if market_cap > 0:
        cash_ratio = (cash - debt) / market_cap
        base_signal += cash_ratio * CASH_RUNWAY_WEIGHT

    # ── Pipeline breadth (scaled) ──
    num_trials = data.get("num_trials", 0)
    if num_trials > 3:
        breadth_bonus = min(num_trials / 20.0, 1.0) * PIPELINE_BREADTH_WEIGHT
        base_signal += breadth_bonus

    # ── Enrollment conviction signal ──
    max_enrollment = data.get("max_single_enrollment", 0)
    if max_enrollment > 100:
        # Larger enrollment = higher conviction from management
        enrollment_bonus = min(math.log10(max_enrollment) / 4.0, 1.0) * ENROLLMENT_WEIGHT
        base_signal += enrollment_bonus

    # ── Literature bonus (scaled) ──
    num_papers = data.get("num_papers", 0)
    if num_papers > 0:
        lit_bonus = min(num_papers / 5.0, 1.0) * LITERATURE_WEIGHT
        base_signal += lit_bonus

    # ── 3-month price momentum ──
    momentum = data.get("finance", {}).get("momentum_3mo")
    if momentum is not None:
        # Momentum as a signal: positive momentum = tailwind
        mom_signal = max(-1.0, min(1.0, momentum))
        base_signal += mom_signal * MOMENTUM_WEIGHT

    # ── FDA safety penalty ──
    fda_serious_ratio = data.get("fda_serious_ratio", 0.0)
    if fda_serious_ratio > 0.5:
        base_signal -= fda_serious_ratio * FDA_SAFETY_PENALTY

    # ── Clamp to [-1, 1] ──
    signal = max(-1.0, min(1.0, base_signal))

    return {
        "signal": signal,
        "pos": round(pos, 4),
        "rnpv": round(rnpv, 2),
        "tam": tam,
        "market_cap": market_cap,
        "rnpv_to_mcap_ratio": round(ratio, 4),
        "best_phase": data.get("best_phase"),
        "num_trials": num_trials,
        "max_enrollment": max_enrollment,
        "num_papers": num_papers,
        "momentum_3mo": momentum,
        "fda_serious_ratio": fda_serious_ratio,
    }
