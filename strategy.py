"""
strategy.py — The file the AI agent modifies.
This is the equivalent of Karpathy's train.py.

Contains the scoring strategy that the agent iterates on.
Everything here is fair game: weights, logic, formulas, heuristics.
The agent's goal is to minimize valuation_error by making this
strategy's predictions match real market outcomes.
"""

# ─── Tunable Parameters ─────────────────────────────────────────────────────────
# The agent can (and should) modify these.

PHASE_WEIGHTS = {
    "EARLY_PHASE1": 0.055,
    "PHASE1":       0.074,
    "PHASE2":       0.152,
    "PHASE3":       0.590,
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

DISCOUNT_RATE = 0.12
PENETRATION_RATE = 0.15
DEFAULT_TAM = 2_000_000_000   # $2B default TAM assumption
CASH_RUNWAY_WEIGHT = 0.10     # How much cash position influences signal
PIPELINE_BREADTH_WEIGHT = 0.05  # Bonus for multiple trials
LITERATURE_WEIGHT = 0.05      # Bonus for strong publication record

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


def estimate_pos(data: dict) -> float:
    """Estimate probability of success from trial data."""
    base_pos = data.get("base_pos", 0.074)
    phase_pos = PHASE_WEIGHTS.get(data.get("best_phase", "PHASE1"), 0.074)
    # Use the higher of the two (in case we have better phase data)
    pos = max(base_pos, phase_pos)
    
    # Disease area adjustment
    disease_mult = get_disease_multiplier(data.get("conditions", []))
    pos *= disease_mult
    
    return min(pos, 1.0)


def estimate_rnpv(pos: float, tam: float, years_to_market: int = 5) -> float:
    """Calculate risk-adjusted NPV."""
    peak_revenue = tam * PENETRATION_RATE
    
    # Simple revenue model: ramp to peak, plateau, decline
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
    tam = DEFAULT_TAM
    
    # Phase-based years to market
    phase_years = {
        "EARLY_PHASE1": 12, "PHASE1": 10, "PHASE2": 7,
        "PHASE3": 3, "NDA_BLA": 1, "APPROVED": 0,
    }
    years = phase_years.get(data.get("best_phase", "PHASE1"), 10)
    
    rnpv = estimate_rnpv(pos, tam, years)
    
    market_cap = data.get("finance", {}).get("marketCap")
    
    if not market_cap or market_cap <= 0:
        # Can't compute alpha without market cap
        return {
            "signal": 0.0,
            "pos": pos,
            "rnpv": rnpv,
            "reason": "No market cap data available",
        }
    
    # Core signal: rNPV vs market cap
    ratio = rnpv / market_cap
    base_signal = (ratio - 1.0)  # positive = undervalued, negative = overvalued
    
    # Cash runway adjustment
    cash = data.get("finance", {}).get("cash") or 0
    debt = data.get("finance", {}).get("debt") or 0
    if market_cap > 0:
        cash_ratio = (cash - debt) / market_cap
        base_signal += cash_ratio * CASH_RUNWAY_WEIGHT
    
    # Pipeline breadth bonus
    num_trials = data.get("num_trials", 0)
    if num_trials > 5:
        base_signal += PIPELINE_BREADTH_WEIGHT
    
    # Literature bonus
    num_papers = data.get("num_papers", 0)
    if num_papers > 3:
        base_signal += LITERATURE_WEIGHT
    
    # Clamp to [-1, 1]
    signal = max(-1.0, min(1.0, base_signal))
    
    return {
        "signal": signal,
        "pos": round(pos, 4),
        "rnpv": round(rnpv, 2),
        "market_cap": market_cap,
        "rnpv_to_mcap_ratio": round(ratio, 4),
        "best_phase": data.get("best_phase"),
        "num_trials": num_trials,
        "num_papers": num_papers,
    }
