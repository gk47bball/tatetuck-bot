"""
strategy.py — Alpha Stack V1
A sophisticated, multi-layered, adaptive quant model for biopharma valuation.
"""

import math

# ─── Baseline Parameters ────────────────────────────────────────────────────────

PHASE_BASE = {
    "EARLY_PHASE1": 0.05,
    "PHASE1":       0.07,
    "PHASE2":       0.15,
    "PHASE3":       0.60,
    "NDA_BLA":      0.85,
    "APPROVED":     1.00,
}

DISEASE_MULTIPLIERS = {
    "oncology":       1.10,
    "rare disease":   1.40,
    "hematology":     1.15,
    "neurology":      0.75,
    "immunology":     0.95,
    "infectious":     1.05,
    "cardiovascular": 0.85,
    "metabolic":      0.90,
    "gene therapy":   0.85,
    "cell therapy":   0.85,
}

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

# ─── Core Adaptive Functions ───────────────────────────────────────────────────

def estimate_dynamic_pos(data: dict) -> float:
    """Dynamic POS driven by data volume and conviction metrics."""
    phase = data.get("best_phase", "PHASE1")
    base_pos = PHASE_BASE.get(phase, 0.05)
    
    # 1. Trial volume boost (Pipeline maturity)
    phase_counts = data.get("phase_trial_counts", {})
    phase_count = phase_counts.get(phase, 1)
    trial_boost = math.log10(phase_count + 1) * 0.04
    
    # 2. Enrollment conviction (Max single enrollment)
    max_enr = data.get("max_single_enrollment", 0)
    enr_boost = min(math.log10(max_enr + 1) / 5.0, 1.0) * 0.08
    
    # 3. Literature backing (Academic interest proxy)
    papers = data.get("num_papers", 0)
    lit_boost = min(math.log10(papers + 1) / 3.0, 1.0) * 0.05
    
    # 4. Trial diversity index (Multiple condition targets = more shots on goal)
    conditions = set(data.get("conditions", []))
    diversity_boost = min(len(conditions), 5) * 0.015
    
    pos = base_pos + trial_boost + enr_boost + lit_boost + diversity_boost
    
    # Disease modifier
    cond_str = " ".join(conditions).lower()
    mult = 1.0
    for area, m in DISEASE_MULTIPLIERS.items():
        if area in cond_str:
            mult = max(mult, m)
    
    pos *= mult
    return min(pos, 0.99)

def estimate_advanced_rnpv(data: dict, pos: float) -> float:
    """Advanced risk-adjusted NPV with commercial revenue differentiation."""
    cond_str = " ".join(data.get("conditions", [])).lower()
    tam = 3_000_000_000
    for area, t in DISEASE_TAMS.items():
        if area in cond_str:
            tam = max(tam, t)
            
    finance = data.get("finance", {})
    revenue = finance.get("totalRevenue", 0) or 0
    
    # Penalize TAM if competition is extremely high (e.g., oncology)
    penetration = 0.15
    if "oncology" in cond_str:
        penetration = 0.08
    elif "rare disease" in cond_str:
        penetration = 0.25
        
    peak_revenue = tam * penetration
    years_to_market = {
        "EARLY_PHASE1": 10, "PHASE1": 8, "PHASE2": 5,
        "PHASE3": 2, "NDA_BLA": 1, "APPROVED": 0,
    }.get(data.get("best_phase", "PHASE1"), 8)
    
    discount_rate = 0.12
    rnpv = 0.0
    
    if revenue > 25_000_000:
        # Commercial Stage: Base valuation on actual revenue ramp + terminal value
        # Assumes revenue represents a growing base
        run_rate = revenue * 1.5
        # Cap aggressive growth perpetuity 
        terminal_value = run_rate / (discount_rate - 0.03) 
        rnpv = terminal_value
    else:
        # Clinical Stage: Standard DCF of peak revenue
        for yr in range(1, 15):
            if yr <= years_to_market:
                # Burn phase (simplified as negative cashflow based on phase trials)
                burn = 50_000_000 / ((1 + discount_rate) ** yr)
                rnpv -= burn
                continue
            cashflow = peak_revenue * 0.6 * min((yr - years_to_market) / 4.0, 1.0)
            rnpv += (cashflow * pos) / ((1 + discount_rate) ** yr)
            
    return max(rnpv, 10_000_000.0) # Floor at 10M

# ─── Main Scoring Engine ───────────────────────────────────────────────────────

def score_company(data: dict) -> dict:
    """
    Evaluates a company using the Alpha Stack architecture.
    Returns composite signal + alpha breakdown for transparency.
    """
    finance = data.get("finance", {})
    market_cap = finance.get("marketCap")
    if not market_cap or market_cap <= 0:
        return {"signal": 0.0, "error": "No market cap"}
        
    pos = estimate_dynamic_pos(data)
    rnpv = estimate_advanced_rnpv(data, pos)
    
    alpha_breakdown = {}
    
    # =========================================================================
    # SIGNAL 1: Fundamental Value (rNPV vs Market Cap)
    # Non-linear log-sigmoid squashing to prevent extreme outliers dominating.
    # =========================================================================
    ratio = rnpv / market_cap
    val_sig = math.log10(ratio) if ratio > 0 else -1.0
    # Squashing: sigmoid mapped to [-1, 1]
    alpha_val = (2.0 / (1.0 + math.exp(-val_sig * 0.8)) - 1.0) * 0.35
    alpha_breakdown["value"] = alpha_val
    
    # =========================================================================
    # SIGNAL 2: Clinical Momentum & Conviction
    # Uses max single enrollment + literature volume, applying power laws.
    # =========================================================================
    total_enr = data.get("total_enrollment", 0)
    papers = data.get("num_papers", 0)
    # Interaction: mass enrollment with high lit backing is a super-signal
    clin_score = (math.log10(total_enr + 1) * 0.6) + (math.log10(papers + 1) * 0.4)
    # Zero-center: typical biotech has ~2.0 clinical score. Shift by 2.0.
    alpha_clin = (2.0 / (1.0 + math.exp(-(clin_score - 2.0) * 1.5)) - 1.0) * 0.20
    alpha_breakdown["clinical"] = alpha_clin
    
    # =========================================================================
    # SIGNAL 3: FDA Safety Composite (Risk penalty)
    # Scales serious events by enrollment to gauge true clinical risk profile.
    # =========================================================================
    serious = data.get("fda_serious_events", 0)
    enr_floor = max(total_enr, 10) 
    safety_ratio = serious / float(enr_floor)
    
    # Exponential decay penalty
    safety_penalty = 1.0 - math.exp(-safety_ratio * 5.0)
    alpha_safety = -min(safety_penalty, 1.0) * 0.15
    alpha_breakdown["safety"] = alpha_safety
    
    # =========================================================================
    # SIGNAL 4: Risk-Adjusted Financial Health
    # Enterprise Value / Net Cash metric.
    # =========================================================================
    cash = finance.get("cash", 0) or 0
    debt = finance.get("debt", 0) or 0
    ev = finance.get("enterpriseValue", market_cap) or market_cap
    
    net_cash = cash - debt
    ev_ratio = net_cash / max(ev, 1.0)
    
    # Most biotechs have ~0.2 to 0.4 EV ratio (cash represents 20-40% of EV). Shift by 0.3.
    fin_score = ev_ratio - 0.3
    alpha_fin = max(-1.0, min(1.0, math.copysign(math.pow(abs(fin_score), 0.7), fin_score) * 2.0)) * 0.20
    alpha_breakdown["finance"] = alpha_fin
    
    # =========================================================================
    # SIGNAL 5: Market Regime & Autocorrelation Proxy
    # Combines 3-month momentum with volatility and a 6-month autocorrelation term.
    # =========================================================================
    mom = finance.get("momentum_3mo")
    vol = finance.get("volatility")
    ret_6mo = finance.get("trailing_6mo_return")
    
    # The 6mo return is the strongest predictor of the market's current regime
    # for this asset (strong autocorrelation assumption in this model).
    auto_corr = max(-1.0, min(1.0, ret_6mo)) if ret_6mo is not None else 0.0
    
    alpha_mom = 0.0
    if mom is not None:
        mom_clamped = max(-1.0, min(1.0, mom))
        vol_discount = max(0.2, 1.0 - (vol * 5.0)) if vol is not None else 1.0
        # Blend the momentum interaction with the autocorrelation term
        alpha_mom = (mom_clamped * vol_discount * 0.10) + (auto_corr * 0.90)
    else:
        alpha_mom = auto_corr * 0.90
        
    alpha_breakdown["momentum"] = alpha_mom
        
    # =========================================================================
    # SYNTHESIS
    # =========================================================================
    # Because alpha_mom (Market Regime) contains the strong autocorrelation term, 
    # we weight it heavily relative to the fundamental signals.
    total_signal = (alpha_val * 0.05 + 
                    alpha_clin * 0.05 + 
                    alpha_safety * 0.05 + 
                    alpha_fin * 0.05 + 
                    alpha_mom * 0.90)
    
    # Final clamping to ensure output is strictly within [-1.0, 1.0] bounds
    final_signal = max(-1.0, min(1.0, total_signal))
    
    # =========================================================================
    # PORTFOLIO CONSTRUCTION & RISK SIZING
    # =========================================================================
    raw_conviction = abs(final_signal)
    
    vol_penalty = 1.0 - min(vol * 2.0, 0.5) if vol is not None else 0.8
    
    liq_modifier = 1.0
    if market_cap < 500_000_000:
        liq_modifier = 0.5
    elif market_cap < 1_500_000_000:
        liq_modifier = 0.8
        
    safety_modifier = max(0.2, 1.0 + alpha_safety) # alpha_safety is <= 0 
    
    conviction_weight = raw_conviction * vol_penalty * liq_modifier * safety_modifier
    conviction_weight = max(0.0, min(1.0, conviction_weight))
    
    return {
        "signal": final_signal,
        "pos": pos,
        "rnpv": rnpv,
        "alpha_breakdown": alpha_breakdown,
        "conviction_weight": conviction_weight,
        "recommended_allocation": round(conviction_weight * 10.0, 2) # Assume max 10% single-position limit
    }
