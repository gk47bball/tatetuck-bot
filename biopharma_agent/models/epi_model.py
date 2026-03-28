"""
Epi-Model: Phase-Based Probability of Success & Risk-Adjusted NPV Valuation

Uses industry-standard clinical phase transition probabilities (BIO/QLS/Informa data)
to calculate a heuristic POS, then computes rNPV and compares to current market cap
to find the alpha signal.
"""
from typing import Dict, Any, List, Optional

# Industry-standard clinical phase transition probabilities
# Source: BIO / QLS Advisors / Informa Pharma Intelligence
PHASE_TRANSITION_RATES = {
    "PHASE1":       {"pos_to_approval": 0.074, "years_remaining": 10, "label": "Phase 1"},
    "PHASE2":       {"pos_to_approval": 0.152, "years_remaining": 7,  "label": "Phase 2"},
    "PHASE3":       {"pos_to_approval": 0.590, "years_remaining": 3,  "label": "Phase 3"},
    "NDA/BLA":      {"pos_to_approval": 0.900, "years_remaining": 1,  "label": "NDA/BLA Filed"},
    "APPROVED":     {"pos_to_approval": 1.000, "years_remaining": 0,  "label": "Approved"},
    "EARLY_PHASE1": {"pos_to_approval": 0.055, "years_remaining": 12, "label": "Pre-Clinical / Early Phase 1"},
}

# Disease-area adjustments (multipliers on base POS)
DISEASE_AREA_ADJUSTMENT = {
    "oncology":       0.85,
    "rare disease":   1.25,
    "hematology":     1.10,
    "neurology":      0.70,
    "immunology":     0.95,
    "infectious":     1.05,
    "cardiovascular": 0.80,
    "metabolic":      0.90,
    "default":        1.00,
}

def classify_phase(phase_list: List[str]) -> str:
    """Map ClinicalTrials.gov phase strings to our lookup key."""
    if not phase_list:
        return "PHASE1"
    phase_str = " ".join(phase_list).upper()
    if "PHASE3" in phase_str or "PHASE 3" in phase_str:
        return "PHASE3"
    if "PHASE2" in phase_str or "PHASE 2" in phase_str:
        return "PHASE2"
    if "PHASE1" in phase_str or "PHASE 1" in phase_str:
        if "PHASE2" in phase_str or "PHASE 2" in phase_str:
            return "PHASE2"
        return "PHASE1"
    if "EARLY" in phase_str:
        return "EARLY_PHASE1"
    return "PHASE1"

def get_disease_multiplier(conditions: List[str]) -> float:
    """Heuristic: adjust POS based on therapeutic area."""
    if not conditions:
        return 1.0
    conditions_lower = " ".join(conditions).lower()
    for area, mult in DISEASE_AREA_ADJUSTMENT.items():
        if area in conditions_lower:
            return mult
    return 1.0


class EpiModel:
    def __init__(self, discount_rate: float = 0.12, peak_sales_multiple: float = 3.0):
        self.discount_rate = discount_rate
        self.peak_sales_multiple = peak_sales_multiple

    def estimate_heuristic_pos(self, trials_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate a heuristic POS from trial phase data using industry base rates.
        Returns the best (most advanced) trial's POS and details.
        """
        if not trials_data:
            return {
                "heuristic_pos": 0.05,
                "phase_label": "Unknown (no trials found)",
                "years_remaining": 10,
                "disease_multiplier": 1.0,
                "details": "No clinical trial data available; defaulting to 5% POS."
            }
        
        best_pos = 0.0
        best_trial = None
        best_phase_key = "PHASE1"

        for trial in trials_data:
            phase_key = classify_phase(trial.get("phase", []))
            phase_info = PHASE_TRANSITION_RATES.get(phase_key, PHASE_TRANSITION_RATES["PHASE1"])
            disease_mult = get_disease_multiplier(trial.get("conditions", []))
            adjusted_pos = phase_info["pos_to_approval"] * disease_mult
            
            if adjusted_pos > best_pos:
                best_pos = adjusted_pos
                best_trial = trial
                best_phase_key = phase_key
        
        phase_info = PHASE_TRANSITION_RATES[best_phase_key]
        disease_mult = get_disease_multiplier(best_trial.get("conditions", []))

        return {
            "heuristic_pos": min(best_pos, 1.0),
            "phase_label": phase_info["label"],
            "years_remaining": phase_info["years_remaining"],
            "disease_multiplier": disease_mult,
            "lead_trial": best_trial.get("title", "N/A"),
            "lead_nct_id": best_trial.get("nct_id", "N/A"),
            "details": f"Lead asset is in {phase_info['label']}. Base POS-to-approval: {phase_info['pos_to_approval']*100:.1f}%. Disease area multiplier: {disease_mult:.2f}x. Adjusted heuristic POS: {best_pos*100:.1f}%."
        }

    def calculate_valuation(
        self,
        pos: float,
        tam: float,
        years_to_peak: int = 5,
        market_cap: Optional[float] = None,
        penetration_rate: float = 0.15,
        gross_margin: float = 0.80,
        royalty_burden: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Calculate risk-adjusted NPV and alpha signal vs current market cap.
        
        pos:              Probability of Success (0-1)
        tam:              Total Addressable Market in USD
        years_to_peak:    Years until peak sales
        market_cap:       Current market cap (if available) for alpha calculation
        penetration_rate: Assumed peak market share
        gross_margin:     Assumed gross margin on drug revenue
        royalty_burden:   Assumed royalty/licensing cost
        """
        peak_revenue = tam * penetration_rate
        net_revenue = peak_revenue * gross_margin * (1 - royalty_burden)
        
        # Build a simplified revenue ramp: 0 → peak over years_to_peak, then 5 years plateau, then decline
        total_npv = 0.0
        revenue_schedule = []
        total_years = years_to_peak + 8  # ramp + plateau + decline
        
        for year in range(1, total_years + 1):
            if year <= years_to_peak:
                # Linear ramp to peak
                rev = net_revenue * (year / years_to_peak)
            elif year <= years_to_peak + 5:
                # Peak plateau
                rev = net_revenue
            else:
                # Decline at 20% per year after patent cliff
                years_past_cliff = year - (years_to_peak + 5)
                rev = net_revenue * (0.80 ** years_past_cliff)
            
            discounted = rev / ((1 + self.discount_rate) ** year)
            total_npv += discounted
            revenue_schedule.append({"year": year, "revenue": rev, "discounted": discounted})
        
        rNPV = total_npv * pos
        
        result = {
            "tam": tam,
            "penetration_rate": penetration_rate,
            "peak_revenue": peak_revenue,
            "net_revenue_at_peak": net_revenue,
            "unadjusted_npv": total_npv,
            "probability_of_success": pos,
            "rNPV": rNPV,
            "revenue_schedule": revenue_schedule,
        }
        
        # Alpha signal: compare rNPV to market cap
        if market_cap and market_cap > 0:
            implied_upside = ((rNPV / market_cap) - 1) * 100
            result["current_market_cap"] = market_cap
            result["implied_upside_pct"] = implied_upside
            if implied_upside > 50:
                result["signal"] = "STRONG BUY — Significantly undervalued vs rNPV"
            elif implied_upside > 15:
                result["signal"] = "BUY — Moderately undervalued vs rNPV"
            elif implied_upside > -15:
                result["signal"] = "HOLD — Fairly valued"
            elif implied_upside > -40:
                result["signal"] = "SELL — Moderately overvalued vs rNPV"
            else:
                result["signal"] = "STRONG SELL — Significantly overvalued vs rNPV"
        
        return result
