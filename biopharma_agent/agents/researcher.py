"""
BiopharmaResearcher: The core AutoResearch orchestrator.

Pulls data from ClinicalTrials.gov, openFDA, PubMed, and Yahoo Finance,
then runs the Epi-Model and (optionally) LLM-enhanced POS estimation
to produce a comprehensive investment research report.
"""
from typing import Dict, Any, List
from ..data.clinical_trials import ClinicalTrialsAPI
from ..data.fda import OpenFDAAPI
from ..data.finance import FinanceAPI
from ..models.pos_estimator import POSEstimator
from ..models.epi_model import EpiModel
from .pubmed_agent import AutoResearchAgent

class BiopharmaResearcher:
    def __init__(self):
        self.clinical_api = ClinicalTrialsAPI()
        self.fda_api = OpenFDAAPI()
        self.finance_api = FinanceAPI()
        self.pos_estimator = POSEstimator()
        self.epi_model = EpiModel()
        self.literature_agent = AutoResearchAgent()

    def run_research(self, ticker: str, company_name: str = None) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"  BIOPHARMA AUTO-RESEARCHER: {ticker}")
        print(f"{'='*60}\n")
        
        # ── Step 1: Financial Data ──────────────────────────────────
        print("[1/6] Fetching financial data from Yahoo Finance...")
        try:
            finance_data = self.finance_api.get_company_data(ticker)
        except Exception as e:
            print(f"  ⚠ Error: {e}")
            finance_data = {"ticker": ticker}
            
        if not company_name:
            company_name = finance_data.get("shortName") or finance_data.get("longName") or ticker
        print(f"  → Company: {company_name}")
        print(f"  → Market Cap: ${finance_data.get('marketCap', 0):,.0f}" if finance_data.get('marketCap') else "  → Market Cap: N/A")
            
        # ── Step 2: Clinical Trials ─────────────────────────────────
        print(f"\n[2/6] Searching ClinicalTrials.gov for: {company_name}...")
        try:
            trials_data = self.clinical_api.search_by_sponsor(company_name, max_results=10)
        except Exception as e:
            print(f"  ⚠ Error: {e}")
            trials_data = []
        print(f"  → Found {len(trials_data)} clinical trials")
        for t in trials_data[:5]:
            phase_str = ", ".join(t.get("phase", [])) or "N/A"
            print(f"    • [{t.get('nct_id')}] {t.get('title', 'N/A')[:70]}... (Phase: {phase_str})")

        # ── Step 3: FDA Data ────────────────────────────────────────
        print(f"\n[3/6] Querying openFDA for safety/label data...")
        fda_data = []
        drug_names = []
        all_conditions = []
        if trials_data:
            drug_names = list(set([
                dx.split()[0].replace(',', '').replace('.', '')
                for t in trials_data
                for dx in t.get("interventions", []) if dx
            ]))
            all_conditions = list(set([
                c for t in trials_data
                for c in t.get("conditions", [])
            ]))
            for drug_name in drug_names[:3]:
                try:
                    events = self.fda_api.get_adverse_events(drug_name, limit=3)
                    fda_data.extend(events)
                    print(f"  → {drug_name}: {len(events)} adverse event reports")
                except Exception as e:
                    print(f"  → {drug_name}: No FDA data ({e})")
        print(f"  → Total FDA records: {len(fda_data)}")

        # ── Step 4: Heuristic POS (Epi-Model Phase-Based) ──────────
        print(f"\n[4/6] Computing heuristic POS from clinical phase data...")
        heuristic = self.epi_model.estimate_heuristic_pos(trials_data)
        print(f"  → {heuristic['details']}")

        # ── Step 5: PubMed AutoResearch (Karpathy-style) ───────────
        print(f"\n[5/6] Running Karpathy-style AutoResearch on PubMed...")
        lit_review = self.literature_agent.generate_literature_review(
            company_name, drug_names[:3], all_conditions[:3]
        )

        # ── Step 6: LLM-Enhanced POS & Epi-Model Valuation ────────
        print(f"\n[6/6] Final POS estimation & Epi-Model valuation...")
        pos_result = self.pos_estimator.estimate_pos(
            company_name, trials_data, fda_data, lit_review,
            heuristic_pos=heuristic["heuristic_pos"]
        )

        # Use LLM TAM if available, otherwise default
        tam = pos_result.estimated_tam
        tam = max(min(tam, 100_000_000_000), 10_000_000)  # sanity bounds

        valuation = self.epi_model.calculate_valuation(
            pos=pos_result.probability_of_success,
            tam=tam,
            years_to_peak=heuristic["years_remaining"],
            market_cap=finance_data.get("marketCap"),
        )

        print(f"\n  → Final POS: {pos_result.probability_of_success*100:.1f}%")
        print(f"  → rNPV:      ${valuation['rNPV']:,.0f}")
        if valuation.get("signal"):
            print(f"  → Signal:    {valuation['signal']}")
            print(f"  → Implied Upside: {valuation.get('implied_upside_pct', 0):+.1f}%")

        return {
            "company": company_name,
            "ticker": ticker,
            "finance_data": finance_data,
            "trials_data": trials_data,
            "drug_names": drug_names,
            "conditions": all_conditions,
            "fda_records": len(fda_data),
            "heuristic_analysis": heuristic,
            "literature_review": lit_review,
            "pos_analysis": {
                "probability_of_success": pos_result.probability_of_success,
                "estimated_tam": pos_result.estimated_tam,
                "reasoning": pos_result.reasoning
            },
            "valuation": valuation,
        }
