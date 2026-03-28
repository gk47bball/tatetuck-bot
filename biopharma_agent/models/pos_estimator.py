"""
POS Estimator: LLM-enhanced Probability of Success scoring.

If a Gemini API key is available, uses the LLM to deeply analyze trial design,
FDA data, and literature context. Otherwise falls back to the heuristic phase-based
estimator in epi_model.py.
"""
import os
import json
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field

class POSResult(BaseModel):
    probability_of_success: float = Field(description="Estimated probability of success (0.0 to 1.0)")
    estimated_tam: float = Field(description="Estimated Total Addressable Market in USD")
    reasoning: str = Field(description="Detailed reasoning for the POS score and TAM estimate")

class POSEstimator:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.client = None
        self.model_name = 'gemini-2.5-flash'
        
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client()
                print("[POSEstimator] Gemini API key found — LLM-enhanced mode enabled.")
            except Exception as e:
                print(f"[POSEstimator] Could not initialize Gemini client: {e}")
                self.client = None
        else:
            print("[POSEstimator] No GEMINI_API_KEY — running in heuristic-only mode.")

    @property
    def has_llm(self) -> bool:
        return self.client is not None

    def estimate_pos(
        self,
        company_name: str,
        trials_data: List[Dict[str, Any]],
        fda_data: List[Dict[str, Any]],
        lit_review: str = "",
        heuristic_pos: Optional[float] = None,
    ) -> POSResult:
        """
        Estimate POS. If LLM is available, use it to deeply evaluate. 
        Otherwise, return the heuristic POS from the Epi-Model.
        """
        if not self.has_llm:
            # Heuristic fallback
            pos = heuristic_pos if heuristic_pos else 0.10
            return POSResult(
                probability_of_success=pos,
                estimated_tam=1_000_000_000,
                reasoning=f"Heuristic estimate based on clinical phase transition rates. POS = {pos*100:.1f}%. "
                          f"Set GEMINI_API_KEY for LLM-enhanced deep analysis."
            )

        prompt = f"""You are a senior biopharma analyst at a top-tier healthcare hedge fund.

Analyze the following data for **{company_name}** and estimate:
1. The Probability of Success (POS) for their lead drug candidate reaching FDA approval.
2. The Total Addressable Market (TAM) in USD for the targeted indication(s).

## Clinical Trials Data
{json.dumps(trials_data, indent=2)}

## FDA Safety / Label Data
{json.dumps(fda_data[:3], indent=2) if fda_data else "No FDA data available."}

## Scientific Literature Review
{lit_review if lit_review else "No literature review available."}

## Heuristic Baseline
The phase-based heuristic POS estimate is {(heuristic_pos or 0.10)*100:.1f}%. Use this as a starting point, but adjust based on trial design quality, endpoint strength, competitive landscape, safety signals, and scientific rationale from the literature.

Be rigorous. Cite specific evidence from the data above. Return JSON with:
- probability_of_success (float 0.0–1.0)
- estimated_tam (float, USD)
- reasoning (string, detailed)"""

        try:
            from google import genai
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': POSResult,
                    'temperature': 0.2
                }
            )
            
            if hasattr(response, 'parsed') and response.parsed is not None:
                return response.parsed
            
            data = json.loads(response.text)
            return POSResult(**data)
        except Exception as e:
            pos = heuristic_pos if heuristic_pos else 0.10
            return POSResult(
                probability_of_success=pos,
                estimated_tam=1_000_000_000,
                reasoning=f"LLM analysis failed ({e}). Falling back to heuristic POS = {pos*100:.1f}%."
            )
