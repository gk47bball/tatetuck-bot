from typing import List, Dict, Any

from prepare import fetch_clinical_trials

class ClinicalTrialsAPI:
    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

    def search_by_sponsor(self, sponsor_name: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Delegate to prepare.py so the package and evaluator share one canonical client."""
        return fetch_clinical_trials(sponsor_name, max_results=max_results)
