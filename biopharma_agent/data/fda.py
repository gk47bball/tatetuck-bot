import requests
from typing import Dict, Any, List

class OpenFDAAPI:
    BASE_URL = "https://api.fda.gov/drug"

    def get_drug_labels(self, brand_name: str, limit: int = 1) -> List[Dict[str, Any]]:
        """Get FDA labels for a given brand name drug."""
        url = f"{self.BASE_URL}/label.json"
        params = {
            "search": f'openfda.brand_name:"{brand_name}"',
            "limit": limit
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return [] # No labels found for this term
            raise e

    def get_adverse_events(self, brand_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent adverse events for a drug."""
        url = f"{self.BASE_URL}/event.json"
        params = {
            "search": f'patient.drug.openfda.brand_name:"{brand_name}"',
            "limit": limit
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return []
            raise e
