import requests
from typing import List, Dict, Any

class ClinicalTrialsAPI:
    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

    def search_by_sponsor(self, sponsor_name: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Fetch ongoing or completed trials by a specific sponsor."""
        params = {
            "query.spons": sponsor_name,
            "pageSize": max_results,
            "format": "json"
        }
        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        studies = data.get("studies", [])
        
        parsed_trials = []
        for study in studies:
            protocol = study.get("protocolSection", {})
            identification = protocol.get("identificationModule", {})
            status = protocol.get("statusModule", {})
            design = protocol.get("designModule", {})
            outcomes = protocol.get("outcomesModule", {})
            conditions = protocol.get("conditionsModule", {})
            interventions = protocol.get("armsInterventionsModule", {})
            
            parsed_trials.append({
                "nct_id": identification.get("nctId"),
                "title": identification.get("briefTitle"),
                "overall_status": status.get("overallStatus"),
                "phase": design.get("phases", []),
                "conditions": conditions.get("conditions", []),
                "interventions": [i.get("name") for i in interventions.get("interventions", [])],
                "primary_outcomes": [o.get("measure") for o in outcomes.get("primaryOutcomes", [])],
                "enrollment": design.get("enrollmentInfo", {}).get("count")
            })
            
        return parsed_trials
