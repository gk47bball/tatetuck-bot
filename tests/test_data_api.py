import unittest
from biopharma_agent.data.clinical_trials import ClinicalTrialsAPI
from biopharma_agent.data.fda import OpenFDAAPI

class TestDataAPI(unittest.TestCase):
    def test_clinical_trials(self):
        api = ClinicalTrialsAPI()
        results = api.search_by_sponsor("CRISPR Therapeutics", max_results=1)
        self.assertIsInstance(results, list)
        
    def test_fda_api(self):
        api = OpenFDAAPI()
        results = api.get_adverse_events("Acetaminophen", limit=1)
        self.assertIsInstance(results, list)

if __name__ == "__main__":
    unittest.main()
