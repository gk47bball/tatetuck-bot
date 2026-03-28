import unittest
from unittest.mock import patch

from biopharma_agent.data.clinical_trials import ClinicalTrialsAPI
from biopharma_agent.data.fda import OpenFDAAPI


class TestDataAPI(unittest.TestCase):
    @patch("biopharma_agent.data.clinical_trials.fetch_clinical_trials")
    def test_clinical_trials(self, mock_fetch):
        mock_fetch.return_value = [{"nct_id": "NCT1", "title": "Trial"}]
        api = ClinicalTrialsAPI()
        results = api.search_by_sponsor("CRISPR Therapeutics", max_results=1)
        self.assertEqual(results[0]["nct_id"], "NCT1")
        mock_fetch.assert_called_once_with("CRISPR Therapeutics", max_results=1)

    @patch("biopharma_agent.data.fda.fetch_fda_adverse_events")
    def test_fda_api(self, mock_fetch):
        mock_fetch.return_value = [{"serious": 1}]
        api = OpenFDAAPI()
        results = api.get_adverse_events("Acetaminophen", limit=1)
        self.assertEqual(len(results), 1)
        mock_fetch.assert_called_once_with("Acetaminophen", limit=1)


if __name__ == "__main__":
    unittest.main()
