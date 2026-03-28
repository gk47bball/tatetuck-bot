import unittest
from unittest.mock import patch

from biopharma_agent.data.finance import FinanceAPI


class TestFinanceAPI(unittest.TestCase):
    @patch("biopharma_agent.data.finance.fetch_financial_data")
    def test_finance_api(self, mock_fetch):
        mock_fetch.return_value = {"ticker": "CRSP", "marketCap": 123, "52WeekChange": 0.1}
        api = FinanceAPI()
        data = api.get_company_data("CRSP")
        self.assertEqual(data.get("ticker"), "CRSP")
        self.assertEqual(data.get("marketCap"), 123)
        mock_fetch.assert_called_once_with("CRSP")


if __name__ == "__main__":
    unittest.main()
