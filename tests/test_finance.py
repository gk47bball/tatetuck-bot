import unittest
from biopharma_agent.data.finance import FinanceAPI

class TestFinanceAPI(unittest.TestCase):
    def test_finance_api(self):
        api = FinanceAPI()
        data = api.get_company_data("CRSP")
        self.assertEqual(data.get("ticker"), "CRSP")
        self.assertIsNotNone(data.get("marketCap"))

if __name__ == "__main__":
    unittest.main()
