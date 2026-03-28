from typing import Dict, Any

from prepare import fetch_financial_data

class FinanceAPI:
    def get_company_data(self, ticker: str) -> Dict[str, Any]:
        """Delegate to prepare.py so finance data uses the evaluator's canonical cache path."""
        data = fetch_financial_data(ticker)
        if "_52WeekChange" in data and "52WeekChange" not in data:
            data["52WeekChange"] = data["_52WeekChange"]
        return data
