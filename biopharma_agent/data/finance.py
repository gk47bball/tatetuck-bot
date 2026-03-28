import yfinance as yf
from typing import Dict, Any

class FinanceAPI:
    def get_company_data(self, ticker: str) -> Dict[str, Any]:
        """Fetch general financial and description data for a ticker using yfinance."""
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            "ticker": ticker,
            "shortName": info.get("shortName"),
            "longName": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "marketCap": info.get("marketCap"),
            "enterpriseValue": info.get("enterpriseValue"),
            "totalRevenue": info.get("totalRevenue"),
            "grossMargins": info.get("grossMargins"),
            "operatingMargins": info.get("operatingMargins"),
            "cash": info.get("totalCash"),
            "debt": info.get("totalDebt"),
            "netIncome": info.get("netIncomeToCommon"),
            "52WeekChange": info.get("52WeekChange"),
            "description": info.get("longBusinessSummary")
        }
