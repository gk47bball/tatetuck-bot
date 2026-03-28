from prepare import gather_company_data, BENCHMARK_TICKERS
import strategy

for ticker, name in BENCHMARK_TICKERS:
    data = gather_company_data(ticker, name)
    res = strategy.score_company(data)
    ab = res.get("alpha_breakdown", {})
    print(f"{ticker:4} | Sig: {res['signal']:+.3f} | V: {ab.get('value',0):+.3f} | C: {ab.get('clinical',0):+.3f} | S: {ab.get('safety',0):+.3f} | F: {ab.get('finance',0):+.3f} | M: {ab.get('momentum',0):+.3f}")
