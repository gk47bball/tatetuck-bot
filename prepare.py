"""
prepare.py — Fixed data-fetching and evaluation harness.
DO NOT MODIFY. This is the read-only evaluation infrastructure.

Equivalent to Karpathy's prepare.py: fixed constants, data loading, and the
ground-truth evaluation function that the agent's strategy is scored against.
"""

import time
import requests
import xml.etree.ElementTree as ET
import yfinance as yf
from typing import List, Dict, Any

# ─── Fixed Constants ────────────────────────────────────────────────────────────

# Benchmark tickers: small/mid-cap biopharma companies with active clinical pipelines.
# The agent's strategy is evaluated across ALL of these.
BENCHMARK_TICKERS = [
    ("CRSP", "CRISPR Therapeutics"),
    ("BEAM", "Beam Therapeutics"),
    ("NTLA", "Intellia Therapeutics"),
    ("EDIT", "Editas Medicine"),
    ("VERV", "Verve Therapeutics"),
    ("PCVX", "Vaxcyte"),
    ("DAWN", "Day One Biopharmaceuticals"),
    ("IMVT", "Immunovant"),
]

# Industry-standard clinical phase transition probabilities (BIO/QLS/Informa)
PHASE_SUCCESS_RATES = {
    "EARLY_PHASE1": 0.055,
    "PHASE1":       0.074,
    "PHASE2":       0.152,
    "PHASE3":       0.590,
    "NDA_BLA":      0.900,
    "APPROVED":     1.000,
}

# ─── Data Fetching (Fixed — agent cannot change how data is gathered) ───────────

def fetch_financial_data(ticker: str) -> Dict[str, Any]:
    """Pull key financial metrics from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="6mo")
        price_6mo_ago = float(hist["Close"].iloc[0]) if len(hist) > 0 else None
        price_now = float(hist["Close"].iloc[-1]) if len(hist) > 0 else None
        trailing_return = ((price_now / price_6mo_ago) - 1) if (price_6mo_ago and price_now) else None
        return {
            "ticker": ticker,
            "shortName": info.get("shortName"),
            "marketCap": info.get("marketCap"),
            "enterpriseValue": info.get("enterpriseValue"),
            "totalRevenue": info.get("totalRevenue"),
            "cash": info.get("totalCash"),
            "debt": info.get("totalDebt"),
            "trailing_6mo_return": trailing_return,
            "price_now": price_now,
        }
    except Exception as e:
        print(f"  [prepare] finance error for {ticker}: {e}")
        return {"ticker": ticker, "trailing_6mo_return": None}


def fetch_clinical_trials(sponsor_name: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Fetch clinical trials from ClinicalTrials.gov API v2."""
    url = "https://clinicaltrials.gov/api/v2/studies"
    params = {"query.spons": sponsor_name, "pageSize": max_results, "format": "json"}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        studies = resp.json().get("studies", [])
        parsed = []
        for study in studies:
            proto = study.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status = proto.get("statusModule", {})
            design = proto.get("designModule", {})
            conditions = proto.get("conditionsModule", {})
            interventions = proto.get("armsInterventionsModule", {})
            outcomes = proto.get("outcomesModule", {})
            parsed.append({
                "nct_id": ident.get("nctId"),
                "title": ident.get("briefTitle"),
                "overall_status": status.get("overallStatus"),
                "phase": design.get("phases", []),
                "conditions": conditions.get("conditions", []),
                "interventions": [i.get("name") for i in interventions.get("interventions", [])],
                "primary_outcomes": [o.get("measure") for o in outcomes.get("primaryOutcomes", [])],
                "enrollment": design.get("enrollmentInfo", {}).get("count"),
            })
        return parsed
    except Exception as e:
        print(f"  [prepare] clinical trials error for {sponsor_name}: {e}")
        return []


def fetch_fda_adverse_events(drug_name: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Fetch adverse event reports from openFDA."""
    url = "https://api.fda.gov/drug/event.json"
    params = {"search": f'patient.drug.openfda.brand_name:"{drug_name}"', "limit": limit}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json().get("results", [])
    except Exception:
        return []


def fetch_pubmed_abstracts(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Fetch paper abstracts from PubMed using NIH E-utilities."""
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    try:
        search_resp = requests.get(f"{base}/esearch.fcgi", params={
            "db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"
        }, timeout=10)
        search_resp.raise_for_status()
        ids = search_resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        time.sleep(0.4)  # rate limit
        fetch_resp = requests.get(f"{base}/efetch.fcgi", params={
            "db": "pubmed", "id": ",".join(ids), "retmode": "xml"
        }, timeout=15)
        fetch_resp.raise_for_status()
        root = ET.fromstring(fetch_resp.content)
        articles = []
        for art in root.findall(".//PubmedArticle"):
            pmid = art.findtext(".//PMID")
            title = art.findtext(".//ArticleTitle")
            abstracts = art.findall(".//AbstractText")
            abstract = " ".join([e.text for e in abstracts if e.text])
            if title and abstract:
                articles.append({"pmid": pmid, "title": title, "abstract": abstract})
        return articles
    except Exception as e:
        print(f"  [prepare] pubmed error: {e}")
        return []


def classify_phase(phase_list: List[str]) -> str:
    """Map ClinicalTrials.gov phase strings to a standard key."""
    if not phase_list:
        return "PHASE1"
    joined = " ".join(phase_list).upper()
    if "PHASE3" in joined or "PHASE 3" in joined:
        return "PHASE3"
    if "PHASE2" in joined or "PHASE 2" in joined:
        return "PHASE2"
    if "PHASE1" in joined or "PHASE 1" in joined:
        return "PHASE1" if "PHASE2" not in joined else "PHASE2"
    if "EARLY" in joined:
        return "EARLY_PHASE1"
    return "PHASE1"


def gather_company_data(ticker: str, company_name: str) -> Dict[str, Any]:
    """Gather ALL available data for a single company. Returns a single dict."""
    print(f"  Gathering data for {ticker} ({company_name})...")
    
    finance = fetch_financial_data(ticker)
    time.sleep(0.3)
    
    trials = fetch_clinical_trials(company_name)
    time.sleep(0.3)
    
    # Extract drug names from trials for FDA/PubMed queries
    drug_names = list(set([
        dx.split()[0].replace(",", "").replace(".", "")
        for t in trials for dx in t.get("interventions", [])
        if dx and len(dx.split()[0]) > 2
    ]))
    
    fda_events = []
    for dn in drug_names[:2]:
        fda_events.extend(fetch_fda_adverse_events(dn))
        time.sleep(0.3)
    
    pubmed_papers = []
    if drug_names:
        query = " OR ".join(f'"{d}"' for d in drug_names[:3]) + " AND clinical trial"
        pubmed_papers = fetch_pubmed_abstracts(query, max_results=5)
    
    # Determine most advanced phase
    best_phase = "PHASE1"
    phase_order = ["EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "NDA_BLA", "APPROVED"]
    for t in trials:
        p = classify_phase(t.get("phase", []))
        if phase_order.index(p) > phase_order.index(best_phase):
            best_phase = p
    
    return {
        "ticker": ticker,
        "company_name": company_name,
        "finance": finance,
        "trials": trials,
        "num_trials": len(trials),
        "drug_names": drug_names,
        "best_phase": best_phase,
        "base_pos": PHASE_SUCCESS_RATES.get(best_phase, 0.074),
        "fda_adverse_events": len(fda_events),
        "pubmed_papers": pubmed_papers,
        "num_papers": len(pubmed_papers),
        "conditions": list(set(c for t in trials for c in t.get("conditions", []))),
    }


# ─── Evaluation (Fixed — the ground-truth scoring function) ─────────────────────

def evaluate_strategy(strategy_module) -> Dict[str, Any]:
    """
    Run the agent's strategy against all benchmark tickers and compute the
    evaluation metric: mean absolute error of predicted signal vs actual return.
    
    The strategy module must expose: score_company(data: dict) -> dict
    The returned dict must contain a "signal" key (float, -1 to +1).
    
    Returns a summary dict with the metric and per-ticker details.
    """
    results = []
    total_error = 0.0
    scored = 0
    
    for ticker, name in BENCHMARK_TICKERS:
        data = gather_company_data(ticker, name)
        actual_return = data["finance"].get("trailing_6mo_return")
        
        try:
            prediction = strategy_module.score_company(data)
            signal = prediction.get("signal", 0.0)
        except Exception as e:
            print(f"  [evaluate] strategy crashed on {ticker}: {e}")
            signal = 0.0
            prediction = {"signal": 0.0, "error": str(e)}
        
        # Normalize actual return to -1..+1 range (cap at ±100%)
        if actual_return is not None:
            actual_signal = max(-1.0, min(1.0, actual_return))
            error = abs(signal - actual_signal)
            total_error += error
            scored += 1
        else:
            actual_signal = None
            error = None
        
        results.append({
            "ticker": ticker,
            "predicted_signal": round(signal, 4),
            "actual_6mo_return": round(actual_return, 4) if actual_return is not None else None,
            "actual_signal": round(actual_signal, 4) if actual_signal is not None else None,
            "error": round(error, 4) if error is not None else None,
            "prediction_details": prediction,
        })
    
    valuation_error = total_error / scored if scored > 0 else 99.0
    
    return {
        "valuation_error": valuation_error,
        "num_scored": scored,
        "num_total": len(BENCHMARK_TICKERS),
        "per_ticker": results,
    }
