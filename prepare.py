"""
prepare.py — Fixed data-fetching and evaluation harness.
DO NOT MODIFY. This is the read-only evaluation infrastructure.

Equivalent to Karpathy's prepare.py: fixed constants, data loading, and the
ground-truth evaluation function that the agent's strategy is scored against.
"""

import os
import json
import hashlib
import time
import requests
import xml.etree.ElementTree as ET
import yfinance as yf
from typing import List, Dict, Any

# ─── Fixed Constants ────────────────────────────────────────────────────────────

# Benchmark tickers: mid-cap biopharma companies with active clinical pipelines.
# Increasing this number makes the metric more robust but slows down experiments.
# We keep this set FIXED during a research session to ensure scores are comparable.
BENCHMARK_TICKERS = [
    ("CRSP", "CRISPR Therapeutics"),
    ("BEAM", "Beam Therapeutics"),
    ("NTLA", "Intellia Therapeutics"),
    ("EDIT", "Editas Medicine"),
    ("PCVX", "Vaxcyte"),
    ("DAWN", "Day One Biopharmaceuticals"),
    ("IMVT", "Immunovant"),
    ("PRME", "Prime Medicine"),
    ("MDGL", "Madrigal Pharmaceuticals"),
    ("BPMC", "Blueprint Medicines"),
    ("KROS", "Keros Therapeutics"),
    ("RCKT", "Rocket Pharmaceuticals"),
    ("SRPT", "Sarepta Therapeutics"),
    ("ARVN", "Arvinas"),
    ("RYTM", "Rhythm Pharmaceuticals"),
    ("APLS", "Apellis Pharmaceuticals"),
    ("FATE", "Fate Therapeutics"),
    ("BBIO", "BridgeBio Pharma"),
    ("IOVA", "Iovance Biotherapeutics"),
    ("VRTX", "Vertex Pharmaceuticals"),
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

# ─── Disk Cache ──────────────────────────────────────────────────────────────────
# Cache API responses for 1 hour so repeated experiment runs don't re-fetch.

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data_cache")
CACHE_TTL_SECONDS = 3600  # 1 hour

def _cache_key(prefix: str, *args) -> str:
    raw = f"{prefix}:{':'.join(str(a) for a in args)}"
    return hashlib.md5(raw.encode()).hexdigest()

def _cache_get(key: str):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        age = time.time() - os.path.getmtime(path)
        if age < CACHE_TTL_SECONDS:
            with open(path, "r") as f:
                return json.load(f)
    return None

def _cache_set(key: str, data):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    with open(path, "w") as f:
        json.dump(data, f)


# ─── Data Fetching (Fixed — agent cannot change how data is gathered) ───────────

def fetch_financial_data(ticker: str) -> Dict[str, Any]:
    """Pull key financial metrics from Yahoo Finance."""
    key = _cache_key("finance", ticker)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="6mo")
        price_6mo_ago = float(hist["Close"].iloc[0]) if len(hist) > 0 else None
        price_now = float(hist["Close"].iloc[-1]) if len(hist) > 0 else None
        trailing_return = ((price_now / price_6mo_ago) - 1) if (price_6mo_ago and price_now) else None
        result = {
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
        _cache_set(key, result)
        return result
    except Exception as e:
        print(f"  [prepare] finance error for {ticker}: {e}")
        return {"ticker": ticker, "trailing_6mo_return": None}


def fetch_clinical_trials(sponsor_name: str, max_results: int = 50) -> List[Dict[str, Any]]:
    """Fetch clinical trials from ClinicalTrials.gov API v2."""
    key = _cache_key("trials", sponsor_name, max_results)
    cached = _cache_get(key)
    if cached is not None:
        return cached
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
        _cache_set(key, parsed)
        return parsed
    except Exception as e:
        print(f"  [prepare] clinical trials error for {sponsor_name}: {e}")
        return []


def fetch_fda_adverse_events(drug_name: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Fetch adverse event reports from openFDA."""
    key = _cache_key("fda", drug_name, limit)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    url = "https://api.fda.gov/drug/event.json"
    params = {"search": f'patient.drug.openfda.brand_name:"{drug_name}"', "limit": limit}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        result = resp.json().get("results", [])
        _cache_set(key, result)
        return result
    except Exception:
        return []


def fetch_pubmed_abstracts(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Fetch paper abstracts from PubMed using NIH E-utilities."""
    key = _cache_key("pubmed", query, max_results)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    try:
        search_resp = requests.get(f"{base}/esearch.fcgi", params={
            "db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"
        }, timeout=10)
        search_resp.raise_for_status()
        ids = search_resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            _cache_set(key, [])
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
        _cache_set(key, articles)
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
    time.sleep(0.2)
    
    trials = fetch_clinical_trials(company_name)
    time.sleep(0.2)
    
    # Extract drug names from trials for FDA/PubMed queries
    drug_names = list(set([
        dx.split()[0].replace(",", "").replace(".", "")
        for t in trials for dx in t.get("interventions", [])
        if dx and len(dx.split()[0]) > 2
    ]))
    
    fda_events = []
    for dn in drug_names[:2]:
        fda_events.extend(fetch_fda_adverse_events(dn))
        time.sleep(0.2)
    
    # Extract seriousness from FDA events
    fda_serious_count = 0
    for evt in fda_events:
        if evt.get("serious") == 1 or evt.get("seriousnessdeath") == 1:
            fda_serious_count += 1
    
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
        "fda_serious_events": fda_serious_count,
        "pubmed_papers": pubmed_papers,
        "num_papers": len(pubmed_papers),
        "conditions": list(set(c for t in trials for c in t.get("conditions", []))),
    }


# ─── Evaluation (Fixed — the ground-truth scoring function) ─────────────────────

def _spearman_rank_correlation(x: List[float], y: List[float]) -> float:
    """Compute Spearman rank correlation between two lists."""
    n = len(x)
    if n < 3:
        return 0.0
    
    def _rank(values):
        sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        for rank_pos, idx in enumerate(sorted_indices):
            ranks[idx] = float(rank_pos + 1)
        # Handle ties: average the ranks
        i = 0
        while i < len(values):
            j = i
            while j < len(values) and values[sorted_indices[j]] == values[sorted_indices[i]]:
                j += 1
            if j > i + 1:
                avg_rank = sum(range(i + 1, j + 1)) / (j - i)
                for k in range(i, j):
                    ranks[sorted_indices[k]] = avg_rank
            i = j
        return ranks
    
    rx = _rank(x)
    ry = _rank(y)
    
    d_sq_sum = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    rho = 1 - (6 * d_sq_sum) / (n * (n ** 2 - 1))
    return rho


def evaluate_strategy(strategy_module) -> Dict[str, Any]:
    """
    Run the agent's strategy against all benchmark tickers and compute the
    evaluation metric: a composite score combining:
    
    1. Spearman Rank Correlation — did we rank companies in the right order?
    2. Directional Accuracy — did we predict the right direction (up/down)?
    3. Mean Absolute Error — how far off were we on magnitude?
    
    The composite metric is: valuation_error = 1 - (0.50 * rank_corr + 0.30 * dir_accuracy + 0.20 * (1 - MAE))
    Lower is better. A perfect score is 0.0.
    
    The strategy module must expose: score_company(data: dict) -> dict
    The returned dict must contain a "signal" key (float, -1 to +1).
    """
    results = []
    predictions = []
    actuals = []
    total_error = 0.0
    correct_direction = 0
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
            
            # Directional accuracy (both non-zero)
            if signal != 0.0:
                if (signal > 0 and actual_return > 0) or (signal < 0 and actual_return < 0):
                    correct_direction += 1
            # Predicting 0.0 counts as wrong direction (no conviction = no credit)
            
            predictions.append(signal)
            actuals.append(actual_signal)
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
    
    # Compute composite metric
    if scored >= 3:
        rank_corr = _spearman_rank_correlation(predictions, actuals)
        # Normalize rank_corr from [-1, 1] to [0, 1] for the composite
        rank_score = (rank_corr + 1.0) / 2.0
        dir_accuracy = correct_direction / scored
        mae = total_error / scored
        mae_score = max(0.0, 1.0 - mae)  # lower MAE = higher score
        
        # Composite: higher = better, so valuation_error = 1 - composite
        composite = 0.50 * rank_score + 0.30 * dir_accuracy + 0.20 * mae_score
        valuation_error = 1.0 - composite
    else:
        rank_corr = 0.0
        dir_accuracy = 0.0
        mae = total_error / scored if scored > 0 else 99.0
        valuation_error = 99.0
    
    return {
        "valuation_error": valuation_error,
        "rank_correlation": round(rank_corr, 4),
        "directional_accuracy": round(dir_accuracy, 4),
        "mae": round(mae, 4) if scored > 0 else None,
        "num_scored": scored,
        "num_total": len(BENCHMARK_TICKERS),
        "per_ticker": results,
    }
