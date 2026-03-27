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
from typing import List, Dict, Any, Optional

# ─── Fixed Constants ────────────────────────────────────────────────────────────

# TRAIN set: 14 tickers the agent optimizes against.
# HOLDOUT set: 6 tickers used ONLY for out-of-sample validation.
# The agent NEVER sees holdout results during the loop — only at the end.

TRAIN_TICKERS = [
    ("CRSP", "CRISPR Therapeutics"),
    ("BEAM", "Beam Therapeutics"),
    ("NTLA", "Intellia Therapeutics"),
    ("EDIT", "Editas Medicine"),
    ("PCVX", "Vaxcyte"),
    ("DAWN", "Day One Biopharmaceuticals"),
    ("IMVT", "Immunovant"),
    ("PRME", "Prime Medicine"),
    ("MDGL", "Madrigal Pharmaceuticals"),
    ("KROS", "Keros Therapeutics"),
    ("SRPT", "Sarepta Therapeutics"),
    ("ARVN", "Arvinas"),
    ("APLS", "Apellis Pharmaceuticals"),
    ("BBIO", "BridgeBio Pharma"),
]

HOLDOUT_TICKERS = [
    ("RCKT", "Rocket Pharmaceuticals"),
    ("RYTM", "Rhythm Pharmaceuticals"),
    ("FATE", "Fate Therapeutics"),
    ("IOVA", "Iovance Biotherapeutics"),
    ("VRTX", "Vertex Pharmaceuticals"),
    ("BPMC", "Blueprint Medicines"),
]

# All tickers combined (for full evaluation)
BENCHMARK_TICKERS = TRAIN_TICKERS + HOLDOUT_TICKERS

# Industry-standard clinical phase transition probabilities (BIO/QLS/Informa)
PHASE_SUCCESS_RATES = {
    "EARLY_PHASE1": 0.055,
    "PHASE1":       0.074,
    "PHASE2":       0.152,
    "PHASE3":       0.590,
    "NDA_BLA":      0.900,
    "APPROVED":     1.000,
}

# Trial statuses considered "active" (everything else is filtered out)
ACTIVE_TRIAL_STATUSES = {
    "RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION",
    "NOT_YET_RECRUITING", "AVAILABLE",
    # Also include unknown/missing status to avoid false negatives
}

# ─── Disk Cache ──────────────────────────────────────────────────────────────────

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data_cache")
CACHE_TTL_SECONDS = 3600  # 1 hour

def _cache_key(prefix: str, *args) -> str:
    raw = f"{prefix}:{':'.join(str(a) for a in args)}"
    return hashlib.md5(raw.encode()).hexdigest()

def _cache_get(key: str) -> Optional[Any]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        age = time.time() - os.path.getmtime(path)
        if age < CACHE_TTL_SECONDS:
            with open(path, "r") as f:
                return json.load(f)
    return None

def _cache_set(key: str, data: Any) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    with open(path, "w") as f:
        json.dump(data, f)


# ─── Retry Logic ─────────────────────────────────────────────────────────────────

def _request_with_retry(url: str, params: dict, timeout: int = 15,
                        max_retries: int = 3) -> Optional[requests.Response]:
    """Make an HTTP GET with exponential backoff. Does NOT retry on 404s."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 404:
                return None  # Expected for pre-clinical drugs, don't retry
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                return None


# ─── Data Fetching (Fixed — agent cannot change how data is gathered) ───────────

def fetch_financial_data(ticker: str) -> Dict[str, Any]:
    """Pull key financial metrics from Yahoo Finance, including 3-month momentum."""
    key = _cache_key("finance_v2", ticker)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="6mo")

        if len(hist) == 0:
            return {"ticker": ticker, "trailing_6mo_return": None, "momentum_3mo": None}

        price_6mo_ago = float(hist["Close"].iloc[0])
        price_now = float(hist["Close"].iloc[-1])
        trailing_return = (price_now / price_6mo_ago) - 1 if price_6mo_ago else None

        # 3-month momentum: return over the most recent half of the price history
        midpoint = len(hist) // 2
        price_3mo_ago = float(hist["Close"].iloc[midpoint]) if midpoint > 0 else None
        momentum_3mo = (price_now / price_3mo_ago) - 1 if price_3mo_ago else None

        # Volatility: standard deviation of daily returns
        daily_returns = hist["Close"].pct_change().dropna()
        volatility = float(daily_returns.std()) if len(daily_returns) > 5 else None

        result = {
            "ticker": ticker,
            "shortName": info.get("shortName"),
            "marketCap": info.get("marketCap"),
            "enterpriseValue": info.get("enterpriseValue"),
            "totalRevenue": info.get("totalRevenue"),
            "cash": info.get("totalCash"),
            "debt": info.get("totalDebt"),
            "trailing_6mo_return": trailing_return,
            "momentum_3mo": momentum_3mo,
            "volatility": volatility,
            "price_now": price_now,
        }
        _cache_set(key, result)
        return result
    except Exception as e:
        print(f"  [prepare] finance error for {ticker}: {e}")
        return {"ticker": ticker, "trailing_6mo_return": None, "momentum_3mo": None}


def fetch_clinical_trials(sponsor_name: str, max_results: int = 50) -> List[Dict[str, Any]]:
    """Fetch clinical trials from ClinicalTrials.gov API v2."""
    key = _cache_key("trials_v2", sponsor_name, max_results)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    url = "https://clinicaltrials.gov/api/v2/studies"
    params = {"query.spons": sponsor_name, "pageSize": max_results, "format": "json"}
    resp = _request_with_retry(url, params, timeout=15)
    if resp is None:
        return []
    try:
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
            overall_status = status.get("overallStatus", "")

            parsed.append({
                "nct_id": ident.get("nctId"),
                "title": ident.get("briefTitle"),
                "overall_status": overall_status,
                "phase": design.get("phases", []),
                "conditions": conditions.get("conditions", []),
                "interventions": [i.get("name") for i in interventions.get("interventions", [])],
                "primary_outcomes": [o.get("measure") for o in outcomes.get("primaryOutcomes", [])],
                "enrollment": design.get("enrollmentInfo", {}).get("count"),
            })
        _cache_set(key, parsed)
        return parsed
    except Exception as e:
        print(f"  [prepare] clinical trials parse error for {sponsor_name}: {e}")
        return []


def fetch_fda_adverse_events(drug_name: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch adverse event reports from openFDA."""
    key = _cache_key("fda_v2", drug_name, limit)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    url = "https://api.fda.gov/drug/event.json"
    params = {"search": f'patient.drug.openfda.brand_name:"{drug_name}"', "limit": limit}
    resp = _request_with_retry(url, params, timeout=10)
    if resp is None:
        _cache_set(key, [])
        return []
    try:
        result = resp.json().get("results", [])
        _cache_set(key, result)
        return result
    except Exception:
        _cache_set(key, [])
        return []


def fetch_pubmed_abstracts(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Fetch paper abstracts from PubMed using NIH E-utilities."""
    key = _cache_key("pubmed_v2", query, max_results)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    resp = _request_with_retry(f"{base}/esearch.fcgi", {
        "db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"
    }, timeout=10)
    if resp is None:
        return []
    try:
        ids = resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            _cache_set(key, [])
            return []
        time.sleep(0.4)  # rate limit
        fetch_resp = _request_with_retry(f"{base}/efetch.fcgi", {
            "db": "pubmed", "id": ",".join(ids), "retmode": "xml"
        }, timeout=15)
        if fetch_resp is None:
            return []
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
    time.sleep(0.15)

    all_trials = fetch_clinical_trials(company_name)
    time.sleep(0.15)

    # ── Filter: only active/recruiting trials ──
    active_trials = []
    inactive_trials = []
    for t in all_trials:
        status = (t.get("overall_status") or "").upper().replace(" ", "_")
        if status in ACTIVE_TRIAL_STATUSES or not status:
            active_trials.append(t)
        else:
            inactive_trials.append(t)

    # Extract drug names from active trials for FDA/PubMed queries
    drug_names = list(set([
        dx.split()[0].replace(",", "").replace(".", "")
        for t in active_trials for dx in t.get("interventions", [])
        if dx and len(dx.split()[0]) > 2
    ]))

    fda_events = []
    for dn in drug_names[:3]:
        fda_events.extend(fetch_fda_adverse_events(dn))
        time.sleep(0.15)

    # Extract seriousness from FDA events
    fda_serious_count = 0
    fda_total_count = len(fda_events)
    for evt in fda_events:
        if evt.get("serious") == 1 or evt.get("seriousnessdeath") == 1:
            fda_serious_count += 1
    fda_serious_ratio = fda_serious_count / fda_total_count if fda_total_count > 0 else 0.0

    pubmed_papers = []
    if drug_names:
        query = " OR ".join(f'"{d}"' for d in drug_names[:3]) + " AND clinical trial"
        pubmed_papers = fetch_pubmed_abstracts(query, max_results=5)

    # ── Phase detection with enrollment weighting ──
    best_phase = "PHASE1"
    phase_order = ["EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "NDA_BLA", "APPROVED"]

    # Enrollment-weighted stats
    total_enrollment = 0
    phase_enrollment = {}  # phase -> total enrollment in that phase
    phase_trial_counts = {}  # phase -> count of active trials

    for t in active_trials:
        p = classify_phase(t.get("phase", []))
        enrollment = t.get("enrollment") or 0

        # Track best phase
        if phase_order.index(p) > phase_order.index(best_phase):
            best_phase = p

        total_enrollment += enrollment
        phase_enrollment[p] = phase_enrollment.get(p, 0) + enrollment
        phase_trial_counts[p] = phase_trial_counts.get(p, 0) + 1

    # Max enrollment in any single trial (indicator of conviction)
    max_single_enrollment = max(
        (t.get("enrollment") or 0 for t in active_trials), default=0
    )

    return {
        "ticker": ticker,
        "company_name": company_name,
        "finance": finance,
        "trials": active_trials,
        "num_trials": len(active_trials),
        "num_total_trials": len(all_trials),
        "num_inactive_trials": len(inactive_trials),
        "drug_names": drug_names,
        "best_phase": best_phase,
        "base_pos": PHASE_SUCCESS_RATES.get(best_phase, 0.074),
        "total_enrollment": total_enrollment,
        "max_single_enrollment": max_single_enrollment,
        "phase_enrollment": phase_enrollment,
        "phase_trial_counts": phase_trial_counts,
        "fda_adverse_events": fda_total_count,
        "fda_serious_events": fda_serious_count,
        "fda_serious_ratio": round(fda_serious_ratio, 4),
        "pubmed_papers": pubmed_papers,
        "num_papers": len(pubmed_papers),
        "conditions": list(set(c for t in active_trials for c in t.get("conditions", []))),
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


def _compute_metrics(predictions: List[float], actuals: List[float],
                     scored: int, total_error: float,
                     correct_direction: int) -> Dict[str, Any]:
    """Compute the composite metric from predictions and actuals."""
    if scored >= 3:
        rank_corr = _spearman_rank_correlation(predictions, actuals)
        rank_score = (rank_corr + 1.0) / 2.0
        dir_accuracy = correct_direction / scored
        mae = total_error / scored
        mae_score = max(0.0, 1.0 - mae)
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
    }


def evaluate_strategy(strategy_module) -> Dict[str, Any]:
    """
    Run the agent's strategy against all benchmark tickers and compute
    evaluation metrics on TRAIN set (used for optimization) and HOLDOUT
    set (used for overfitting detection).

    Composite metric: valuation_error = 1 - (0.50 * rank_corr + 0.30 * dir_accuracy + 0.20 * (1 - MAE))
    Lower is better.
    """
    all_results = []

    # Gather data for all tickers
    all_data = {}
    for ticker, name in BENCHMARK_TICKERS:
        data = gather_company_data(ticker, name)
        all_data[ticker] = data

    train_tickers_set = {t[0] for t in TRAIN_TICKERS}
    holdout_tickers_set = {t[0] for t in HOLDOUT_TICKERS}

    # Accumulators for train and holdout
    train_preds, train_actuals, train_error, train_scored, train_dir = [], [], 0.0, 0, 0
    holdout_preds, holdout_actuals, holdout_error, holdout_scored, holdout_dir = [], [], 0.0, 0, 0

    for ticker, name in BENCHMARK_TICKERS:
        data = all_data[ticker]
        actual_return = data["finance"].get("trailing_6mo_return")

        try:
            prediction = strategy_module.score_company(data)
            signal = prediction.get("signal", 0.0)
        except Exception as e:
            print(f"  [evaluate] strategy crashed on {ticker}: {e}")
            signal = 0.0
            prediction = {"signal": 0.0, "error": str(e)}

        if actual_return is not None:
            actual_signal = max(-1.0, min(1.0, actual_return))
            error = abs(signal - actual_signal)

            dir_correct = 0
            if signal != 0.0:
                if (signal > 0 and actual_return > 0) or (signal < 0 and actual_return < 0):
                    dir_correct = 1

            if ticker in train_tickers_set:
                train_preds.append(signal)
                train_actuals.append(actual_signal)
                train_error += error
                train_scored += 1
                train_dir += dir_correct
            elif ticker in holdout_tickers_set:
                holdout_preds.append(signal)
                holdout_actuals.append(actual_signal)
                holdout_error += error
                holdout_scored += 1
                holdout_dir += dir_correct
        else:
            actual_signal = None
            error = None

        all_results.append({
            "ticker": ticker,
            "set": "train" if ticker in train_tickers_set else "holdout",
            "predicted_signal": round(signal, 4),
            "actual_6mo_return": round(actual_return, 4) if actual_return is not None else None,
            "actual_signal": round(actual_signal, 4) if actual_signal is not None else None,
            "error": round(error, 4) if error is not None else None,
            "prediction_details": prediction,
        })

    train_metrics = _compute_metrics(train_preds, train_actuals, train_scored, train_error, train_dir)
    holdout_metrics = _compute_metrics(holdout_preds, holdout_actuals, holdout_scored, holdout_error, holdout_dir)

    # Overfitting detection
    overfit_gap = abs(train_metrics["valuation_error"] - holdout_metrics["valuation_error"])
    overfit_warning = overfit_gap > 0.15

    return {
        "valuation_error": train_metrics["valuation_error"],  # Primary metric (train only)
        "rank_correlation": train_metrics["rank_correlation"],
        "directional_accuracy": train_metrics["directional_accuracy"],
        "mae": train_metrics["mae"],
        "num_scored": train_metrics["num_scored"],
        "num_total": len(BENCHMARK_TICKERS),
        "holdout_valuation_error": holdout_metrics["valuation_error"],
        "holdout_rank_correlation": holdout_metrics["rank_correlation"],
        "holdout_directional_accuracy": holdout_metrics["directional_accuracy"],
        "overfit_gap": round(overfit_gap, 4),
        "overfit_warning": overfit_warning,
        "per_ticker": all_results,
    }
