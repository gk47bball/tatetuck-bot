# Tatetuck Bot

**Tatetuck Bot** is an event-driven biopharma alpha platform for retail investors that rivals institutional-grade biotech fund workflows. It systematically identifies and sizes positions in small/mid-cap biotech stocks ahead of binary catalysts — Phase 3 readouts, FDA PDUFA dates, and clinical holds — using empirically calibrated probability-of-success priors, risk-adjusted NPV valuation, and walk-forward machine learning trained on real catalyst outcomes.

The platform ingests SEC filings, ClinicalTrials.gov data, and live market data; scores every drug program on a multi-dimensional feature vector; trains a ridge regression ensemble on historical outcomes; and generates portfolio-weight recommendations with paper execution orders via Alpaca. Results surface daily through a Discord bot.

---

## Architecture Overview

```
tatetuck-bot/
├── discord_bot.py              # Discord interface — !analyze, !top5, !status
├── trade_vnext.py              # Paper trade execution pipeline
├── operate_vnext.py            # Full research pipeline (archive → label → eval)
├── evaluate_vnext.py           # Model smoke test and walk-forward evaluation
├── bootstrap_history_vnext.py  # Historical snapshot rebuild from filing anchors
├── sync_universe_vnext.py      # Active/delisted universe sync from EODHD
├── main.py                     # Legacy CLI report generator
├── strategy.py                 # Core POS/rNPV scoring engine
└── biopharma_agent/vnext/      # Primary research platform
    ├── facade.py               # TatetuckPlatform — main analysis entry point
    ├── graph.py                # Company → Program → CatalystEvent graph builder
    ├── features.py             # Feature engineering (60+ signals per program)
    ├── models.py               # EventDrivenEnsemble (ridge regression + rules)
    ├── portfolio.py            # PortfolioConstructor — scenario-based sizing
    ├── execution.py            # PMExecutionPlanner + AlpacaPaperBroker
    ├── evaluation.py           # WalkForwardEvaluator — temporal backtesting
    ├── labels.py               # Return labeling with XBI benchmark-relative alpha
    ├── entities.py             # Typed dataclasses: CompanySnapshot, Program, Trial, etc.
    ├── archive.py              # Universe archival to Parquet store
    ├── replay.py               # Historical snapshot reconstruction
    ├── market_profile.py       # Valuation and expectation lens (rNPV)
    ├── failure_universe.py     # 48 curated failures for survivorship-bias correction
    ├── kol_proxy.py            # Academic Medical Center site quality scoring
    ├── taxonomy.py             # Event type classification and bucketing
    ├── settings.py             # VNextSettings — all config via env vars
    └── ops.py                  # Readiness audit and health reporting
```

---

## Scientific Design

### Catalyst-First Framework

The core thesis is that binary catalyst outcomes are partially predictable from public information, and that the market systematically misprices this risk in small biotech because retail flow dominates and institutional research coverage is thin.

**Probability of Success (POS)** priors are anchored to BIO/Informa 2023 empirical likelihood-of-approval data by phase:

| Phase | Base LoA |
|---|---|
| Early Phase 1 | 5.8% |
| Phase 1 | 7.9% |
| Phase 2 | 14.9% |
| Phase 3 | 57.0% |
| NDA/BLA | 90.0% |

These priors are adjusted by indication multipliers reflecting empirical LoA divergence from the cross-indication average. Critically, **oncology multiplier is 0.82** (Phase 3 LoA ~47%, well below average) — the single most common mis-ranking error in retail biotech investing.

| Indication | Multiplier | Rationale |
|---|---|---|
| Hematology | 1.33 | ~76% empirical LoA |
| Infectious disease | 1.18 | ~68% empirical LoA |
| Rare disease | 1.20 | Breakthrough/orphan pathways |
| Immunology | 1.05 | Above average LoA |
| Cardiovascular | 0.96 | Near average |
| Gene/cell therapy | 0.95 | Manufacturing risk |
| Metabolic | 0.90 | Endpoint ambiguity |
| Neurology | 0.85 | High Phase 3 attrition |
| **Oncology** | **0.82** | **~47% LoA, below average** |

### Risk-Adjusted NPV (rNPV)

Valuation uses phase-weighted discount rates rather than a flat WACC:

| Phase | Discount Rate |
|---|---|
| Approved / NDA | 12% |
| Phase 3 | 15% |
| Phase 2 | 18% |
| Phase 1 / Early | 20% |

Market penetration rates are averaged across indication peer sets (not naively maximized), and valuation component weights are normalized to sum ≤ 1.0 to prevent systematic overvaluation.

### Feature Engineering (60+ Signals)

Features are organized into five families:

- **Program quality**: POS prior, modality risk, endpoint quality (FDA-calibrated — OS/death: 0.90, ORR/surrogate: 0.55, biomarker: 0.30), KOL proxy via AMC site scoring, interaction terms
- **Catalyst timing**: Horizon days, event type priority, options IV implied move, filing freshness, stale catalyst detection
- **Balance sheet**: Cash-to-cap, debt-to-cap, runway months, financing pressure, floor support
- **Commercial execution**: Revenue scale, launch progress, lifecycle management score, TAM/cap ratio
- **Market flow**: 3-month momentum, volatility, crowding signal (momentum-decoupled)

**Interaction features** capture non-linear relationships invisible to linear models:
- `interaction_quality_x_timing`: POS × near-term urgency
- `interaction_phase_x_moat`: Phase score × competitive positioning
- `interaction_value_gap_x_catalyst`: TAM/cap ratio × clinical focus
- `interaction_cash_stress_x_horizon`: Financing pressure × runway × near-term

### IV Implied Move Signal

At runtime, fetches the ATM straddle price from the options chain bracketing the catalyst date: `(call_ask + put_ask) / spot`. When the model's expected return exceeds the options-implied move, the setup is underpriced; when below, the binary is already fully reflected.

For historical training rows (where live options chains are unavailable), a Black-Scholes proxy fills the signal: `σ × √(T/252) × 0.798` (Brenner-Subrahmanyam 1988 approximation), giving every training row a non-zero, economically grounded IV proxy.

### KOL Proxy via AMC Site Scoring

Top pharma funds (Baker Brothers, RA Capital) have direct KOL relationships. This platform substitutes with public site quality scoring:

- **Tier 1 AMCs**: MGH, MD Anderson, Hopkins, MSK, Memorial Sloan Kettering, Mayo Clinic, Stanford, UCSF, Dana-Farber, Penn, Vanderbilt, and 20+ others
- **Tier 2 AMCs**: Strong regional and international equivalents
- Composite score: `AMC presence × 0.60 + site diversity × 0.25 + international reach × 0.15`

### Survivorship Bias Correction

The training set is augmented with 48 curated real biopharma failures (2019–2025) across Phase 3 failures, CRLs, bankruptcies, clinical holds, and commercial failures. Categories include oncology, CNS, NASH/metabolic, rare disease, gene therapy, vaccines, and cardiovascular. Without this, the model only sees the survivors it happens to track — systematically overstating confidence.

### Walk-Forward Backtesting

- Temporal train/test splits prevent look-ahead bias
- Labels are **XBI benchmark-relative** (`target_alpha_90d = stock_return − XBI_return`) to strip biotech sector beta from the training signal
- Minimum 50 training rows per window
- Drawdown computed on cumulative product (not sum)
- `alpha_rank_ic` and `stale_catalyst_rate` reported per window

### Position Sizing

- **Kelly correlation discount**: `weight × 1/√N` for N catalyst events co-occurring in the same 21-day window — prevents correlated binary bet blowup
- **ADV cap**: Single-session orders capped at 5% of 20-day average dollar volume to avoid market impact
- **Weight caps**: 4% max for pre-catalyst longs and commercial compounders; 2.5% crowded; 2.0% distressed
- **Stale catalyst detection**: SEC 8-K filings scanned for readout keywords ("topline", "results", "failed", "discontinued") within 180 days; matched synthetic catalysts deprioritized

---

## Installation

```bash
git clone https://github.com/gk47bball/tatetuck-bot.git
cd tatetuck-bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

---

## Configuration

All settings are controlled via environment variables. Copy `.env.example` to `.env` and populate:

```env
# Market data
EODHD_API_KEY=your_eodhd_key
SEC_USER_AGENT=YourName/1.0 your@email.com

# Discord
DISCORD_TOKEN=your_bot_token
DISCORD_CHANNEL_ID=your_research_channel_id
DISCORD_TRADE_LOG_CHANNEL_ID=your_trade_alert_channel_id  # optional

# Alpaca paper trading
APCA_API_KEY_ID=your_alpaca_paper_key
APCA_API_SECRET_KEY=your_alpaca_paper_secret
APCA_API_BASE_URL=https://paper-api.alpaca.markets
APCA_PAPER_ACCOUNT_ID=your_account_id  # optional

# Execution tuning (optional — defaults shown)
TATETUCK_EXECUTION_ADV_PCT_CAP=0.05        # max 5% of daily volume per order
TATETUCK_EXECUTION_MAX_WEIGHT_PCT=4.0      # max position size %
TATETUCK_SIMULATED_PAPER_EQUITY=100000     # fallback if no Alpaca credentials
```

---

## Running the Platform

### First-time setup — initialize the research store

```bash
python operate_vnext.py
```

This archives the universe, builds training data, labels outcomes, and runs the walk-forward evaluator. Must be run before the Discord bot or trading pipeline will have live data.

### Daily workflow

```bash
# Refresh data and retrain (run daily or on a cron)
python operate_vnext.py

# Generate and submit paper trades to Alpaca
python trade_vnext.py --submit

# Start the Discord bot (keep this running)
python discord_bot.py
```

To keep the bot running in the background:

```bash
nohup python discord_bot.py > bot.log 2>&1 &
tail -f bot.log  # monitor
```

### Discord bot commands

| Command | Description |
|---|---|
| `!analyze TICKER` | Full tear sheet for a biotech company |
| `!top5` | Current top-ranked deployable PM ideas |
| `!status` | Research engine health and readiness audit |
| `!guide` | Plain-English explanation of what the bot does |
| `!setup` | Posts and pins the guide, displays channel ID |
| `!channelid` | Shows the current channel ID |

The bot posts a **morning briefing automatically at 9:30am EST** (13:30 UTC) if `DISCORD_CHANNEL_ID` is set.

### Validation and testing

```bash
# Validate Discord and Alpaca connectivity without going live
python discord_bot.py --self-check

# Walk-forward model evaluation smoke test
python evaluate_vnext.py

# Historical rebuild from SEC filing anchors
python bootstrap_history_vnext.py --max-anchors-per-ticker 4

# Sync active/delisted universe from EODHD
python sync_universe_vnext.py
```

---

## Research Store

Data is persisted locally in `.tatetuck_store/` (Parquet + optional DuckDB):

```
.tatetuck_store/
├── raw/          # Source payloads and compatibility captures
├── tables/       # Normalized Parquet: snapshots, programs, trials, catalysts,
│                 # financing events, feature vectors, predictions, event tape,
│                 # historical universe membership
├── models/       # Serialized model artifacts
└── experiments/  # Experiment registry entries
```

---

## Health Metrics

The `!status` command and `evaluate_vnext.py` report:

| Metric | Institutional Threshold |
|---|---|
| Walk-forward windows | ≥ 3 |
| Leakage audit | pass |
| PM context coverage | ≥ 95% |
| Exact event coverage | ≥ 40% |
| Synthetic event share | ≤ 60% |
| Stale catalyst rate | ≤ 10% |
| Snapshot freshness | ≤ 36 hours |
| From-failure-universe rate | Logged per run |

---

## Key Dependencies

| Package | Purpose |
|---|---|
| `yfinance` | Live price data, options chains |
| `discord.py >= 2.3.2` | Discord bot framework |
| `alpaca-trade-api` | Paper trading execution |
| `scikit-learn` | Ridge regression, isotonic calibration |
| `pandas`, `numpy` | Data processing |
| `pyarrow` | Parquet storage |
| `duckdb` | Optional SQL query surface over store |
| `requests` | SEC EDGAR, ClinicalTrials.gov API |

---

## Notes

- All trading is **paper-only** by design. The `AlpacaPaperBroker` enforces paper endpoints at the code level and cannot be redirected to a live account without modifying the source.
- The platform is designed for retail capital but built to institutional standards — empirical LoA priors, benchmark-relative labels, walk-forward validation, and ADV-constrained execution.
- LLM synthesis is optional. When present it is used as structured research support, not an opaque final scoring engine.
- The KOL proxy (AMC site scoring) requires `locations` data from ClinicalTrials.gov's `contactsLocationsModule`. The infrastructure is wired; extend `prepare.py`'s `fetch_clinical_trials()` to extract that field for full signal coverage.
