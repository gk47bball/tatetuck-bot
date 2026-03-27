# autoresearch (biopharma edition)

This is an experiment to have the LLM do its own research — on biopharma valuation.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar27`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context (if it exists).
   - `prepare.py` — fixed data-fetching and evaluation harness. **Do not modify.**
   - `strategy.py` — the file you modify. Scoring weights, POS logic, TAM assumptions, signal generation.
4. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
5. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs the scoring strategy against **20 biopharma tickers**, split into:
- **14 TRAIN tickers**: Your `valuation_error` is computed against these. This is what you optimize.
- **6 HOLDOUT tickers**: Evaluated but NOT used for your primary score. If the holdout error diverges from train by >0.15, you get an **OVERFIT WARNING**.

Data is fetched from:
- **ClinicalTrials.gov** — trial phases, enrollment, and status (only active/recruiting trials are counted).
- **openFDA** — adverse event reports with seriousness scoring.
- **Yahoo Finance** — market cap, cash, debt, 6-month return, 3-month momentum, volatility.
- **NIH PubMed** — peer-reviewed publication count and abstracts.

API responses are cached to `.data_cache/` for 1 hour, so repeated runs are fast (~5 seconds with cached data).

### The Metric

The composite metric `valuation_error` combines:
- **50% Spearman Rank Correlation**: Did you rank companies in the right order? (best → worst)
- **30% Directional Accuracy**: Did you predict the right direction? (up vs down)
- **20% Mean Absolute Error**: How close was your predicted magnitude?

**Note**: predicting `0.0` for everything gets **zero credit** for directional accuracy. You must have conviction to score well.

You launch it as: `python evaluate.py > run.log 2>&1`

### Rules

**What you CAN do:**
- Modify `strategy.py` — this is the only file you edit. Everything is fair game: phase weights, disease multipliers, discount rates, TAM estimation (disease-specific TAMs are already seeded in `DISEASE_TAMS`), cash adjustments, pipeline breadth scoring, enrollment weighting, momentum weighting, literature scoring, FDA safety penalties, or any entirely new scoring logic you invent.

**What you CANNOT do:**
- Modify `prepare.py`, `evaluate.py`, or `dashboard.html`. They are read-only.
- Install new packages or add dependencies.

**The goal is simple: get the lowest `valuation_error` on the TRAIN set, without overfitting the HOLDOUT set.**

**The first run**: Always establish the baseline first.

## Output format

```
--- TRAIN (13 scored) ---
valuation_error:  0.543210
rank_correlation: 0.4500
dir_accuracy:     0.6000
mae:              0.3500

--- HOLDOUT ---
holdout_error:    0.520000
holdout_rank:     0.3800
holdout_dir:      0.5000

--- SUMMARY ---
num_total:        20
elapsed_seconds:  5.2
---
```

You extract the key metric from the log:

```
grep "^valuation_error:" run.log
```

## Logging results

Log to `results.tsv` (tab-separated). 4 columns:

```
commit	valuation_error	status	description
```

## The experiment loop

LOOP FOREVER:

1. Look at the git state: current branch/commit
2. Tune `strategy.py` with an experimental idea
3. `git commit`
4. Run: `python evaluate.py > run.log 2>&1`
5. Read: `grep "^valuation_error:" run.log`
6. If empty → crashed. `tail -n 50 run.log` to debug.
7. Log to `results.tsv` (do NOT git-track this file)
8. If improved → keep the commit
9. If worse → `git reset --hard HEAD~1`

### Data fields available to use

From `data["finance"]`:
- `marketCap`, `cash`, `debt`, `totalRevenue`, `enterpriseValue`
- `trailing_6mo_return`, `momentum_3mo`, `volatility`, `price_now`

From trial data:
- `num_trials` (active only), `num_total_trials`, `num_inactive_trials`
- `best_phase`, `total_enrollment`, `max_single_enrollment`
- `phase_enrollment` (dict: phase → total enrollment)
- `phase_trial_counts` (dict: phase → count)
- `conditions` (list of disease areas)
- `drug_names` (list)

From FDA/literature:
- `fda_adverse_events`, `fda_serious_events`, `fda_serious_ratio`
- `num_papers`, `pubmed_papers` (list of dicts with title/abstract)

### Ideas to try
- Adjust phase weights, disease multipliers, TAMs
- Use enrollment as a conviction signal (bigger trial → more confident management)
- Use momentum as a trend signal (positive momentum → likely continues)
- Penalize high FDA serious event ratios
- Weight by volatility (high vol = higher risk premium)
- Use revenue data to distinguish pre-revenue vs commercial companies
- Create non-linear combinations of features
- Invent entirely new scoring logic from the available data

**Timeout**: ~5 seconds with cached data, ~90 seconds on first run.

**NEVER STOP**: The loop runs until the human interrupts you, period.
