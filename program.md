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

Each experiment runs the scoring strategy against **20 benchmark biopharma tickers**. It fetches real clinical trial data from ClinicalTrials.gov, FDA safety data from openFDA, financial data from Yahoo Finance, and scientific literature from PubMed. API responses are cached to `.data_cache/` for 1 hour, so repeated runs are fast (~5 seconds with cached data).

The evaluation computes a **composite metric** called `valuation_error`:
- **50% Spearman Rank Correlation**: Did you rank the companies in the right order? (best → worst)
- **30% Directional Accuracy**: Did you predict the right direction? (up vs down)
- **20% Mean Absolute Error**: How close was your predicted magnitude?

Note: predicting `0.0` for everything gets **zero credit** for directional accuracy. You must have conviction to score well.

You launch it simply as: `python evaluate.py > run.log 2>&1`

**What you CAN do:**
- Modify `strategy.py` — this is the only file you edit. Everything is fair game: phase weights, disease multipliers, discount rates, TAM estimation, cash adjustments, pipeline breadth scoring, literature scoring, or any entirely new scoring logic you invent.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed data-fetching, benchmark tickers, and evaluation function.
- Modify `evaluate.py`. It is read-only.
- Install new packages or add dependencies.
- Modify the evaluation harness. The `evaluate_strategy` function in `prepare.py` is the ground truth.

**The goal is simple: get the lowest `valuation_error`.** This is a composite metric combining Spearman rank correlation, directional accuracy, and MAE. Lower is better.

**The first run**: Your very first run should always be to establish the baseline, so you will run the evaluation as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
valuation_error:  0.543210
rank_correlation: 0.4500
dir_accuracy:     0.6000
mae:              0.3500
num_scored:       20
num_total:        20
elapsed_seconds:  45.2
---
```

You extract the key metric from the log:

```
grep "^valuation_error:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 4 columns:

```
commit	valuation_error	status	description
```

1. git commit hash (short, 7 chars)
2. valuation_error achieved (e.g. 0.543210) — use 0.000000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

Example:

```
commit	valuation_error	status	description
a1b2c3d	0.543210	keep	baseline
b2c3d4e	0.498700	keep	increased PHASE3 weight to 0.65
c3d4e5f	0.556000	discard	switched to flat TAM of $5B
d4e5f6g	0.000000	crash	division by zero in new cash model
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar27`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `strategy.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python evaluate.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^valuation_error:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If valuation_error improved (lower), you "advance" the branch, keeping the git commit
9. If valuation_error is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous biopharma researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Ideas to try:**
- Adjust phase transition weights (maybe Phase 3 drugs deserve even more weight?)
- Change disease area multipliers (maybe oncology is too harsh? maybe gene therapy deserves a premium?)
- Modify the TAM estimation (use different defaults per disease area — already seeded in DISEASE_TAMS)
- Change the discount rate
- Weight the cash/debt position differently
- Add momentum signals from the financial data
- Use the number and quality of PubMed publications as a signal
- Invent entirely new scoring features from the available data
- Combine signals in non-linear ways
- Use `fda_serious_events` as a negative signal

**Timeout**: Each experiment should take ~5 seconds with cached data, ~90 seconds on first run. If a run exceeds 3 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes (a bug, division by zero, etc.), use your judgment: If it's something dumb and easy to fix, fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash", and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read `prepare.py` for new data fields you haven't used, try combining previous near-misses, try more radical approaches. The loop runs until the human interrupts you, period.
