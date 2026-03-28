# Tatetuck Bot

**Tatetuck Bot** is now a local-first, event-driven biopharma alpha platform.
It keeps the legacy CLI and Discord surfaces, but the research core has been
rebuilt around a typed company-program-catalyst graph, persisted research
artifacts, feature generation, model scoring, and a walk-forward evaluator.
It now also carries a survivorship-aware historical universe and an exact event
tape built from SEC filings plus EODHD calendar/news data.

## What Changed

- `prepare.py` and `evaluate.py` remain available as legacy compatibility harnesses.
- The new primary platform lives under `biopharma_agent/vnext/`.
- Current company analysis flows through a `TatetuckPlatform` facade that:
  - ingests and stores raw payloads and normalized snapshots
  - builds `CompanySnapshot`, `Program`, `CatalystEvent`, and `FinancingEvent`
  - generates event-driven feature vectors
  - scores them with a rules + ensemble prediction layer
  - emits signal artifacts and portfolio recommendations
- Historical validation now includes:
  - archived snapshots rebuilt from filing anchors and price history
  - exact SEC filing timestamps where available
  - exact EODHD earnings/news event timestamps where available
  - active and delisted universe membership from EODHD exchange symbol lists
- The local store uses **Parquet** for tables and now exposes an optional
  **DuckDB** query surface.

## Main Entry Points

- `main.py`
  Generates a company research report through the vNext compatibility facade.
- `discord_bot.py`
  Serves event-driven tear sheets and benchmark scans from the vNext platform.
- `BOT_GUIDE.md`
  Plain-English explanation of what the Discord bot does and how to read it.
- `evaluate.py`
  Legacy scorer for the original heuristic benchmark.
- `evaluate_vnext.py`
  New platform smoke path for ingestion, ranking, and walk-forward evaluation.
- `bootstrap_history_vnext.py`
  Rebuilds dated historical snapshots, labels, and evaluation artifacts.
- `sync_universe_vnext.py`
  Syncs active and delisted exchange membership from EODHD for survivorship-aware validation.

## Storage

The new local research store is written to `.tatetuck_store/` and contains:

- `raw/`
  Source payloads and compatibility captures
- `tables/`
  Normalized Parquet tables for snapshots, programs, trials, catalysts,
  financing events, feature vectors, predictions, exact event tape, and
  historical universe membership
- `models/`
  Serialized model artifacts
- `experiments/`
  Experiment registry entries

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Legacy-style report:

```bash
python main.py --ticker CRSP
```

Legacy benchmark:

```bash
python evaluate.py > run.log 2>&1
```

vNext platform run:

```bash
python evaluate_vnext.py > vnext_run.log 2>&1
```

Historical rebuild:

```bash
python bootstrap_history_vnext.py --max-anchors-per-ticker 4
```

Universe sync:

```bash
python sync_universe_vnext.py
```

Discord bot self-check:

```bash
python discord_bot.py --self-check
```

Discord bot run:

```bash
python discord_bot.py
```

Discord setup requires:

- `DISCORD_TOKEN`
- `DISCORD_CHANNEL_ID`
- `DISCORD_TRADE_LOG_CHANNEL_ID` (optional, for paper-trade alerts; falls back to `DISCORD_CHANNEL_ID`)

Once online, the bot supports:

- `!analyze TICKER`
- `!top5`
- `!guide`
- `!setup`
- `!channelid`
- `!status`

Paper trading:

```bash
python trade_vnext.py --submit
```

When paper orders are submitted to Alpaca, Tatetuck now posts a Discord trade
alert to `DISCORD_TRADE_LOG_CHANNEL_ID` when available, and falls back to
`DISCORD_CHANNEL_ID` if the trade-log channel is permission-blocked.

The execution engine is state-aware. It applies different entry bars and size
caps for:

- hard catalyst setups
- soft catalyst setups
- commercial launch asymmetries
- franchise / capital-allocation setups
- sentiment-floor setups

Execution asymmetry settings such as
`TATETUCK_EXECUTION_MIN_INTERNAL_UPSIDE_PCT=0.08` and
`TATETUCK_EXECUTION_MIN_FLOOR_SUPPORT_PCT=0.10` are expressed as decimal
returns rather than whole percentages.

## Notes

- The strict evaluator now reports both headline and institutionally stricter
  metrics, including PM-context coverage, exact-event coverage, synthetic-event
  share, and latest-window audit rows.
- The current platform state has passed the internal readiness audit with
  exact-event coverage above the institutional threshold, active/delisted
  universe membership synced from EODHD, and leakage checks passing.
- The walk-forward stack is now survivorship-aware, but exact event coverage is
  still only as good as the archived source tape. SEC filings and EODHD
  earnings/news are now first-class; broader event expansion can still improve
  coverage over time.
- LLM synthesis is still optional. When present, it is treated as structured
  research support rather than an opaque final scoring engine.
