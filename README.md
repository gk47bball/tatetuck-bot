# Tatetuck Bot

**Tatetuck Bot** is now a local-first, event-driven biopharma alpha platform.
It keeps the legacy CLI and Discord surfaces, but the research core has been
rebuilt around a typed company-program-catalyst graph, persisted research
artifacts, feature generation, model scoring, and a walk-forward evaluator.

## What Changed

- `prepare.py` and `evaluate.py` remain available as legacy compatibility harnesses.
- The new primary platform lives under `biopharma_agent/vnext/`.
- Current company analysis flows through a `TatetuckPlatform` facade that:
  - ingests and stores raw payloads and normalized snapshots
  - builds `CompanySnapshot`, `Program`, `CatalystEvent`, and `FinancingEvent`
  - generates event-driven feature vectors
  - scores them with a rules + ensemble prediction layer
  - emits signal artifacts and portfolio recommendations
- The local store uses **Parquet** for tables and now exposes an optional
  **DuckDB** query surface.

## Main Entry Points

- `main.py`
  Generates a company research report through the vNext compatibility facade.
- `discord_bot.py`
  Serves event-driven tear sheets and benchmark scans from the vNext platform.
- `evaluate.py`
  Legacy scorer for the original heuristic benchmark.
- `evaluate_vnext.py`
  New platform smoke path for ingestion, ranking, and walk-forward evaluation.

## Storage

The new local research store is written to `.tatetuck_store/` and contains:

- `raw/`
  Source payloads and compatibility captures
- `tables/`
  Normalized Parquet tables for snapshots, programs, trials, catalysts,
  financing events, feature vectors, and predictions
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

## Notes

- The new walk-forward evaluator expects **archived snapshots over time**. On a
  fresh store it will ingest data and rank ideas successfully, but historical
  backtest metrics will remain empty until enough dated snapshots accumulate.
- LLM synthesis is still optional. When present, it is treated as structured
  research support rather than an opaque final scoring engine.
