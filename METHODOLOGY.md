# Tatetuck vNext Methodology

## 1. Core Worldview

The platform no longer treats a biotech as a single scalar score. It models a
company as a graph of:

- `CompanySnapshot`
- `Program`
- `Trial`
- `ApprovedProduct`
- `CatalystEvent`
- `FinancingEvent`

This lets Tatetuck reason about late-stage programs, commercial assets,
financing pressure, and the timing and importance of catalysts inside the same
research artifact.

## 2. Data Platform

The vNext platform is local-first:

- raw payloads are persisted to `.tatetuck_store/raw/`
- normalized tables are written as Parquet in `.tatetuck_store/tables/`
- model artifacts and experiment metadata are stored alongside them
- DuckDB can query the normalized Parquet tables directly

`prepare.py` is still used as the compatibility ingestion layer, but scoring and
evaluation are intended to operate from the local store rather than from live
point-in-time API calls.

## 3. Feature Families

The event-driven feature engine is split into five families:

1. `program_quality`
   POS priors, phase score, enrollment scale, endpoint strength, modality risk,
   and TAM-to-market-cap context.
2. `catalyst_timing`
   Nearest milestone horizon, event probability, event importance, and
   crowdedness.
3. `commercial_execution`
   Revenue scale, revenue-to-market-cap, approved product presence, and growth
   signal.
4. `balance_sheet`
   Cash-to-cap, debt-to-cap, runway months, and financing pressure.
5. `market_flow`
   3-month momentum and volatility only. The vNext feature engine does not use
   the legacy trailing-6-month target as an input feature.

## 4. Model Layer

The prediction stack is a hybrid:

- a transparent rule model provides baseline expected return and catalyst
  success estimates
- an optional sklearn ensemble can be trained from archived snapshots when the
  store has enough historical labeled rows
- predictions are surfaced as `ModelPrediction` and aggregated into a
  `SignalArtifact`

Each signal artifact exposes:

- `expected_return`
- `catalyst_success_prob`
- `confidence`
- `crowding_risk`
- `financing_risk`
- `thesis_horizon`

## 5. Portfolio Construction

Portfolio recommendations are thesis-aware rather than simple score sorts. The
constructor blends expected return, confidence, crowding, and financing risk to
emit:

- stance
- target weight
- max weight
- scenario label
- risk flags

Scenario labels currently include:

- `pre-catalyst long`
- `watchlist only`
- `commercial compounder`
- `avoid due to financing`
- `pairs candidate`

## 6. Evaluation

The new primary evaluator is `evaluate_vnext.py`, backed by the
`WalkForwardEvaluator`.

It is designed for:

- `30d`, `90d`, and `180d` forward returns
- binary catalyst-success labels
- rolling time-based splits only
- leakage audits
- cross-sectional and portfolio metrics

Because the store is snapshot-based, meaningful walk-forward results require
multiple archived dates over time. On a fresh store, Tatetuck can ingest and
rank names immediately, but the backtest metrics remain intentionally empty
until enough dated observations accumulate.
