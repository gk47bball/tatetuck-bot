from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .evaluation import WalkForwardEvaluator, WalkForwardSummary
from .settings import VNextSettings
from .storage import LocalResearchStore


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class PipelineRunRecord:
    job_name: str
    status: str
    started_at: str
    finished_at: str
    duration_seconds: float
    metrics: dict[str, Any]
    config: dict[str, Any]
    notes: str = ""

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReadinessReport:
    status: str
    generated_at: str
    store_dir: str
    eodhd_configured: bool
    sec_user_agent_configured: bool
    snapshot_rows: int
    distinct_snapshot_dates: int
    latest_snapshot_age_hours: float | None
    label_rows: int
    event_label_rows: int
    matured_return_90d_rows: int
    matured_event_rows: int
    archive_run_count: int
    successful_archive_runs: int
    backfill_run_count: int
    successful_backfill_runs: int
    evaluate_run_count: int
    successful_evaluate_runs: int
    eodhd_cache_files: int
    walkforward_rows: int
    walkforward_windows: int
    leakage_passed: bool
    blockers: list[str]
    warnings: list[str]
    evaluation_message: str


def record_pipeline_run(
    store: LocalResearchStore,
    job_name: str,
    status: str,
    started_at: str,
    finished_at: str,
    metrics: dict[str, Any],
    config: dict[str, Any],
    notes: str = "",
) -> PipelineRunRecord:
    started_ts = pd.Timestamp(started_at)
    finished_ts = pd.Timestamp(finished_at)
    duration_seconds = float(max((finished_ts - started_ts).total_seconds(), 0.0))
    record = PipelineRunRecord(
        job_name=job_name,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration_seconds,
        metrics=metrics,
        config=config,
        notes=notes,
    )
    store.write_pipeline_run(record.to_row())
    return record


def build_readiness_report(
    store: LocalResearchStore | None = None,
    settings: VNextSettings | None = None,
    refresh_labels: bool = False,
) -> ReadinessReport:
    settings = settings or VNextSettings.from_env()
    store = store or LocalResearchStore(settings.store_dir)
    snapshots = store.read_table("company_snapshots")
    labels = store.read_table("labels")
    event_labels = store.read_table("event_labels")
    pipeline_runs = store.read_table("pipeline_runs")
    evaluator = WalkForwardEvaluator(store=store, settings=settings)
    evaluation = evaluator.evaluate() if (not snapshots.empty and not labels.empty) or refresh_labels else WalkForwardSummary(
        num_rows=0,
        num_windows=0,
        rank_ic=0.0,
        hit_rate=0.0,
        top_bottom_spread=0.0,
        turnover=0.0,
        max_drawdown=0.0,
        beta_adjusted_return=0.0,
        calibrated_brier=1.0,
        leakage_passed=False,
        message="Need snapshots and labels before evaluation can run.",
    )

    snapshot_rows = len(snapshots)
    distinct_snapshot_dates = 0
    latest_snapshot_age_hours: float | None = None
    if not snapshots.empty and "as_of" in snapshots.columns:
        snapshot_ts = pd.to_datetime(
            snapshots["as_of"],
            errors="coerce",
            utc=True,
            format="mixed",
        ).dt.tz_convert(None)
        snapshot_ts = snapshot_ts.dropna()
        if not snapshot_ts.empty:
            distinct_snapshot_dates = int(snapshot_ts.dt.normalize().nunique())
            latest_snapshot_age_hours = float(
                (pd.Timestamp.now(tz="UTC").tz_convert(None) - snapshot_ts.max()).total_seconds() / 3600.0
            )

    matured_return_90d_rows = int(labels["target_return_90d"].notna().sum()) if not labels.empty and "target_return_90d" in labels.columns else 0
    matured_event_rows = int(event_labels["target_event_return_10d"].notna().sum()) if not event_labels.empty and "target_event_return_10d" in event_labels.columns else 0
    archive_run_count, successful_archive_runs = _run_counts(pipeline_runs, "archive_vnext")
    backfill_run_count, successful_backfill_runs = _run_counts(pipeline_runs, "backfill_vnext")
    evaluate_run_count, successful_evaluate_runs = _run_counts(pipeline_runs, "evaluate_vnext")
    eodhd_cache_files = len(store.list_raw_payload_paths("market_prices_eodhd"))

    blockers: list[str] = []
    warnings: list[str] = []

    if not settings.eodhd_api_key and eodhd_cache_files == 0:
        blockers.append("EODHD market data is not configured and no EODHD price cache exists.")
    if distinct_snapshot_dates < settings.min_snapshot_dates:
        blockers.append(
            f"Only {distinct_snapshot_dates} distinct snapshot dates are available; need at least {settings.min_snapshot_dates}."
        )
    if matured_return_90d_rows < settings.min_matured_return_rows:
        blockers.append(
            f"Only {matured_return_90d_rows} matured 90-day labels are available; need at least {settings.min_matured_return_rows}."
        )
    if successful_archive_runs < settings.min_archive_runs:
        blockers.append(
            f"Only {successful_archive_runs} successful archive runs recorded; need at least {settings.min_archive_runs}."
        )
    if evaluation.num_windows < settings.min_walkforward_windows:
        blockers.append(
            f"Only {evaluation.num_windows} walk-forward windows available; need at least {settings.min_walkforward_windows}."
        )
    if not evaluation.leakage_passed:
        blockers.append("Leakage audit is not passing.")

    if latest_snapshot_age_hours is not None and latest_snapshot_age_hours > settings.max_snapshot_age_hours:
        warnings.append(
            f"Latest snapshot is {latest_snapshot_age_hours:.1f} hours old, exceeding the freshness target of {settings.max_snapshot_age_hours} hours."
        )
    if successful_backfill_runs == 0:
        warnings.append("No successful backfill runs recorded yet.")
    if successful_evaluate_runs == 0:
        warnings.append("No successful evaluate runs recorded yet.")
    if matured_event_rows == 0:
        warnings.append("No matured event-window labels are available yet.")

    status = "production_ready" if not blockers else "needs_attention"
    return ReadinessReport(
        status=status,
        generated_at=utc_now_iso(),
        store_dir=str(store.base_dir),
        eodhd_configured=bool(settings.eodhd_api_key),
        sec_user_agent_configured=bool(settings.sec_user_agent and "support@" not in settings.sec_user_agent),
        snapshot_rows=snapshot_rows,
        distinct_snapshot_dates=distinct_snapshot_dates,
        latest_snapshot_age_hours=latest_snapshot_age_hours,
        label_rows=len(labels),
        event_label_rows=len(event_labels),
        matured_return_90d_rows=matured_return_90d_rows,
        matured_event_rows=matured_event_rows,
        archive_run_count=archive_run_count,
        successful_archive_runs=successful_archive_runs,
        backfill_run_count=backfill_run_count,
        successful_backfill_runs=successful_backfill_runs,
        evaluate_run_count=evaluate_run_count,
        successful_evaluate_runs=successful_evaluate_runs,
        eodhd_cache_files=eodhd_cache_files,
        walkforward_rows=evaluation.num_rows,
        walkforward_windows=evaluation.num_windows,
        leakage_passed=evaluation.leakage_passed,
        blockers=blockers,
        warnings=warnings,
        evaluation_message=evaluation.message,
    )


def _run_counts(pipeline_runs: pd.DataFrame, job_name: str) -> tuple[int, int]:
    if pipeline_runs.empty or "job_name" not in pipeline_runs.columns:
        return 0, 0
    subset = pipeline_runs[pipeline_runs["job_name"] == job_name]
    if subset.empty:
        return 0, 0
    successes = int((subset["status"] == "success").sum()) if "status" in subset.columns else 0
    return len(subset), successes
