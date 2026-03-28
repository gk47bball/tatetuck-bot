from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from .archive import ArchiveSummary, archive_universe
from .evaluation import WalkForwardEvaluator, WalkForwardSummary
from .facade import TatetuckPlatform
from .labels import LabelSummary, PointInTimeLabeler
from .ops import ReadinessReport, build_readiness_report, record_pipeline_run, utc_now_iso
from .replay import HistoricalReplayEngine, ReplaySummary
from .settings import VNextSettings
from .storage import LocalResearchStore


@dataclass(slots=True)
class PipelineExecutionSummary:
    archive: ArchiveSummary
    replay: ReplaySummary
    labels: LabelSummary
    evaluation: WalkForwardSummary
    readiness: ReadinessReport
    store_dir: str


def run_vnext_pipeline(
    universe: Iterable[tuple[str, str]],
    settings: VNextSettings | None = None,
    store: LocalResearchStore | None = None,
    include_literature: bool | None = None,
    as_of: datetime | None = None,
    replay_limit: int = 0,
    run_evaluation: bool = True,
) -> PipelineExecutionSummary:
    settings = settings or VNextSettings.from_env()
    store = store or LocalResearchStore(settings.store_dir)
    include_literature = settings.include_literature if include_literature is None else include_literature
    started_at = utc_now_iso()

    try:
        platform = TatetuckPlatform(store=store)
        _, archive_summary = archive_universe(
            platform,
            universe=universe,
            include_literature=include_literature,
            as_of=as_of,
        )
        replay_summary = HistoricalReplayEngine(store=store).rebuild_from_archived_snapshots(limit=replay_limit)
        label_summary = PointInTimeLabeler(store=store).materialize_labels()
        if run_evaluation:
            evaluation = WalkForwardEvaluator(store=store).evaluate()
        else:
            evaluation = WalkForwardSummary(
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
                message="evaluation skipped",
            )
        readiness = build_readiness_report(store=store, settings=settings)
        finished_at = utc_now_iso()
        record_pipeline_run(
            store=store,
            job_name="operate_vnext",
            status="success",
            started_at=started_at,
            finished_at=finished_at,
            metrics={
                "archived_companies": archive_summary.archived_companies,
                "replayed_snapshots": replay_summary.replayed_snapshots,
                "snapshot_label_rows": label_summary.snapshot_label_rows,
                "matured_return_90d_rows": label_summary.matured_return_90d_rows,
                "walkforward_windows": evaluation.num_windows,
                "readiness_blockers": len(readiness.blockers),
            },
            config={
                "include_literature": include_literature,
                "replay_limit": replay_limit,
                "run_evaluation": run_evaluation,
                "store_dir": settings.store_dir,
            },
            notes=readiness.status,
        )
        return PipelineExecutionSummary(
            archive=archive_summary,
            replay=replay_summary,
            labels=label_summary,
            evaluation=evaluation,
            readiness=readiness,
            store_dir=str(store.base_dir),
        )
    except Exception as exc:
        finished_at = utc_now_iso()
        record_pipeline_run(
            store=store,
            job_name="operate_vnext",
            status="failed",
            started_at=started_at,
            finished_at=finished_at,
            metrics={},
            config={
                "include_literature": include_literature,
                "replay_limit": replay_limit,
                "run_evaluation": run_evaluation,
                "store_dir": settings.store_dir,
            },
            notes=f"{type(exc).__name__}: {exc}",
        )
        raise
