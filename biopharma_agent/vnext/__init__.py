"""vNext event-driven biopharma alpha platform."""

from .archive import ArchiveSummary, archive_universe
from .audit import ResearchAudit, ResearchAuditBuilder
from .eodhd import EODHDEventTapeClient, EODHDUniverseClient, UniverseSyncSummary
from .execution import (
    AlpacaPaperBroker,
    ExecutionFeedbackSummary,
    ExecutionInstruction,
    ExecutionPlan,
    OrderSubmission,
    PMExecutionPlanner,
    materialize_execution_feedback,
)
from .facade import TatetuckPlatform
from .history import HistoricalSnapshotBootstrapper, HistoryBootstrapSummary
from .labels import LabelSummary, PointInTimeLabeler
from .ops import PipelineRunRecord, ReadinessReport, build_readiness_report, record_pipeline_run
from .pipeline import PipelineExecutionSummary, run_vnext_pipeline
from .replay import HistoricalReplayEngine, ReplaySummary, snapshot_from_dict
from .settings import VNextSettings
from .universe import UniverseResolver

__all__ = [
    "AlpacaPaperBroker",
    "ArchiveSummary",
    "ExecutionFeedbackSummary",
    "ResearchAudit",
    "ResearchAuditBuilder",
    "ExecutionInstruction",
    "ExecutionPlan",
    "EODHDEventTapeClient",
    "EODHDUniverseClient",
    "HistoricalReplayEngine",
    "HistoricalSnapshotBootstrapper",
    "HistoryBootstrapSummary",
    "LabelSummary",
    "OrderSubmission",
    "PMExecutionPlanner",
    "PipelineExecutionSummary",
    "PipelineRunRecord",
    "PointInTimeLabeler",
    "ReadinessReport",
    "ReplaySummary",
    "TatetuckPlatform",
    "UniverseSyncSummary",
    "UniverseResolver",
    "VNextSettings",
    "archive_universe",
    "build_readiness_report",
    "materialize_execution_feedback",
    "record_pipeline_run",
    "run_vnext_pipeline",
    "snapshot_from_dict",
]
