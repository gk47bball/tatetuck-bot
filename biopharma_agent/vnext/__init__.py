"""vNext event-driven biopharma alpha platform."""

from .archive import ArchiveSummary, archive_universe
from .autonomy import (
    BrokerReconciliationSummary,
    record_portfolio_nav,
    record_trade_decision_run,
    reconcile_broker_state,
    write_autonomy_health_snapshot,
)
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
from .monitor import AutonomyMonitor, MonitorRunResult, MonitorTrigger
from .ops import PipelineRunRecord, ReadinessReport, build_readiness_report, record_pipeline_run
from .pipeline import PipelineExecutionSummary, run_vnext_pipeline
from .replay import HistoricalReplayEngine, ReplaySummary, snapshot_from_dict
from .settings import VNextSettings
from .trigger_ingestion import RealTimeTriggerIngestor, TriggerIngestionSummary
from .universe import UniverseResolver

__all__ = [
    "AlpacaPaperBroker",
    "ArchiveSummary",
    "BrokerReconciliationSummary",
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
    "AutonomyMonitor",
    "MonitorRunResult",
    "MonitorTrigger",
    "RealTimeTriggerIngestor",
    "OrderSubmission",
    "PMExecutionPlanner",
    "PipelineExecutionSummary",
    "PipelineRunRecord",
    "PointInTimeLabeler",
    "ReadinessReport",
    "ReplaySummary",
    "TriggerIngestionSummary",
    "TatetuckPlatform",
    "UniverseSyncSummary",
    "UniverseResolver",
    "VNextSettings",
    "archive_universe",
    "build_readiness_report",
    "materialize_execution_feedback",
    "record_portfolio_nav",
    "record_pipeline_run",
    "record_trade_decision_run",
    "reconcile_broker_state",
    "run_vnext_pipeline",
    "snapshot_from_dict",
    "write_autonomy_health_snapshot",
]
