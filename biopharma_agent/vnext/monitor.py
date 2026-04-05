from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import time
from typing import Any, Iterable, Protocol

import pandas as pd

from .autonomy import (
    BrokerReconciliationSummary,
    record_portfolio_nav,
    record_trade_decision_run,
    reconcile_broker_state,
    write_autonomy_health_snapshot,
)
from .execution import (
    AlpacaPaperBroker,
    DiscordTradeNotifier,
    ExecutionPlan,
    OrderSubmission,
    PMExecutionPlanner,
    execute_plan,
)
from .facade import TatetuckPlatform
from .ops import build_readiness_report, record_pipeline_run, utc_now_iso
from .settings import VNextSettings
from .storage import LocalResearchStore
from .trigger_ingestion import RealTimeTriggerIngestor, TriggerIngestionSummary


def _ts(value: Any) -> pd.Timestamp | pd.NaT:
    ts = pd.to_datetime(value, errors="coerce", utc=True, format="mixed")
    if pd.isna(ts):
        return pd.NaT
    return ts.tz_convert(None)


def _parse_listish(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return []


class MonitorBroker(Protocol):
    def is_configured(self) -> bool: ...

    def simulated_account(self): ...

    def ensure_expected_account(self, account): ...

    def account(self): ...

    def positions(self): ...

    def recent_orders(self, limit: int = 200): ...

    def ensure_paper_only(self) -> None: ...

    def submit_market_notional_buy(self, symbol: str, notional: float) -> OrderSubmission: ...

    def submit_market_qty_sell(self, symbol: str, qty: float) -> OrderSubmission: ...


@dataclass(slots=True)
class MonitorTrigger:
    symbol: str
    company_name: str
    priority: float
    source: str
    reason: str
    event_type: str | None = None
    event_timestamp: str | None = None

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class MonitorRunResult:
    generated_at: str
    status: str
    trigger_count: int
    refreshed_symbols: list[str]
    actionable_orders: int
    submitted_orders: int
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    note: str = ""


class AutonomyMonitor:
    def __init__(
        self,
        settings: VNextSettings,
        store: LocalResearchStore | None = None,
        platform: TatetuckPlatform | None = None,
        broker: MonitorBroker | None = None,
        planner: PMExecutionPlanner | None = None,
        notifier: DiscordTradeNotifier | None = None,
        trigger_ingestor: RealTimeTriggerIngestor | None = None,
    ):
        self.settings = settings
        self.store = store or LocalResearchStore(settings.store_dir)
        self.platform = platform or TatetuckPlatform(store=self.store)
        self.broker = broker or AlpacaPaperBroker(settings=settings)
        self.planner = planner or PMExecutionPlanner(settings=settings, store=self.store)
        self.notifier = notifier or DiscordTradeNotifier(settings=settings)
        self.trigger_ingestor = trigger_ingestor or RealTimeTriggerIngestor(store=self.store)
        self._last_trigger_ingestion_summary: TriggerIngestionSummary | None = None

    def detect_triggers(
        self,
        *,
        now: datetime | None = None,
        manual_symbols: Iterable[str] | None = None,
        actual_symbols: Iterable[str] | None = None,
        max_symbols: int | None = None,
        ingest_live_events: bool = True,
    ) -> list[MonitorTrigger]:
        now = now or datetime.now(timezone.utc)
        if ingest_live_events:
            self._last_trigger_ingestion_summary = self._refresh_live_trigger_events(
                now=now,
                manual_symbols=manual_symbols,
                actual_symbols=actual_symbols,
            )
        else:
            self._last_trigger_ingestion_summary = None
        by_symbol: dict[str, MonitorTrigger] = {}

        def add_trigger(trigger: MonitorTrigger) -> None:
            existing = by_symbol.get(trigger.symbol)
            if existing is None or trigger.priority > existing.priority:
                by_symbol[trigger.symbol] = trigger

        for symbol in manual_symbols or []:
            ticker = str(symbol).upper().strip()
            if not ticker:
                continue
            add_trigger(
                MonitorTrigger(
                    symbol=ticker,
                    company_name=self._company_name_for(ticker),
                    priority=100.0,
                    source="manual",
                    reason="Manual intraday monitor refresh requested.",
                )
            )

        for trigger in self._recent_event_triggers(now):
            add_trigger(trigger)

        active_symbols = set(str(symbol).upper() for symbol in (actual_symbols or []) if str(symbol).strip())
        active_symbols.update(self._reconciliation_active_symbols())
        active_symbols.update(self._recent_active_symbols(now))
        for trigger in self._stale_snapshot_triggers(now, active_symbols):
            add_trigger(trigger)

        if not by_symbol and active_symbols:
            for symbol in sorted(active_symbols):
                add_trigger(
                    MonitorTrigger(
                        symbol=symbol,
                        company_name=self._company_name_for(symbol),
                        priority=40.0,
                        source="lifecycle",
                        reason="Existing live exposure needs a routine lifecycle review.",
                    )
                )

        ordered = sorted(
            by_symbol.values(),
            key=lambda item: (-item.priority, item.symbol),
        )
        limit = max_symbols if max_symbols is not None and max_symbols > 0 else int(self.settings.monitor_max_symbols_per_cycle)
        if limit > 0:
            ordered = ordered[:limit]
        return ordered

    def run_once(
        self,
        *,
        submit: bool = False,
        allow_blocked_readiness: bool = False,
        include_literature: bool = False,
        prefer_live: bool = True,
        manual_symbols: Iterable[str] | None = None,
        max_symbols: int | None = None,
    ) -> MonitorRunResult:
        started_at = utc_now_iso()
        start = time.time()
        readiness = build_readiness_report(
            store=self.store,
            settings=self.settings,
            prefer_cached_validation=True,
        )
        account = None
        positions = []
        plan = None
        submissions: list[OrderSubmission] = []
        notification = None
        reconciliation = None
        triggers: list[MonitorTrigger] = []
        ingestion_summary: TriggerIngestionSummary | None = None
        try:
            if self.broker.is_configured():
                account = self.broker.account()
                self.broker.ensure_expected_account(account)
                positions = list(self.broker.positions())
            else:
                account = self.broker.simulated_account()
                readiness.warnings.append(
                    "Broker credentials are not configured. Monitor is running in simulated-paper mode."
                )
            record_portfolio_nav(store=self.store, account=account, captured_at=started_at)

            triggers = self.detect_triggers(
                now=datetime.now(timezone.utc),
                manual_symbols=manual_symbols,
                actual_symbols=[position.symbol for position in positions],
                max_symbols=max_symbols,
            )
            ingestion_summary = self._last_trigger_ingestion_summary
            if ingestion_summary is not None:
                plan_warning = (
                    f"Live trigger ingestion polled {ingestion_summary.symbols_polled} symbols and found "
                    f"{ingestion_summary.symbols_with_new_events} with new events."
                )
                readiness.warnings.append(plan_warning)
                readiness.warnings.extend(ingestion_summary.warnings)
            analyses = []
            for trigger in triggers:
                analyses.append(
                    self.platform.analyze_ticker(
                        trigger.symbol,
                        company_name=trigger.company_name,
                        include_literature=include_literature or self.settings.include_literature,
                        prefer_archive=not prefer_live,
                        fallback_to_archive=True,
                        persist=True,
                    )
                )

            if analyses:
                plan = self.planner.build_plan(
                    analyses=analyses,
                    account=account,
                    positions=positions,
                    readiness=readiness,
                )
            else:
                plan = ExecutionPlan(
                    generated_at=utc_now_iso(),
                    account_id=account.account_id,
                    equity=float(account.equity),
                    buying_power=float(account.buying_power),
                    deployable_notional=0.0,
                    selected_symbols=[],
                    instructions=[],
                    blockers=[],
                    warnings=["No symbols were trigger-worthy this cycle."],
                    readiness_status=readiness.status,
                    gross_cap_pct=float(self.settings.max_gross_exposure_pct),
                    gross_cap_multiplier=1.0,
                    exposure_governor={},
                )

            if not self.broker.is_configured() and submit:
                plan.blockers.append("Cannot submit orders without broker credentials.")
            submit_attempted = bool(submit)
            if plan.blockers and not allow_blocked_readiness:
                submit_attempted = False

            submissions = execute_plan(
                plan=plan,
                broker=self.broker,
                store=self.store,
                submit=submit_attempted,
            )
            if submit_attempted and submissions:
                try:
                    notification = self.notifier.post_trade_alert(
                        plan=plan,
                        submissions=submissions,
                        instructions=plan.instructions,
                    )
                except Exception as exc:
                    plan.warnings.append(f"Discord trade alert failed: {type(exc).__name__}: {exc}")

            if self.broker.is_configured():
                try:
                    reconciliation = reconcile_broker_state(
                        store=self.store,
                        broker=self.broker,
                        plan=plan,
                        submissions=submissions,
                    )
                except Exception as exc:
                    plan.warnings.append(f"Broker reconciliation failed: {type(exc).__name__}: {exc}")

            record_trade_decision_run(
                store=self.store,
                plan=plan,
                analyses=analyses,
                readiness=readiness,
                settings=self.settings,
                account=account,
                submit_requested=bool(submit),
                submit_attempted=submit_attempted,
                submissions=submissions,
                reconciliation=reconciliation,
            )
            health = write_autonomy_health_snapshot(store=self.store, settings=self.settings, readiness=readiness)
            result = MonitorRunResult(
                generated_at=plan.generated_at,
                status="success",
                trigger_count=len(triggers),
                refreshed_symbols=[item.symbol for item in triggers],
                actionable_orders=len([item for item in plan.instructions if item.action in {"buy", "sell"}]),
                submitted_orders=len([item for item in submissions if item.status == "submitted"]),
                blockers=list(plan.blockers),
                warnings=list(plan.warnings),
                note=(
                    "No trigger-worthy symbols."
                    if not triggers
                    else f"health={health['status']}"
                ),
            )
            self._record_monitor_run(
                started_at=started_at,
                finished_at=utc_now_iso(),
                result=result,
                plan=plan,
                triggers=triggers,
                notification=notification,
                reconciliation=reconciliation,
                ingestion_summary=ingestion_summary,
                elapsed_seconds=time.time() - start,
            )
            return result
        except Exception as exc:
            if account is not None:
                record_portfolio_nav(store=self.store, account=account)
            if plan is not None and account is not None:
                record_trade_decision_run(
                    store=self.store,
                    plan=plan,
                    analyses=[],
                    readiness=readiness,
                    settings=self.settings,
                    account=account,
                    submit_requested=bool(submit),
                    submit_attempted=bool(submit),
                    submissions=submissions,
                    reconciliation=reconciliation,
                    error=f"{type(exc).__name__}: {exc}",
                )
            write_autonomy_health_snapshot(store=self.store, settings=self.settings, readiness=readiness)
            failure = MonitorRunResult(
                generated_at=utc_now_iso(),
                status="failed",
                trigger_count=len(triggers),
                refreshed_symbols=[item.symbol for item in triggers],
                actionable_orders=0,
                submitted_orders=0,
                blockers=[str(exc)],
                warnings=[],
                note=f"{type(exc).__name__}: {exc}",
            )
            self._record_monitor_run(
                started_at=started_at,
                finished_at=utc_now_iso(),
                result=failure,
                plan=plan,
                triggers=triggers,
                notification=notification,
                reconciliation=reconciliation,
                ingestion_summary=ingestion_summary,
                elapsed_seconds=time.time() - start,
            )
            raise

    def run_forever(
        self,
        *,
        interval_seconds: int | None = None,
        submit: bool = False,
        allow_blocked_readiness: bool = False,
        include_literature: bool = False,
        prefer_live: bool = True,
    ) -> None:
        interval = max(int(interval_seconds or self.settings.monitor_loop_interval_seconds), 30)
        while True:
            self.run_once(
                submit=submit,
                allow_blocked_readiness=allow_blocked_readiness,
                include_literature=include_literature,
                prefer_live=prefer_live,
            )
            time.sleep(interval)

    def _record_monitor_run(
        self,
        *,
        started_at: str,
        finished_at: str,
        result: MonitorRunResult,
        plan: ExecutionPlan | None,
        triggers: list[MonitorTrigger],
        notification: Any,
        reconciliation: BrokerReconciliationSummary | None,
        ingestion_summary: TriggerIngestionSummary | None,
        elapsed_seconds: float,
    ) -> None:
        metrics = {
            "trigger_count": result.trigger_count,
            "refreshed_symbols": len(result.refreshed_symbols),
            "actionable_orders": result.actionable_orders,
            "submitted_orders": result.submitted_orders,
            "ingested_symbols": 0 if ingestion_summary is None else ingestion_summary.symbols_polled,
            "symbols_with_new_events": 0 if ingestion_summary is None else ingestion_summary.symbols_with_new_events,
            "elapsed_seconds": round(float(elapsed_seconds), 2),
        }
        config = {
            "store_dir": self.settings.store_dir,
            "max_symbols_per_cycle": int(self.settings.monitor_max_symbols_per_cycle),
            "event_trigger_lookback_hours": int(self.settings.monitor_event_trigger_lookback_hours),
            "snapshot_stale_hours": int(self.settings.monitor_snapshot_stale_hours),
        }
        notes = result.note
        if notification is not None:
            notes += f" | discord_channel={notification.channel_id}"
        if reconciliation is not None:
            notes += f" | reconciliation_missing={len(reconciliation.missing_symbols)}"
        record_pipeline_run(
            store=self.store,
            job_name="monitor_vnext",
            status=result.status,
            started_at=started_at,
            finished_at=finished_at,
            metrics=metrics,
            config=config,
            notes=notes,
        )
        heartbeat = {
            "captured_at": finished_at,
            "status": result.status,
            "trigger_count": result.trigger_count,
            "refreshed_symbols": json.dumps(result.refreshed_symbols),
            "actionable_orders": result.actionable_orders,
            "submitted_orders": result.submitted_orders,
            "blocker_count": len(result.blockers),
            "warning_count": len(result.warnings),
        }
        self.store.append_records("monitor_heartbeats", [heartbeat])
        payload = {
            "started_at": started_at,
            "finished_at": finished_at,
            "result": asdict(result),
            "plan": None
            if plan is None
            else {
                "generated_at": plan.generated_at,
                "selected_symbols": list(plan.selected_symbols),
                "blockers": list(plan.blockers),
                "warnings": list(plan.warnings),
                "gross_cap_pct": plan.gross_cap_pct,
                "gross_cap_multiplier": plan.gross_cap_multiplier,
            },
            "triggers": [item.to_payload() for item in triggers],
            "ingestion_summary": None if ingestion_summary is None else asdict(ingestion_summary),
            "reconciliation": None if reconciliation is None else asdict(reconciliation),
        }
        self.store.write_raw_payload(
            "monitor_runs",
            f"monitor_run_{finished_at.replace(':', '-')}",
            payload,
        )

    def _refresh_live_trigger_events(
        self,
        *,
        now: datetime,
        manual_symbols: Iterable[str] | None = None,
        actual_symbols: Iterable[str] | None = None,
    ) -> TriggerIngestionSummary | None:
        if self.trigger_ingestor is None:
            return None
        watch_symbols = self._watch_symbols(
            now=now,
            manual_symbols=manual_symbols,
            actual_symbols=actual_symbols,
        )
        if not watch_symbols:
            return None
        return self.trigger_ingestor.ingest_symbols(
            watch_symbols,
            as_of=now,
        )

    def _watch_symbols(
        self,
        *,
        now: datetime,
        manual_symbols: Iterable[str] | None = None,
        actual_symbols: Iterable[str] | None = None,
    ) -> list[str]:
        ordered: list[str] = []

        def add(symbol: str) -> None:
            ticker = str(symbol).upper().strip()
            if not ticker or ticker in ordered:
                return
            ordered.append(ticker)

        for symbol in manual_symbols or []:
            add(symbol)
        for symbol in actual_symbols or []:
            add(symbol)
        for symbol in sorted(self._reconciliation_active_symbols()):
            add(symbol)
        for symbol in sorted(self._recent_active_symbols(now)):
            add(symbol)
        for symbol in self._latest_research_watch_symbols():
            add(symbol)

        limit = max(int(self.settings.monitor_event_watchlist_limit), 1)
        return ordered[:limit]

    def _recent_event_triggers(self, now: datetime) -> list[MonitorTrigger]:
        event_tape = self.store.read_table("event_tape")
        if event_tape.empty or "ticker" not in event_tape.columns:
            return []
        event_tape = event_tape.copy()
        event_tape["event_ts"] = event_tape["event_timestamp"].map(_ts)
        event_tape = event_tape.dropna(subset=["event_ts"])
        if event_tape.empty:
            return []
        now_ts = pd.Timestamp(now).tz_convert(None)
        lower = now_ts - pd.Timedelta(hours=max(int(self.settings.monitor_event_trigger_lookback_hours), 1))
        upper = now_ts + pd.Timedelta(hours=max(int(self.settings.monitor_event_trigger_forward_hours), 1))
        if "timing_exact" in event_tape.columns:
            event_tape = event_tape[event_tape["timing_exact"].fillna(False).astype(bool)]
        event_tape = event_tape[(event_tape["event_ts"] >= lower) & (event_tape["event_ts"] <= upper)]
        if event_tape.empty:
            return []
        event_tape["event_priority"] = event_tape["event_ts"].map(
            lambda ts: 95.0 if ts <= now_ts else 88.0 if ts <= now_ts + pd.Timedelta(hours=24) else 82.0
        )
        event_tape = event_tape.sort_values(["event_priority", "event_ts"], ascending=[False, False])
        latest = event_tape.drop_duplicates(subset=["ticker"], keep="first")
        return [
            MonitorTrigger(
                symbol=str(row.ticker).upper(),
                company_name=self._company_name_for(str(row.ticker)),
                priority=float(row.event_priority),
                source="event_tape",
                reason=(
                    f"Exact catalyst {row.event_type} "
                    f"{'just fired' if row.event_ts <= now_ts else 'is imminent'} at {row.event_ts.isoformat()}."
                ),
                event_type=None if pd.isna(row.event_type) else str(row.event_type),
                event_timestamp=row.event_ts.isoformat(),
            )
            for row in latest.itertuples(index=False)
        ]

    def _reconciliation_active_symbols(self) -> set[str]:
        reconciliations = self.store.read_table("broker_reconciliations")
        if reconciliations.empty or "captured_at" not in reconciliations.columns:
            return set()
        reconciliations = reconciliations.copy()
        reconciliations["captured_at_ts"] = reconciliations["captured_at"].map(_ts)
        row = reconciliations.sort_values("captured_at_ts").iloc[-1]
        symbols = set(_parse_listish(row.get("selected_symbols")))
        symbols.update(_parse_listish(row.get("actual_symbols")))
        symbols.update(_parse_listish(row.get("missing_symbols")))
        symbols.update(_parse_listish(row.get("unexpected_symbols")))
        return {str(symbol).upper() for symbol in symbols if str(symbol).strip()}

    def _recent_active_symbols(self, now: datetime) -> set[str]:
        decisions = self.store.read_table("trade_decisions")
        if decisions.empty or "generated_at" not in decisions.columns:
            return set()
        decisions = decisions.copy()
        decisions["generated_at_ts"] = decisions["generated_at"].map(_ts)
        cutoff = pd.Timestamp(now).tz_convert(None) - pd.Timedelta(
            hours=max(int(self.settings.monitor_recent_decision_lookback_hours), 1)
        )
        decisions = decisions[decisions["generated_at_ts"] >= cutoff]
        return {
            str(symbol).upper()
            for symbol in decisions.get("symbol", pd.Series(dtype=str)).dropna().astype(str).tolist()
            if symbol.strip()
        }

    def _stale_snapshot_triggers(self, now: datetime, active_symbols: Iterable[str]) -> list[MonitorTrigger]:
        snapshots = self.store.read_table("company_snapshots")
        if snapshots.empty or "ticker" not in snapshots.columns or "as_of" not in snapshots.columns:
            return []
        active = {str(symbol).upper() for symbol in active_symbols if str(symbol).strip()}
        if not active:
            return []
        snapshots = snapshots.copy()
        snapshots["ticker"] = snapshots["ticker"].astype(str).str.upper()
        snapshots = snapshots[snapshots["ticker"].isin(active)]
        if snapshots.empty:
            return []
        snapshots["as_of_ts"] = snapshots["as_of"].map(_ts)
        snapshots = snapshots.dropna(subset=["as_of_ts"]).sort_values("as_of_ts").drop_duplicates(subset=["ticker"], keep="last")
        if snapshots.empty:
            return []
        now_ts = pd.Timestamp(now).tz_convert(None)
        stale_cutoff = pd.Timedelta(hours=max(int(self.settings.monitor_snapshot_stale_hours), 1))
        triggers: list[MonitorTrigger] = []
        for row in snapshots.itertuples(index=False):
            age = now_ts - row.as_of_ts
            if age <= stale_cutoff:
                continue
            age_hours = age.total_seconds() / 3600.0
            triggers.append(
                MonitorTrigger(
                    symbol=str(row.ticker).upper(),
                    company_name=self._company_name_for(str(row.ticker)),
                    priority=65.0,
                    source="stale_snapshot",
                    reason=f"Active symbol snapshot is {age_hours:.1f}h old and needs an intraday refresh.",
                )
            )
        return triggers

    def _latest_research_watch_symbols(self) -> list[str]:
        recommendations = self.store.read_table("portfolio_recommendations")
        if recommendations.empty or "ticker" not in recommendations.columns:
            return []
        recommendations = recommendations.copy()
        if "as_of" in recommendations.columns:
            recommendations["as_of_ts"] = recommendations["as_of"].map(_ts)
            recommendations = recommendations.sort_values("as_of_ts").drop_duplicates(subset=["ticker"], keep="last")
        stance = recommendations["stance"].fillna("").astype(str).str.lower() if "stance" in recommendations.columns else ""
        if isinstance(stance, str):
            filtered = recommendations
        else:
            filtered = recommendations[stance.isin({"long", "short"})].copy()
            if filtered.empty:
                filtered = recommendations
        if "target_weight" not in filtered.columns:
            filtered["target_weight"] = 0.0
        if "confidence" not in filtered.columns:
            filtered["confidence"] = 0.0
        filtered["watch_score"] = (
            filtered["target_weight"].astype(float).abs() * filtered["confidence"].astype(float)
        )
        filtered = filtered.sort_values(["watch_score", "ticker"], ascending=[False, True])
        return filtered["ticker"].astype(str).str.upper().head(max(int(self.settings.monitor_event_watchlist_limit), 1)).tolist()

    def _company_name_for(self, symbol: str) -> str:
        ticker = str(symbol).upper()
        snapshots = self.store.read_table("company_snapshots")
        if not snapshots.empty and "ticker" in snapshots.columns:
            subset = snapshots[snapshots["ticker"].astype(str).str.upper() == ticker].copy()
            if not subset.empty:
                subset["as_of_ts"] = subset["as_of"].map(_ts) if "as_of" in subset.columns else pd.NaT
                subset = subset.sort_values("as_of_ts")
                row = subset.iloc[-1]
                value = str(row.get("company_name") or ticker).strip()
                if value:
                    return value
        membership = self.store.read_table("universe_membership")
        if not membership.empty and "ticker" in membership.columns:
            subset = membership[membership["ticker"].astype(str).str.upper() == ticker]
            if not subset.empty:
                row = subset.iloc[-1]
                value = str(row.get("company_name") or ticker).strip()
                if value:
                    return value
        return ticker
