"""
trade_vnext.py — PM-facing paper trading workflow for Tatetuck vNext.

Builds a fresh universe view, checks broker state, creates a PM-safe rebalance
plan, and optionally submits market orders to Alpaca paper trading.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

from biopharma_agent.vnext import TatetuckPlatform, VNextSettings, build_readiness_report
from biopharma_agent.vnext.autonomy import (
    record_portfolio_nav,
    record_trade_decision_run,
    reconcile_broker_state,
    write_autonomy_health_snapshot,
)
from biopharma_agent.vnext.execution import (
    AlpacaPaperBroker,
    DiscordTradeNotifier,
    PMExecutionPlanner,
    execute_plan,
    materialize_execution_feedback,
    record_trade_run,
)
from biopharma_agent.vnext.ops import utc_now_iso
from biopharma_agent.vnext.storage import LocalResearchStore
from biopharma_agent.vnext.universe import UniverseResolver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate or submit a Tatetuck vNext paper-trading plan.")
    parser.add_argument("--ticker", help="Restrict the PM run to a single ticker.")
    parser.add_argument("--company-name", help="Optional company name when using --ticker.")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of benchmark names analyzed.")
    parser.add_argument("--submit", action="store_true", help="Submit actionable orders to Alpaca paper trading.")
    parser.add_argument(
        "--allow-blocked-readiness",
        action="store_true",
        help="Allow paper order submission even if readiness blockers remain.",
    )
    parser.add_argument(
        "--include-literature",
        action="store_true",
        help="Include literature synthesis during the PM run.",
    )
    parser.add_argument(
        "--prefer-live",
        action="store_true",
        help="Prefer fresh live ingestion instead of the latest archived snapshot when both are available.",
    )
    return parser.parse_args()


def resolve_universe(args: argparse.Namespace, store: LocalResearchStore) -> list[tuple[str, str]]:
    if args.ticker:
        ticker = args.ticker.upper()
        return [(ticker, args.company_name or ticker)]
    resolver = UniverseResolver(store=store)
    return resolver.resolve_default_universe(limit=args.limit, prefer_archive=not args.prefer_live)


def main() -> None:
    args = parse_args()
    settings = VNextSettings.from_env()
    store = LocalResearchStore(settings.store_dir)
    platform = TatetuckPlatform(store=store)
    broker = AlpacaPaperBroker(settings=settings)
    planner = PMExecutionPlanner(settings=settings, store=store)
    started_at = utc_now_iso()
    start = time.time()

    analyses = platform.analyze_universe(
        resolve_universe(args, store=store),
        include_literature=args.include_literature or settings.include_literature,
        prefer_archive=not args.prefer_live,
        fallback_to_archive=True,
    )
    readiness = build_readiness_report(store=store, settings=settings)
    account = None
    positions = []
    submissions = []
    plan = None
    notification = None
    reconciliation = None
    try:
        if broker.is_configured():
            account = broker.account()
            broker.ensure_expected_account(account)
            positions = broker.positions()
        else:
            account = broker.simulated_account()
            readiness = replace(
                readiness,
                warnings=readiness.warnings
                + [
                    "Alpaca credentials are not configured. Using a simulated paper account and assuming no live positions."
                ],
            )
        record_portfolio_nav(store=store, account=account, captured_at=started_at)

        plan = planner.build_plan(
            analyses=analyses,
            account=account,
            positions=positions,
            readiness=readiness,
        )
        if not broker.is_configured() and args.submit:
            plan.blockers.append("Cannot submit orders without APCA_API_KEY_ID and APCA_API_SECRET_KEY.")

        submit = bool(args.submit)
        if plan.blockers and not args.allow_blocked_readiness:
            submit = False
        submissions = execute_plan(
            plan=plan,
            broker=broker,
            store=store,
            submit=submit,
        )
        if submit and submissions:
            notifier = DiscordTradeNotifier(settings=settings)
            try:
                notification = notifier.post_trade_alert(
                    plan=plan,
                    submissions=submissions,
                    instructions=plan.instructions,
                )
            except Exception as exc:
                plan.warnings.append(f"Discord trade alert failed: {type(exc).__name__}: {exc}")
        if broker.is_configured():
            try:
                reconciliation = reconcile_broker_state(
                    store=store,
                    broker=broker,
                    plan=plan,
                    submissions=submissions,
                )
                if submit and reconciliation.missing_symbols:
                    plan.warnings.append(
                        f"Broker is still missing {len(reconciliation.missing_symbols)} planned symbols after reconciliation."
                    )
                if reconciliation.unexpected_symbols:
                    plan.warnings.append(
                        f"Broker is carrying {len(reconciliation.unexpected_symbols)} unexpected symbols outside the current plan."
                    )
                if submit and reconciliation.rejected_order_count:
                    plan.warnings.append(
                        f"{reconciliation.rejected_order_count} broker orders were rejected or cancelled."
                    )
            except Exception as exc:
                plan.warnings.append(f"Broker reconciliation failed: {type(exc).__name__}: {exc}")
        record_trade_decision_run(
            store=store,
            plan=plan,
            analyses=analyses,
            readiness=readiness,
            settings=settings,
            account=account,
            submit_requested=bool(args.submit),
            submit_attempted=submit,
            submissions=submissions,
            reconciliation=reconciliation,
        )
        record_trade_run(
            store=store,
            settings=settings,
            plan=plan,
            submissions=submissions,
            submit=submit,
            started_at=started_at,
            status="success",
            notes="paper trade run completed",
        )
        feedback_summary = materialize_execution_feedback(store=store)
        write_autonomy_health_snapshot(store=store, settings=settings, readiness=readiness)
    except Exception as exc:
        if plan is None:
            from biopharma_agent.vnext.execution import ExecutionPlan

            plan = ExecutionPlan(
                generated_at=utc_now_iso(),
                account_id=settings.alpaca_paper_account_id,
                equity=0.0,
                buying_power=0.0,
                deployable_notional=0.0,
                selected_symbols=[],
                instructions=[],
                blockers=[str(exc)],
                warnings=[],
                readiness_status=readiness.status,
            )
        if account is not None:
            record_portfolio_nav(store=store, account=account)
        if account is not None and plan is not None:
            record_trade_decision_run(
                store=store,
                plan=plan,
                analyses=analyses,
                readiness=readiness,
                settings=settings,
                account=account,
                submit_requested=bool(args.submit),
                submit_attempted=bool(args.submit),
                submissions=submissions,
                reconciliation=reconciliation,
                error=f"{type(exc).__name__}: {exc}",
            )
        record_trade_run(
            store=store,
            settings=settings,
            plan=plan,
            submissions=submissions,
            submit=bool(args.submit),
            started_at=started_at,
            status="failed",
            notes=f"{type(exc).__name__}: {exc}",
        )
        write_autonomy_health_snapshot(store=store, settings=settings, readiness=readiness)
        raise

    print("=" * 72)
    print("  TATETUCK BOT vNEXT — PM PAPER TRADING")
    print("=" * 72)
    print(f"alpaca_account_id:         {account.account_id}")
    print(f"broker_mode:               {'live-paper' if broker.is_configured() else 'simulated-paper'}")
    print(f"paper_endpoint:            {broker.base_url}")
    print(f"readiness_status:          {readiness.status}")
    print(f"equity:                    {account.equity:,.2f}")
    print(f"buying_power:              {account.buying_power:,.2f}")
    print(f"deployable_notional:       {plan.deployable_notional:,.2f}")
    print(f"selected_symbols:          {', '.join(plan.selected_symbols) if plan.selected_symbols else 'none'}")
    print(f"submitted_orders:          {len([item for item in submissions if item.status == 'submitted'])}")
    print(f"execution_feedback_rows:   {feedback_summary.feedback_rows if 'feedback_summary' in locals() else 0}")
    if notification is not None:
        print(
            "discord_trade_alert:       "
            f"posted to {notification.channel_id} ({notification.order_count} orders"
            f"{', fallback' if notification.fallback_used else ''})"
        )
    else:
        print("discord_trade_alert:       none")

    print("\n[blockers]")
    if plan.blockers:
        for blocker in plan.blockers:
            print(f"- {blocker}")
    else:
        print("- none")

    print("\n[warnings]")
    if plan.warnings:
        for warning in plan.warnings:
            print(f"- {warning}")
    else:
        print("- none")

    print("\n[instructions]")
    if plan.instructions:
        for instruction in plan.instructions[:10]:
            payload = (
                f"{instruction.symbol:5} | action={instruction.action:4} | "
                f"scenario={instruction.scenario} | "
                f"target={instruction.scaled_target_weight:>4.1f}% | "
                f"delta=${instruction.delta_notional:>8.2f}"
            )
            if instruction.notional is not None:
                payload += f" | buy_notional=${instruction.notional:.2f}"
            if instruction.qty is not None:
                payload += f" | sell_qty={instruction.qty:.4f}"
            print(payload)
    else:
        print("- none")

    print(f"\n[summary] elapsed_seconds={time.time() - start:.1f}")


if __name__ == "__main__":
    main()
