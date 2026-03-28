"""
trade_vnext.py — PM-facing paper trading workflow for Tatetuck vNext.

Builds a fresh universe view, checks broker state, creates a PM-safe rebalance
plan, and optionally submits market orders to Alpaca paper trading.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import prepare

from biopharma_agent.vnext import TatetuckPlatform, VNextSettings, build_readiness_report
from biopharma_agent.vnext.execution import AlpacaPaperBroker, PMExecutionPlanner, execute_plan, record_trade_run
from biopharma_agent.vnext.ops import utc_now_iso
from biopharma_agent.vnext.storage import LocalResearchStore


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


def resolve_universe(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.ticker:
        ticker = args.ticker.upper()
        return [(ticker, args.company_name or ticker)]
    universe = list(prepare.BENCHMARK_TICKERS)
    if args.limit and args.limit > 0:
        return universe[: args.limit]
    return universe


def main() -> None:
    args = parse_args()
    settings = VNextSettings.from_env()
    store = LocalResearchStore(settings.store_dir)
    platform = TatetuckPlatform(store=store)
    broker = AlpacaPaperBroker(settings=settings)
    planner = PMExecutionPlanner(settings=settings)
    started_at = utc_now_iso()
    start = time.time()

    analyses = platform.analyze_universe(
        resolve_universe(args),
        include_literature=args.include_literature or settings.include_literature,
        prefer_archive=not args.prefer_live,
        fallback_to_archive=True,
    )
    readiness = build_readiness_report(store=store, settings=settings)
    account = None
    positions = []
    submissions = []
    plan = None
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
