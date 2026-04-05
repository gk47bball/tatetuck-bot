from __future__ import annotations

import argparse
import asyncio
import os
from datetime import datetime, time, timezone
from functools import partial
from pathlib import Path

import discord
import pandas as pd
from discord.ext import commands, tasks
from dotenv import load_dotenv

from prepare import HOLDOUT_TICKERS, TRAIN_TICKERS

from biopharma_agent.vnext import (
    AlpacaPaperBroker,
    PMExecutionPlanner,
    TatetuckPlatform,
    VNextSettings,
    build_readiness_report,
)
from biopharma_agent.vnext.storage import LocalResearchStore
from biopharma_agent.vnext.taxonomy import event_type_bucket, event_type_priority, is_synthetic_event
from biopharma_agent.vnext.universe import UniverseResolver


load_dotenv()

BOT_GUIDE_PATH = Path(__file__).with_name("BOT_GUIDE.md")

GUIDE_OVERVIEW = (
    "Tatetuck Bot is a biotech research assistant. It looks at a company’s drug programs, "
    "upcoming catalysts, cash runway, and recent market behavior, then turns that into a "
    "simple PM-style view of which names look strongest over the next 1-6 months."
)

GUIDE_HOW_IT_WORKS = (
    "In plain English: the bot treats each biotech company like a small portfolio of bets. "
    "It asks which drug is most important, how likely the next catalyst is to matter, how big "
    "the commercial opportunity could be, and whether the company has enough cash to get there "
    "without painful dilution."
)

GUIDE_OUTPUTS = (
    "`Confidence` is how strong the overall setup looks. "
    "`Expected Return` is the model’s directional 90-day view. "
    "`Catalyst Success` is the probability the next key event is favorable. "
    "`Action` tells you whether the bot currently likes the name as a long or a short. "
    "`Target Weight` is a paper-portfolio sizing suggestion, not a guarantee."
)

GUIDE_COMMANDS = (
    "`!analyze TICKER` for one tear sheet, `!top5` for the current best long and short ideas, "
    "`!guide` for the layman version, and `!status` for bot + research health."
)

LIVE_ANALYSIS_MAX_ARCHIVE_AGE_DAYS = 14
TOP5_UNIVERSE_LIMIT = len(TRAIN_TICKERS) + len(HOLDOUT_TICKERS)
TOP5_SCAN_MAX_CONCURRENCY = 6


def live_first_analysis_kwargs(*, include_literature: bool = False, persist: bool = False) -> dict[str, object]:
    return {
        "include_literature": include_literature,
        "prefer_archive": False,
        "fallback_to_archive": True,
        "persist": persist,
        "max_archive_age_days": LIVE_ANALYSIS_MAX_ARCHIVE_AGE_DAYS,
        "allow_stale_archive_fallback": False,
    }


def _normalize_now(now_dt: datetime | None = None) -> datetime:
    if now_dt is None:
        return datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        return now_dt.replace(tzinfo=timezone.utc)
    return now_dt.astimezone(timezone.utc)


def _has_value(value) -> bool:
    if value is None:
        return False
    try:
        return not pd.isna(value)
    except TypeError:
        return True


def _coerce_event_timestamp(value) -> datetime | None:
    if not _has_value(value):
        return None
    ts = pd.to_datetime(value, errors="coerce", utc=True, format="mixed")
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()


def _is_upcoming_event(expected_date, now_dt: datetime) -> bool:
    event_dt = _coerce_event_timestamp(expected_date)
    if event_dt is None:
        return False
    text = str(expected_date).strip()
    if "T" not in text and len(text) <= 10:
        return event_dt.date() >= now_dt.date()
    return event_dt >= now_dt


def _event_exact(event) -> bool:
    timing_exact = getattr(event, "timing_exact", None)
    if timing_exact is not None:
        return bool(timing_exact)
    status = str(getattr(event, "status", "") or "")
    return "exact" in status


def _select_primary_catalyst(snapshot, preferred_event_type: str | None = None):
    if not getattr(snapshot, "catalyst_events", None):
        return None
    candidates = list(snapshot.catalyst_events)
    if preferred_event_type:
        typed = [event for event in candidates if event.event_type == preferred_event_type]
        if typed:
            candidates = typed
    reference_time = _coerce_event_timestamp(getattr(snapshot, "as_of", None)) or _normalize_now()
    upcoming = [event for event in candidates if _is_upcoming_event(event.expected_date, reference_time)]
    if upcoming:
        candidates = upcoming
    return min(
        candidates,
        key=lambda event: (-event_type_priority(event.event_type), event.horizon_days, -event.importance),
    )


def upcoming_event_text(snapshot, preferred_event_type: str | None = None, now_dt: datetime | None = None) -> str:
    now_dt = _normalize_now(now_dt)
    event = _select_primary_catalyst(snapshot, preferred_event_type=preferred_event_type)
    if event is None or not _is_upcoming_event(event.expected_date, now_dt):
        return "none upcoming"
    date_text = str(event.expected_date)[:10] if _has_value(event.expected_date) else "TBD"
    timing_text = "exact" if _event_exact(event) else "estimated"
    return f"{event.event_type} on {date_text} ({timing_text})"


def brief_event_text(snapshot, preferred_event_type: str | None = None, now_dt: datetime | None = None) -> str:
    now_dt = _normalize_now(now_dt)
    metadata = getattr(snapshot, "metadata", {}) or {}
    if str(metadata.get("special_situation") or "") == "pending_transaction":
        return str(metadata.get("special_situation_label") or "pending transaction")
    event = _select_primary_catalyst(snapshot, preferred_event_type=preferred_event_type)
    if event is None or not _is_upcoming_event(event.expected_date, now_dt):
        return "no clean upcoming catalyst"
    event_bucket = event_type_bucket(getattr(event, "event_type", None), getattr(event, "title", None))
    status = str(getattr(event, "status", "") or "")
    synthetic = is_synthetic_event(status, getattr(event, "title", None))
    date_text = str(event.expected_date)[:10] if _has_value(event.expected_date) else "TBD"
    if event_bucket == "commercial" and synthetic:
        return "franchise setup, no clean dated catalyst"
    if event_bucket in {"clinical", "regulatory", "strategic"} and float(getattr(event, "horizon_days", 999.0) or 999.0) > 120.0:
        return f"{event.event_type} on {date_text} (longer-dated)"
    if status == "guided_company_event":
        return f"{event.event_type} targeted for {date_text} (guided)"
    timing_text = "exact" if _event_exact(event) else "estimated"
    return f"{event.event_type} on {date_text} ({timing_text})"


def brief_setup_driver_text(snapshot, preferred_event_type: str | None = None) -> str:
    metadata = getattr(snapshot, "metadata", {}) or {}
    driver_label = str(metadata.get("driver_label") or "").strip()
    driver_indication = str(metadata.get("driver_indication") or "").strip()
    if driver_label:
        return driver_label if not driver_indication else f"{driver_label} | {driver_indication}"
    primary_event = _select_primary_catalyst(snapshot, preferred_event_type=preferred_event_type)
    approved_products = list(getattr(snapshot, "approved_products", []) or [])
    programs = list(getattr(snapshot, "programs", []) or [])
    if approved_products:
        lead_product = approved_products[0]
        indication = getattr(lead_product, "indication", None) or "unspecified indication"
        return f"{lead_product.name} | {indication}"
    if primary_event is not None and getattr(primary_event, "program_id", None):
        program = next((item for item in programs if item.program_id == primary_event.program_id), None)
        if program is not None:
            condition = program.conditions[0] if getattr(program, "conditions", None) else "unspecified indication"
            return f"{program.name} | {condition}"
    if programs:
        program = programs[0]
        condition = program.conditions[0] if getattr(program, "conditions", None) else "unspecified indication"
        return f"{program.name} | {condition}"
    return str(getattr(snapshot, "ticker", "unknown"))


def brief_pick_profile(analysis, now_dt: datetime | None = None) -> str:
    now_dt = _normalize_now(now_dt)
    special_situation = str((getattr(analysis.snapshot, "metadata", {}) or {}).get("special_situation") or "")
    if special_situation == "pending_transaction":
        return "special_situation"
    event = _select_primary_catalyst(analysis.snapshot, preferred_event_type=analysis.signal.primary_event_type)
    if event is None or not _is_upcoming_event(event.expected_date, now_dt):
        return "franchise_setup"
    bucket = event_type_bucket(getattr(event, "event_type", None), getattr(event, "title", None))
    synthetic = is_synthetic_event(str(getattr(event, "status", "") or ""), getattr(event, "title", None))
    if bucket == "commercial" and synthetic:
        return "franchise_setup"
    if bucket in {"clinical", "regulatory", "strategic"} and float(getattr(event, "horizon_days", 999.0) or 999.0) <= 120.0:
        return "timely_catalyst"
    if bucket in {"clinical", "regulatory", "strategic"}:
        return "extended_catalyst"
    return "franchise_setup"


def rank_deployable_ideas(
    deployable: list[tuple[object, object]],
    now_dt: datetime | None = None,
) -> list[tuple[object, object]]:
    now_dt = _normalize_now(now_dt)

    def sort_key(item: tuple[object, object]) -> tuple[float, ...]:
        instruction, analysis = item
        signal = analysis.signal
        event = _select_primary_catalyst(analysis.snapshot, preferred_event_type=signal.primary_event_type)
        profile = brief_pick_profile(analysis, now_dt=now_dt)
        profile_score = {
            "timely_catalyst": 3.0,
            "franchise_setup": 2.0,
            "extended_catalyst": 1.0,
            "special_situation": 0.0,
        }.get(profile, 0.0)
        if event is None or not _is_upcoming_event(event.expected_date, now_dt):
            event_clean = 0.0
            timing_score = -1.0
            bucket_score = -1.0
            importance = 0.0
            horizon_score = -999.0
        else:
            bucket = event_type_bucket(getattr(event, "event_type", None), getattr(event, "title", None))
            status = str(getattr(event, "status", "") or "")
            synthetic = is_synthetic_event(status, getattr(event, "title", None))
            event_clean = 0.0 if bucket == "commercial" and synthetic else 1.0 if bucket in {"clinical", "regulatory", "strategic"} else 0.5
            if _event_exact(event):
                timing_score = 2.0
            elif status == "guided_company_event":
                timing_score = 1.0
            elif bucket == "commercial" and synthetic:
                timing_score = -1.0
            else:
                timing_score = 0.0
            bucket_score = {
                "strategic": 3.0,
                "regulatory": 2.5,
                "clinical": 2.0,
                "commercial": 1.0,
                "earnings": 0.5,
                "financing": -1.0,
            }.get(bucket, 0.0)
            importance = float(getattr(event, "importance", 0.0) or 0.0)
            horizon_score = -float(getattr(event, "horizon_days", 999.0) or 999.0)
        return (
            profile_score,
            event_clean,
            timing_score,
            bucket_score,
            float(signal.expected_return),
            float(signal.confidence),
            float(getattr(instruction, "scaled_target_weight", 0.0) or 0.0),
            importance,
            horizon_score,
        )

    return sorted(deployable, key=sort_key, reverse=True)


def idea_action_label(analysis) -> str:
    stance = str(getattr(analysis.portfolio, "stance", "") or "").strip().lower()
    if stance == "short":
        return "short"
    if stance == "long":
        return "long"
    return "avoid"


def qualifies_top_idea(analysis) -> bool:
    stance = idea_action_label(analysis)
    signal = analysis.signal
    target_weight = float(getattr(analysis.portfolio, "target_weight", 0.0) or 0.0)
    confidence = float(getattr(analysis.portfolio, "confidence", 0.0) or 0.0)
    special_situation = str((getattr(analysis.snapshot, "metadata", {}) or {}).get("special_situation") or "")
    if special_situation == "pending_transaction":
        return False
    if stance == "long":
        return (
            target_weight >= 1.0
            and float(signal.expected_return) > 0.05
            and confidence >= 0.55
            and analysis.portfolio.scenario not in {"watchlist only", "avoid due to financing"}
        )
    if stance == "short":
        return (
            target_weight >= 1.0
            and float(signal.expected_return) < -0.05
            and float(signal.internal_upside_pct or 0.0) <= -0.10
            and confidence >= 0.55
        )
    return False


def rank_top_idea_analyses(
    analyses: list,
    now_dt: datetime | None = None,
) -> list:
    now_dt = _normalize_now(now_dt)
    candidates = [analysis for analysis in analyses if qualifies_top_idea(analysis)]

    def sort_key(analysis) -> tuple[float, ...]:
        signal = analysis.signal
        event = _select_primary_catalyst(analysis.snapshot, preferred_event_type=signal.primary_event_type)
        profile = brief_pick_profile(analysis, now_dt=now_dt)
        profile_score = {
            "timely_catalyst": 3.0,
            "franchise_setup": 2.0,
            "extended_catalyst": 1.0,
            "special_situation": 0.0,
        }.get(profile, 0.0)
        if event is None or not _is_upcoming_event(event.expected_date, now_dt):
            event_clean = 0.0
            timing_score = -1.0
            bucket_score = -1.0
            importance = 0.0
            horizon_score = -999.0
        else:
            bucket = event_type_bucket(getattr(event, "event_type", None), getattr(event, "title", None))
            status = str(getattr(event, "status", "") or "")
            synthetic = is_synthetic_event(status, getattr(event, "title", None))
            event_clean = 0.0 if bucket == "commercial" and synthetic else 1.0 if bucket in {"clinical", "regulatory", "strategic"} else 0.5
            if _event_exact(event):
                timing_score = 2.0
            elif status == "guided_company_event":
                timing_score = 1.0
            elif bucket == "commercial" and synthetic:
                timing_score = -1.0
            else:
                timing_score = 0.0
            bucket_score = {
                "strategic": 3.0,
                "regulatory": 2.5,
                "clinical": 2.0,
                "commercial": 1.0,
                "earnings": 0.5,
                "financing": -1.0,
            }.get(bucket, 0.0)
            importance = float(getattr(event, "importance", 0.0) or 0.0)
            horizon_score = -float(getattr(event, "horizon_days", 999.0) or 999.0)
        return (
            profile_score,
            event_clean,
            timing_score,
            bucket_score,
            abs(float(signal.expected_return)),
            abs(float(signal.internal_upside_pct or 0.0)),
            float(signal.confidence),
            float(getattr(analysis.portfolio, "target_weight", 0.0) or 0.0),
            1.0 if idea_action_label(analysis) == "short" else 0.0,
            importance,
            horizon_score,
        )

    return sorted(candidates, key=sort_key, reverse=True)


def catalyst_alpha_gate(catalyst_audit: dict | None) -> tuple[bool, dict[str, dict], str | None]:
    payload = catalyst_audit or {}
    sleeve_scorecards = payload.get("sleeve_scorecards", {}) or {}
    scorecard = {
        "pre_event_long": dict(sleeve_scorecards.get("pre_event_long", {}) or {}),
        "post_event_reaction_long": dict(sleeve_scorecards.get("post_event_reaction_long", {}) or {}),
        "event_short_or_pairs": dict(sleeve_scorecards.get("event_short_or_pairs", {}) or {}),
    }
    pre_event = scorecard["pre_event_long"]
    if not pre_event or float(pre_event.get("rows", 0.0) or 0.0) <= 0.0:
        return False, scorecard, "missing_scorecard"

    gates = payload.get("gates", {}) or {}
    if not bool((gates.get("pre_event_long", {}) or {}).get("passed", False)):
        return False, scorecard, "pre_event_gated"
    family_depth = gates.get("family_depth", {}) or {}
    if family_depth and not bool(family_depth.get("passed", False)):
        return False, scorecard, "family_depth_gated"

    strong_enough = (
        float(pre_event.get("rows", 0.0) or 0.0) >= 25.0
        and float(pre_event.get("rank_ic", 0.0) or 0.0) >= 0.0
        and float(pre_event.get("cost_adjusted_top_bottom_spread", 0.0) or 0.0) > 0.0
    )
    return strong_enough, scorecard, None if strong_enough else "weak_pre_event_edge"


def rank_top_catalyst_ideas(analyses: list, now_dt: datetime | None = None) -> list[tuple[object, object, float]]:
    now_dt = _normalize_now(now_dt)
    ranked: list[tuple[object, object, float]] = []
    for analysis in analyses:
        metadata = getattr(analysis, "metadata", {}) or {}
        signal = analysis.signal
        setup_type = metadata.get("setup_type") or signal.setup_type or analysis.portfolio.setup_type
        if setup_type not in {"hard_catalyst", "soft_catalyst"} and signal.primary_event_bucket not in {"clinical", "regulatory", "strategic"}:
            continue
        event = _select_primary_catalyst(analysis.snapshot, preferred_event_type=signal.primary_event_type)
        if event is None or not _is_upcoming_event(event.expected_date, now_dt):
            continue
        score = (
            (2.5 if _event_exact(event) else 0.0)
            + (1.25 if setup_type == "hard_catalyst" else 0.5 if setup_type == "soft_catalyst" else 0.0)
            + (0.75 if signal.primary_event_bucket == "strategic" else 0.0)
            + (0.80 * float(signal.catalyst_success_prob))
            + (0.55 * float(signal.confidence))
            + (0.35 * float(signal.expected_return))
            + (0.12 * float(analysis.portfolio.target_weight))
            + (0.05 * float(event.importance))
            - (0.01 * max(float(event.horizon_days), 0.0))
        )
        ranked.append((analysis, event, float(score)))
    return sorted(ranked, key=lambda item: item[2], reverse=True)


def _materialize_event_master_row(row: pd.Series) -> dict[str, object]:
    payload = row.to_dict()
    for field in (
        "expected_date",
        "timing_exact",
        "timing_synthetic",
        "deployable_pre_event",
        "pre_event_score",
        "confidence_proxy",
        "success_probability_proxy",
        "floor_support_pct",
        "financing_risk_proxy",
        "importance",
        "source",
    ):
        latest_value = payload.get(f"latest_{field}")
        if _has_value(latest_value):
            payload[field] = latest_value
    return payload


def rank_current_catalyst_events(
    event_master: pd.DataFrame,
    catalyst_audit: dict | None,
    now_dt: datetime | None = None,
) -> list[dict[str, object]]:
    now_dt = _normalize_now(now_dt)
    if event_master is None or event_master.empty:
        return []

    allowed, _scorecard, _reason = catalyst_alpha_gate(catalyst_audit)
    if not allowed:
        return []

    family_scorecards = ((catalyst_audit or {}).get("family_scorecards", {}) or {})
    ranked: list[dict[str, object]] = []
    for _, raw_row in event_master.iterrows():
        row = _materialize_event_master_row(raw_row)
        if not _is_upcoming_event(row.get("expected_date"), now_dt):
            continue
        if not bool(row.get("deployable_pre_event", False)):
            continue
        family_metrics = family_scorecards.get(str(row.get("event_family") or ""), {}) or {}
        score = (
            float(row.get("pre_event_score", 0.0) or 0.0)
            + (0.35 if bool(row.get("timing_exact")) else 0.0)
            + (0.20 * float(family_metrics.get("cost_adjusted_top_bottom_spread", 0.0) or 0.0))
            + (0.10 * float(row.get("confidence_proxy", 0.0) or 0.0))
            + (0.10 * float(row.get("success_probability_proxy", 0.0) or 0.0))
            + (0.08 * float(row.get("floor_support_pct", 0.0) or 0.0))
            + (0.05 * float(row.get("importance", 0.0) or 0.0))
            - (0.10 * float(row.get("financing_risk_proxy", 0.0) or 0.0))
        )
        row["ranking_score"] = float(score)
        ranked.append(row)

    return sorted(
        ranked,
        key=lambda item: (
            float(item.get("ranking_score", 0.0) or 0.0),
            bool(item.get("timing_exact", False)),
            float(item.get("pre_event_score", 0.0) or 0.0),
        ),
        reverse=True,
    )


def _format_metric_line(label: str, metrics: dict | None, metric_key: str = "cost_adjusted_top_bottom_spread") -> str:
    metrics = metrics or {}
    spread = float(metrics.get(metric_key, 0.0) or 0.0)
    rank_ic = float(metrics.get("rank_ic", 0.0) or 0.0)
    rows = float(metrics.get("rows", 0.0) or 0.0)
    return f"{label}: spread {spread:+.2f} | IC {rank_ic:+.2f} | rows {rows:.0f}"


def build_dashboard_embed(audit_payload: dict, catalyst_audit: dict | None, readiness) -> discord.Embed:
    payload = audit_payload or {}
    catalyst_allowed, catalyst_scorecard, catalyst_reason = catalyst_alpha_gate(catalyst_audit)
    baseline_model = ((payload.get("baseline_scorecards", {}) or {}).get("model", {}) or {})
    alpha_sleeves = payload.get("alpha_sleeve_scorecards", {}) or {}
    setup_scorecards = payload.get("setup_type_scorecards", {}) or {}
    a_grade_gates = payload.get("a_grade_gates", {}) or {}
    catalyst_surface = (a_grade_gates.get("catalyst_surface", {}) or {}).get("status", "unknown")

    embed = discord.Embed(
        title="PM Alpha Dashboard",
        description="Validation health for the PM stack and the catalyst sleeve.",
        color=0x2ECC71 if getattr(readiness, "status", "") == "production_ready" else 0xF39C12,
    )
    embed.add_field(
        name="Core Scorecard",
        value=(
            f"Strict Rank IC: {float(payload.get('strict_rank_ic', 0.0) or 0.0):+.2f}\n"
            f"Exact primary events: {float(payload.get('exact_primary_event_rate', 0.0) or 0.0) * 100:.1f}%\n"
            f"Synthetic primary events: {float(payload.get('synthetic_primary_event_rate', 0.0) or 0.0) * 100:.1f}%\n"
            f"Model sleeve: {_format_metric_line('Model', baseline_model, metric_key='cost_adjusted_top_bottom_spread')}"
        ),
        inline=False,
    )
    embed.add_field(
        name="Catalyst Bot",
        value=(
            f"Pre-event gate: {'open' if catalyst_allowed else 'closed'}\n"
            f"Reason: {catalyst_reason or 'validated'}\n"
            f"Pre-event sleeve: {_format_metric_line('Pre', catalyst_scorecard.get('pre_event_long'))}\n"
            f"Surface status: {catalyst_surface}"
        ),
        inline=False,
    )
    embed.add_field(
        name="Alpha Sleeves",
        value=(
            f"PM: {_format_metric_line('PM', alpha_sleeves.get('pm_alpha'))}\n"
            f"Franchise: {_format_metric_line('Franchise', alpha_sleeves.get('franchise_alpha'))}\n"
            f"Catalyst: {catalyst_surface}"
        ),
        inline=False,
    )
    watch_items = [
        f"Readiness: {getattr(readiness, 'status', 'unknown')}",
        f"A-grade ready: {'yes' if bool((a_grade_gates.get('a_grade_ready', {}) or {}).get('passed', False)) else 'no'}",
        f"Launch asymmetry: {float((setup_scorecards.get('launch_asymmetry', {}) or {}).get('cost_adjusted_top_bottom_spread', 0.0) or 0.0):+.2f}",
        f"Hard catalyst: {float((setup_scorecards.get('hard_catalyst', {}) or {}).get('cost_adjusted_top_bottom_spread', 0.0) or 0.0):+.2f}",
    ]
    blockers = list(getattr(readiness, "blockers", []) or [])
    if blockers:
        watch_items.extend(blockers[:2])
    embed.add_field(name="Watch Items", value="\n".join(watch_items), inline=False)
    embed.set_footer(text="Use !status for the live operational view.")
    return embed


def get_discord_token() -> str | None:
    return os.getenv("DISCORD_TOKEN") or os.getenv("DISCORD_BOT_TOKEN")


def get_target_channel_id() -> str | None:
    return os.getenv("DISCORD_CHANNEL_ID") or os.getenv("TATETUCK_DISCORD_CHANNEL_ID")


async def collect_fresh_analyses(
    platform: TatetuckPlatform,
    universe: list[tuple[str, str]],
    *,
    include_literature: bool = False,
    persist: bool = False,
    max_concurrency: int = TOP5_SCAN_MAX_CONCURRENCY,
) -> list:
    loop = asyncio.get_running_loop()
    semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
    analysis_kwargs = live_first_analysis_kwargs(include_literature=include_literature, persist=persist)
    deduped_universe: list[tuple[str, str]] = []
    seen: set[str] = set()
    for ticker, company_name in universe:
        normalized = str(ticker).upper()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped_universe.append((normalized, company_name))

    async def analyze_symbol(ticker: str, company_name: str):
        async with semaphore:
            try:
                return await loop.run_in_executor(
                    None,
                    partial(
                        platform.analyze_ticker,
                        ticker,
                        company_name=company_name,
                        **analysis_kwargs,
                    ),
                )
            except Exception:
                return None

    tasks = [analyze_symbol(ticker, company_name) for ticker, company_name in deduped_universe]
    results = await asyncio.gather(*tasks)
    return [analysis for analysis in results if analysis is not None]


def build_platform() -> tuple[TatetuckPlatform, VNextSettings, LocalResearchStore]:
    settings = VNextSettings.from_env()
    store = LocalResearchStore(settings.store_dir)
    return TatetuckPlatform(store=store), settings, store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or self-check the Tatetuck Discord bot.")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Validate local bot configuration and the research backend without connecting to Discord.",
    )
    return parser.parse_args()


def run_self_check() -> int:
    token = get_discord_token()
    channel_id = get_target_channel_id()
    platform, settings, store = build_platform()
    universe_resolver = UniverseResolver(store=store)
    broker = AlpacaPaperBroker(settings=settings)
    planner = PMExecutionPlanner(settings=settings)
    readiness = build_readiness_report(store=store, settings=settings)

    print("=" * 72)
    print("  TATETUCK BOT — DISCORD SELF CHECK")
    print("=" * 72)
    print(f"discord_token_present:     {bool(token)}")
    print(f"discord_channel_present:   {bool(channel_id)}")
    print(f"discord_py_version:        {discord.__version__}")
    print(f"store_dir:                 {settings.store_dir}")
    print(f"readiness_status:          {readiness.status}")
    print(f"walkforward_windows:       {readiness.walkforward_windows}")
    print(f"leakage_passed:            {readiness.leakage_passed}")

    try:
        analysis = platform.analyze_ticker(
            "CRSP",
            company_name="CRISPR Therapeutics",
            **live_first_analysis_kwargs(include_literature=False, persist=False),
        )
        print(f"sample_analysis:           ok ({analysis.snapshot.ticker} {analysis.portfolio.scenario})")
    except Exception as exc:
        print(f"sample_analysis:           failed ({type(exc).__name__}: {exc})")
        return 1

    if not token:
        print("\n[result]")
        print("- Bot code is healthy, but it cannot connect to Discord until DISCORD_TOKEN is set.")
        return 1

    print("\n[result]")
    print("- Bot is configured locally and ready to connect to Discord.")
    return 0


def build_guide_embed() -> discord.Embed:
    embed = discord.Embed(
        title="Tatetuck Bot Guide",
        description="Biotech event-driven research, explained for someone with basic finance knowledge.",
        color=0x2ecc71,
    )
    embed.add_field(name="What It Does", value=GUIDE_OVERVIEW, inline=False)
    embed.add_field(name="How It Thinks", value=GUIDE_HOW_IT_WORKS, inline=False)
    embed.add_field(name="How To Read The Output", value=GUIDE_OUTPUTS, inline=False)
    embed.add_field(name="Commands", value=GUIDE_COMMANDS, inline=False)
    embed.set_footer(text="Tatetuck Bot vNext")
    return embed


async def send_and_optionally_pin_guide(channel: discord.abc.Messageable) -> tuple[discord.Message | None, str]:
    guide_embed = build_guide_embed()
    message = await channel.send(embed=guide_embed)
    pin_status = "posted"
    try:
        if isinstance(message.channel, discord.TextChannel):
            await message.pin(reason="Pinned Tatetuck Bot guide")
            pin_status = "posted and pinned"
    except discord.Forbidden:
        pin_status = "posted but not pinned (missing Manage Messages permission)"
    except discord.HTTPException:
        pin_status = "posted but pinning failed"
    return message, pin_status


def build_bot() -> commands.Bot:
    token = get_discord_token()
    if not token:
        raise RuntimeError("DISCORD_TOKEN is not set. Add it to .env before starting the Discord bot.")

    target_channel_id = get_target_channel_id()
    platform, settings, store = build_platform()
    universe_resolver = UniverseResolver(store=store)
    broker = AlpacaPaperBroker(settings=settings)
    planner = PMExecutionPlanner(settings=settings)

    intents = discord.Intents.default()
    intents.message_content = True
    intents.guilds = True

    bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

    async def analyze_one(ticker: str, company_name: str):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(
                platform.analyze_ticker,
                ticker,
                company_name=company_name,
                **live_first_analysis_kwargs(include_literature=False, persist=False),
            ),
        )

    async def build_pm_top_ideas():
        universe = universe_resolver.resolve_default_universe(
            limit=TOP5_UNIVERSE_LIMIT,
            prefer_archive=True,
        )
        analyses = await collect_fresh_analyses(
            platform,
            universe,
            include_literature=False,
            persist=False,
        )
        loop = asyncio.get_running_loop()
        readiness = await loop.run_in_executor(
            None,
            partial(build_readiness_report, store=store, settings=settings),
        )
        top_ideas = rank_top_idea_analyses(analyses)
        return top_ideas, readiness

    def resolve_channel() -> discord.abc.Messageable | None:
        channel = None
        if target_channel_id:
            try:
                channel = bot.get_channel(int(target_channel_id))
            except ValueError:
                channel = None
        if channel is not None:
            return channel

        for guild in bot.guilds:
            me = guild.me or guild.get_member(bot.user.id if bot.user else 0)
            if me is None:
                continue
            for text_channel in guild.text_channels:
                if text_channel.permissions_for(me).send_messages:
                    return text_channel
        return None

    def format_money(value: float | None) -> str:
        if value is None:
            return "n/a"
        absolute = abs(float(value))
        if absolute >= 1_000_000_000:
            return f"${value / 1_000_000_000:.2f}B"
        if absolute >= 1_000_000:
            return f"${value / 1_000_000:.1f}M"
        return f"${value:,.0f}"

    def runway_label(snapshot) -> str:
        runway = float(snapshot.metadata.get("runway_months", 0.0) or 0.0)
        capped = bool(snapshot.metadata.get("runway_months_capped"))
        if capped:
            return "120m+"
        if runway <= 0:
            return "0m"
        return f"{runway:.1f}m"

    def primary_catalyst(snapshot, preferred_event_type: str | None = None):
        return _select_primary_catalyst(snapshot, preferred_event_type=preferred_event_type)

    def lead_program(snapshot, primary_event):
        programs = list(getattr(snapshot, "programs", []) or [])
        if not programs:
            return None
        if primary_event is not None and getattr(primary_event, "program_id", None):
            event_program = next((item for item in programs if item.program_id == primary_event.program_id), None)
            if event_program is not None:
                return event_program
        return programs[0]

    def build_tearsheet_embed(analysis) -> discord.Embed:
        snapshot = analysis.snapshot
        primary_event = primary_catalyst(snapshot, analysis.signal.primary_event_type)
        lead_program_item = lead_program(snapshot, primary_event)
        action_label = idea_action_label(analysis)
        target_weight_text = (
            f"{analysis.portfolio.target_weight}% {action_label}"
            if action_label in {"long", "short"}
            else f"{analysis.portfolio.target_weight}%"
        )
        approved_names = ", ".join(item.name for item in snapshot.approved_products[:2]) if snapshot.approved_products else ""
        commercial_truth = (
            approved_names
            if approved_names
            else ("reported commercial revenue (product map unverified)" if snapshot.metadata.get("commercial_revenue_present") else "pre-commercial")
        )
        net_cash = snapshot.cash - snapshot.debt
        source = analysis.metadata.get("analysis_source", "unknown")
        snapshot_age_days = analysis.metadata.get("snapshot_age_days")
        company_state = (analysis.metadata.get("company_state") or "unknown").replace("_", " ")
        setup_type = (analysis.metadata.get("setup_type") or "watchful").replace("_", " ")
        internal_value = analysis.metadata.get("internal_value")
        internal_price_target = analysis.metadata.get("internal_price_target")
        internal_upside_pct = analysis.metadata.get("internal_upside_pct")
        asymmetry_label = analysis.metadata.get("asymmetry_label", "unclear asymmetry")
        embed = discord.Embed(
            title=f"Tatetuck Analyst Tear Sheet: {snapshot.ticker}",
            description="Event-driven biopharma alpha profile",
            color=0x2ecc71 if analysis.signal.expected_return > 0 else 0xe74c3c,
        )
        embed.add_field(name="Company State", value=company_state, inline=True)
        embed.add_field(name="Setup Type", value=setup_type, inline=True)
        embed.add_field(name="Asymmetry", value=asymmetry_label, inline=True)
        special_situation_label = analysis.metadata.get("special_situation_label")
        if special_situation_label:
            embed.add_field(name="Special Situation", value=str(special_situation_label).replace("_", " "), inline=True)
        embed.add_field(name="Action", value=action_label, inline=True)
        embed.add_field(name="Confidence", value=f"{round(analysis.signal.confidence * 100, 1)}%", inline=True)
        embed.add_field(name="Target Weight", value=target_weight_text, inline=True)
        embed.add_field(name="Scenario", value=analysis.portfolio.scenario, inline=True)

        embed.add_field(name="Expected Return (90d)", value=f"{analysis.signal.expected_return * 100:+.1f}%", inline=True)
        embed.add_field(name="Catalyst Success", value=f"{analysis.signal.catalyst_success_prob * 100:.1f}%", inline=True)
        embed.add_field(name="Financing Risk", value=f"{analysis.signal.financing_risk:.2f}", inline=True)

        if primary_event is not None:
            timing_confidence = {
                "calendar_estimate": "medium",
                "estimated_from_revenue": "medium",
                "phase_timing_estimate": "low",
            }.get(primary_event.status, "medium")
            embed.add_field(
                name="Primary Event",
                value=(
                    f"{primary_event.title}\n"
                    f"Type: `{primary_event.event_type}` | Date: `{primary_event.expected_date or 'TBD'}`\n"
                    f"Horizon: `{primary_event.horizon_days}d` | Timing confidence: `{timing_confidence}`"
                ),
                inline=False,
            )

        embed.add_field(
            name="Why Now",
            value=analysis.metadata.get("why_now", "Why-now context is still loading."),
            inline=False,
        )

        rationale = "\n".join(f"• {line}" for line in analysis.signal.rationale[:4]) or "No rationale available."
        embed.add_field(name="Core Thesis", value=rationale, inline=False)

        if snapshot.approved_products:
            lead_product = snapshot.approved_products[0]
            embed.add_field(
                name="Lead Franchise",
                value=(
                    f"{lead_product.name} | {lead_product.indication}\n"
                    "Commercial-stage product recognized from the approved-product registry."
                ),
                inline=False,
            )
        elif lead_program_item is not None:
            lead_condition = lead_program_item.conditions[0] if lead_program_item.conditions else "unspecified indication"
            lead_catalyst = lead_program_item.catalyst_events[0] if lead_program_item.catalyst_events else None
            lead_date = lead_catalyst.expected_date if lead_catalyst else "TBD"
            embed.add_field(
                name="Lead Program",
                value=(
                    f"{lead_program_item.name} | {lead_program_item.phase} | {lead_condition}\n"
                    f"Lead event: `{lead_catalyst.event_type if lead_catalyst else 'none'}` on `{lead_date}`"
                ),
                inline=False,
            )

        embed.add_field(
            name="Balance Sheet",
            value=(
                f"Mkt Cap: {format_money(snapshot.market_cap)} | EV: {format_money(snapshot.enterprise_value)}\n"
                f"Revenue: {format_money(snapshot.revenue)} | Cash: {format_money(snapshot.cash)} | Net Cash: {format_money(net_cash)}\n"
                f"Runway: {runway_label(snapshot)}"
            ),
            inline=False,
        )

        embed.add_field(
            name="Commercial Reality",
            value=(
                f"{commercial_truth}\n"
                f"Source: `{source}` | As of: `{snapshot.as_of[:10]}`"
                + (f" | Age: `{int(snapshot_age_days)}d`" if snapshot_age_days is not None else "")
            ),
            inline=False,
        )

        if source != "live" or (snapshot_age_days is not None and float(snapshot_age_days) > 14):
            embed.add_field(
                name="Data Freshness",
                value=(
                    "This tear sheet is not using a current live snapshot. Treat it as informational only until a fresh refresh succeeds."
                ),
                inline=False,
            )

        embed.add_field(
            name="Market Setup",
            value=analysis.metadata.get("expectations_summary", "Expectations context unavailable."),
            inline=False,
        )

        internal_value_text = (
            f"Peer-anchored value: {format_money(internal_value)}"
            if internal_value is not None
            else "Peer-anchored value: unavailable"
        )
        if internal_price_target is not None:
            internal_value_text += f" | PT: ${float(internal_price_target):.2f}"
        if internal_upside_pct is not None:
            internal_value_text += f" | Gap: {float(internal_upside_pct) * 100:+.1f}%"
        value_method = analysis.metadata.get("value_method")
        if value_method:
            internal_value_text += f"\nMethod: {value_method}"
        embed.add_field(name="Tatetuck Value View", value=internal_value_text, inline=False)

        valuation_summary = analysis.metadata.get("valuation_summary", "Valuation context unavailable.")
        peer_tickers = analysis.metadata.get("peer_tickers") or []
        if peer_tickers:
            valuation_summary = f"{valuation_summary}\nClosest archived peers: {', '.join(peer_tickers)}"
        embed.add_field(name="Valuation Lens", value=valuation_summary, inline=False)

        state_focus_name = {
            "pre_commercial": "Pre-Commercial Lens",
            "commercial_launch": "Launch Lens",
            "commercialized": "Franchise Lens",
        }.get(analysis.metadata.get("company_state"), "PM Lens")
        state_focus_lines = [
            analysis.metadata.get("state_focus", "State-specific PM framing unavailable."),
            analysis.metadata.get("competitive_summary", "Competitive-landscape summary unavailable."),
        ]
        differentiation_focus = analysis.metadata.get("differentiation_focus")
        if differentiation_focus:
            state_focus_lines.append(f"Differentiation bar: {differentiation_focus}")
        embed.add_field(name=state_focus_name, value="\n".join(state_focus_lines), inline=False)

        market_view = analysis.metadata.get("market_view")
        asymmetry_summary = analysis.metadata.get("asymmetry_summary")
        if market_view or asymmetry_summary:
            embed.add_field(
                name="Expectation Gap",
                value="\n".join(
                    item
                    for item in [market_view, asymmetry_summary]
                    if item
                ),
                inline=False,
            )

        kill_points = analysis.metadata.get("kill_points") or []
        embed.add_field(
            name="What Breaks The Short" if action_label == "short" else "What Breaks The Thesis",
            value="\n".join(f"• {item}" for item in kill_points[:3]),
            inline=False,
        )

        if analysis.portfolio.risk_flags:
            embed.add_field(name="Risk Flags", value=", ".join(analysis.portfolio.risk_flags), inline=False)

        embed.set_footer(text="Tatetuck Bot vNext - Event-Driven Biopharma Alpha Platform")
        return embed

    async def generate_top5_embed():
        top_ideas, readiness = await build_pm_top_ideas()
        top_picks = top_ideas[:5]
        if not top_picks:
            return None

        long_count = sum(1 for analysis in top_ideas if idea_action_label(analysis) == "long")
        short_count = sum(1 for analysis in top_ideas if idea_action_label(analysis) == "short")
        embed = discord.Embed(
            title="Morning Biotech Brief: Top Biotech Ideas",
            description="Highest-conviction long and short ideas from the latest research scan.",
            color=0x3498DB,
        )
        embed.add_field(
            name="Scan Context",
            value=(
                f"Readiness: `{readiness.status}` | "
                f"Longs: `{long_count}` | "
                f"Shorts: `{short_count}` | "
                f"Selected: `{len(top_picks)}`"
            ),
            inline=False,
        )
        for index, analysis in enumerate(top_picks, 1):
            primary_event = primary_catalyst(analysis.snapshot, analysis.signal.primary_event_type)
            profile = brief_pick_profile(analysis)
            action_label = idea_action_label(analysis)
            event_text = (
                brief_event_text(analysis.snapshot, analysis.signal.primary_event_type)
                if primary_event is not None
                else "no clean upcoming catalyst"
            )
            if event_text == "franchise setup, no clean dated catalyst":
                event_line = (
                    f"**Driver**: {brief_setup_driver_text(analysis.snapshot, analysis.signal.primary_event_type)}"
                    " | no clean dated catalyst"
                )
            else:
                event_line = f"**Event**: {event_text}"
            state_line = f"**State**: {(analysis.metadata.get('company_state') or 'unknown').replace('_', ' ')} | **Setup**: {(analysis.metadata.get('setup_type') or 'watchful').replace('_', ' ')}"
            brief_line = f"**Brief Type**: {profile.replace('_', ' ')}"
            source = str(analysis.metadata.get("analysis_source") or "unknown").replace("_", " ")
            snapshot_age_days = analysis.metadata.get("snapshot_age_days")
            freshness_line = (
                f"**Data**: {source} | **Age**: {int(snapshot_age_days)}d"
                if snapshot_age_days is not None
                else f"**Data**: {source}"
            )
            embed.add_field(
                name=f"#{index} — {analysis.snapshot.ticker}",
                value=(
                    f"**Action**: {action_label} | **Idea Profile**: {profile.replace('_', ' ')}\n"
                    f"**Confidence**: {analysis.signal.confidence * 100:.1f}% | "
                    f"**Target Weight**: {analysis.portfolio.target_weight:.1f}% {action_label}\n"
                    f"**Expected Return**: {analysis.signal.expected_return * 100:+.1f}% | "
                    f"**Scenario**: {analysis.portfolio.scenario}\n"
                    f"{state_line}\n"
                    f"{freshness_line}\n"
                    f"{event_line}"
                ),
                inline=False,
            )
        embed.set_footer(text="Tatetuck Bot vNext")
        return embed

    @bot.event
    async def on_ready():
        print(f"✅ Tatetuck Bot ({bot.user}) is online.")
        await bot.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="biotech catalysts",
            )
        )
        if not morning_briefing.is_running():
            morning_briefing.start()

    @bot.command(name="analyze")
    async def analyze(ctx, ticker: str | None = None):
        if not ticker:
            await ctx.send("Please provide a ticker. Example: `!analyze CRSP`")
            return

        ticker = ticker.upper()
        all_companies = {item[0]: item[1] for item in TRAIN_TICKERS + HOLDOUT_TICKERS}
        company_name = all_companies.get(ticker, ticker)
        status_msg = await ctx.send(
            f"🔬 Running Tatetuck analysis for **{ticker}** with a fresh live snapshot when available..."
        )
        try:
            analysis = await analyze_one(ticker, company_name)
            await status_msg.edit(content=None, embed=build_tearsheet_embed(analysis))
        except Exception as exc:
            await status_msg.edit(content=f"❌ Analysis failed for `{ticker}`: {type(exc).__name__}: {exc}")

    @bot.command(name="top5")
    async def top5(ctx):
        status_msg = await ctx.send("🚀 Running the PM deployment scan. This usually takes a few seconds...")
        try:
            embed = await generate_top5_embed()
        except Exception as exc:
            await status_msg.edit(content=f"❌ PM deployment scan failed: {type(exc).__name__}: {exc}")
            return
        if embed is None:
            await status_msg.edit(content="❌ No ranked ideas were available.")
            return
        await status_msg.edit(content=None, embed=embed)

    @bot.command(name="guide")
    async def guide(ctx):
        await ctx.send(embed=build_guide_embed())

    @bot.command(name="channelid")
    async def channelid(ctx):
        await ctx.send(f"Current channel ID: `{ctx.channel.id}`")

    @bot.command(name="setup")
    async def setup(ctx):
        status_msg = await ctx.send("Setting up the Tatetuck channel guide...")
        try:
            _message, pin_status = await send_and_optionally_pin_guide(ctx.channel)
        except Exception as exc:
            await status_msg.edit(content=f"❌ Setup failed: {type(exc).__name__}: {exc}")
            return

        await status_msg.edit(
            content=(
                f"✅ Tatetuck setup {pin_status} in <#{ctx.channel.id}>.\n"
                f"Channel ID: `{ctx.channel.id}`\n"
                "If you want automated morning posts here, set `DISCORD_CHANNEL_ID` to that value."
            )
        )

    @bot.command(name="status")
    async def status(ctx):
        loop = asyncio.get_running_loop()
        readiness = await loop.run_in_executor(
            None,
            partial(build_readiness_report, store=store, settings=settings),
        )
        embed = discord.Embed(
            title="Tatetuck Bot Status",
            description="Current research-engine and Discord-surface health.",
            color=0x1ABC9C if readiness.status == "production_ready" else 0xF39C12,
        )
        embed.add_field(name="Research Status", value=readiness.status, inline=True)
        embed.add_field(name="Walk-Forward Windows", value=str(readiness.walkforward_windows), inline=True)
        embed.add_field(name="Latest Snapshot Dates", value=str(readiness.distinct_snapshot_dates), inline=True)
        embed.add_field(name="Matured 90d Labels", value=str(readiness.matured_return_90d_rows), inline=True)
        embed.add_field(name="Evaluate Runs", value=f"{readiness.successful_evaluate_runs}/{readiness.evaluate_run_count}", inline=True)
        embed.add_field(name="Leakage Audit", value="pass" if readiness.leakage_passed else "fail", inline=True)
        if readiness.blockers:
            embed.add_field(name="Blockers", value="\n".join(f"• {item}" for item in readiness.blockers[:5]), inline=False)
        else:
            embed.add_field(name="Blockers", value="None", inline=False)
        if readiness.warnings:
            embed.add_field(name="Warnings", value="\n".join(f"• {item}" for item in readiness.warnings[:5]), inline=False)
        embed.set_footer(text="Use !guide if you want the plain-English explanation.")
        await ctx.send(embed=embed)

    @bot.command(name="help")
    async def custom_help(ctx):
        embed = discord.Embed(
            title="Tatetuck Bot Commands",
            description="Event-driven biotech research in Discord.",
            color=0x2ECC71,
        )
        embed.add_field(name="`!analyze TICKER`", value="Build a tear sheet for one biotech name.", inline=False)
        embed.add_field(name="`!top5`", value="Show the current best long and short ideas from the research scan.", inline=False)
        embed.add_field(name="`!guide`", value="Show the layman bot guide and how to read the outputs.", inline=False)
        embed.add_field(name="`!setup`", value="Post the guide in the current channel, try to pin it, and show the channel ID.", inline=False)
        embed.add_field(name="`!channelid`", value="Show the current Discord channel ID.", inline=False)
        embed.add_field(name="`!status`", value="Show research-engine health and readiness.", inline=False)
        await ctx.send(embed=embed)

    @tasks.loop(time=time(hour=13, minute=30, tzinfo=timezone.utc))
    async def morning_briefing():
        try:
            embed = await generate_top5_embed()
        except Exception as exc:
            print(f"ERROR: morning briefing failed: {type(exc).__name__}: {exc}")
            return
        if embed is None:
            print("ERROR: morning briefing produced no ranked ideas.")
            return

        channel = resolve_channel()
        if channel is None:
            print("ERROR: no writable Discord channel found for the morning briefing.")
            return
        await channel.send("🔔 **Automated Tatetuck morning scan complete**", embed=embed)

    return bot


def main() -> int:
    args = parse_args()
    if args.self_check:
        return run_self_check()

    token = get_discord_token()
    if not token:
        print("CRITICAL ERROR: DISCORD_TOKEN is not set in .env")
        print("Add DISCORD_TOKEN and DISCORD_CHANNEL_ID, then rerun the bot.")
        return 1

    bot = build_bot()
    bot.run(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
