from __future__ import annotations

import argparse
import asyncio
import os
from datetime import time, timezone
from functools import partial
from pathlib import Path

import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv

from prepare import HOLDOUT_TICKERS, TRAIN_TICKERS

from biopharma_agent.vnext import TatetuckPlatform, VNextSettings, build_readiness_report
from biopharma_agent.vnext.storage import LocalResearchStore


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
    "`Target Weight` is a paper-portfolio sizing suggestion, not a guarantee."
)

GUIDE_COMMANDS = (
    "`!analyze TICKER` for one tear sheet, `!top5` for the current best ideas, "
    "`!guide` for the layman version, and `!status` for bot + research health."
)


def get_discord_token() -> str | None:
    return os.getenv("DISCORD_TOKEN") or os.getenv("DISCORD_BOT_TOKEN")


def get_target_channel_id() -> str | None:
    return os.getenv("DISCORD_CHANNEL_ID") or os.getenv("TATETUCK_DISCORD_CHANNEL_ID")


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
            include_literature=False,
            prefer_archive=True,
            fallback_to_archive=True,
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
                include_literature=False,
                prefer_archive=True,
                fallback_to_archive=True,
            ),
        )

    async def analyze_universe():
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(
                platform.analyze_universe,
                TRAIN_TICKERS + HOLDOUT_TICKERS,
                include_literature=False,
                prefer_archive=True,
                fallback_to_archive=True,
            ),
        )

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

    def build_tearsheet_embed(analysis) -> discord.Embed:
        embed = discord.Embed(
            title=f"Tatetuck Analyst Tear Sheet: {analysis.snapshot.ticker}",
            description="Event-driven biopharma alpha profile",
            color=0x2ecc71 if analysis.signal.expected_return > 0 else 0xe74c3c,
        )
        embed.add_field(name="Confidence", value=f"{round(analysis.signal.confidence * 100, 1)}%", inline=True)
        embed.add_field(name="Target Weight", value=f"{analysis.portfolio.target_weight}%", inline=True)
        embed.add_field(name="Scenario", value=analysis.portfolio.scenario, inline=True)

        embed.add_field(name="Expected Return (90d)", value=f"{analysis.signal.expected_return * 100:+.1f}%", inline=True)
        embed.add_field(name="Catalyst Success", value=f"{analysis.signal.catalyst_success_prob * 100:.1f}%", inline=True)
        embed.add_field(name="Financing Risk", value=f"{analysis.signal.financing_risk:.2f}", inline=True)

        embed.add_field(
            name="How To Read This",
            value="Higher confidence + higher target weight usually means the bot sees a cleaner catalyst setup with less financing risk.",
            inline=False,
        )

        rationale = "\n".join(f"• {line}" for line in analysis.signal.rationale[:4]) or "No rationale available."
        embed.add_field(name="Core Thesis", value=rationale, inline=False)

        if analysis.snapshot.programs:
            program_lines = []
            for program in analysis.snapshot.programs[:3]:
                catalyst = program.catalyst_events[0] if program.catalyst_events else None
                horizon = catalyst.horizon_days if catalyst else 180
                program_lines.append(f"**{program.name}** | {program.phase} | {horizon}d catalyst")
            embed.add_field(name="Program Dashboard", value="\n".join(program_lines), inline=False)

        if analysis.portfolio.risk_flags:
            embed.add_field(name="Risk Flags", value=", ".join(analysis.portfolio.risk_flags), inline=False)

        embed.set_footer(text="Tatetuck Bot vNext - Event-Driven Biopharma Alpha Platform")
        return embed

    async def generate_top5_embed():
        analyses = await analyze_universe()
        ranked = sorted(
            analyses,
            key=lambda item: (item.portfolio.target_weight, item.signal.confidence, item.signal.expected_return),
            reverse=True,
        )
        top_picks = ranked[:5]
        if not top_picks:
            return None

        embed = discord.Embed(
            title="Morning Biotech Brief: Top 5 Alpha Signals",
            description="Highest ranked names from the Tatetuck event-driven benchmark scan.",
            color=0x3498DB,
        )
        for index, analysis in enumerate(top_picks, 1):
            embed.add_field(
                name=f"#{index} — {analysis.snapshot.ticker}",
                value=(
                    f"**Confidence**: {analysis.signal.confidence * 100:.1f}% | "
                    f"**Target Weight**: {analysis.portfolio.target_weight:.1f}%\n"
                    f"**Expected Return**: {analysis.signal.expected_return * 100:+.1f}% | "
                    f"**Scenario**: {analysis.portfolio.scenario}"
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
            f"🔬 Running Tatetuck analysis for **{ticker}** using the archive-first biotech catalyst engine..."
        )
        try:
            analysis = await analyze_one(ticker, company_name)
            await status_msg.edit(content=None, embed=build_tearsheet_embed(analysis))
        except Exception as exc:
            await status_msg.edit(content=f"❌ Analysis failed for `{ticker}`: {type(exc).__name__}: {exc}")

    @bot.command(name="top5")
    async def top5(ctx):
        status_msg = await ctx.send("🚀 Running the benchmark scan. This usually takes a few seconds...")
        try:
            embed = await generate_top5_embed()
        except Exception as exc:
            await status_msg.edit(content=f"❌ Benchmark scan failed: {type(exc).__name__}: {exc}")
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
        embed.add_field(name="`!top5`", value="Show the current top benchmark ideas.", inline=False)
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
