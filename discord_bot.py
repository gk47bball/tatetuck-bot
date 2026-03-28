import os
import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv
import asyncio
from datetime import time, timezone

# Import the core biopharma engine
from prepare import TRAIN_TICKERS, HOLDOUT_TICKERS
from biopharma_agent.vnext import TatetuckPlatform

# Load environment variables
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
TARGET_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID")

if not TOKEN or TOKEN == "your_discord_bot_token_here":
    print("CRITICAL ERROR: DISCORD_TOKEN is not set in .env")
    print("Please add your token to .env and restart.")
    exit(1)

# Initialize bot with message intents
intents = discord.Intents.default()
intents.message_content = True

# We use the '!' prefix for commands
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)
platform = TatetuckPlatform()

@bot.event
async def on_ready():
    print(f"✅ Tatetuck Analyst ({bot.user}) is online and ready for scans.")
    # Set custom status
    await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name="Phase 3 Trials"))
    
    # Start the daily scheduled task if it's not already running
    if not morning_briefing.is_running():
        morning_briefing.start()

@bot.command(name="analyze")
async def analyze(ctx, ticker: str = None):
    """
    Analyzes a single biopharma ticker and returns a detailed Tear Sheet.
    Usage: !analyze CRSP
    """
    if not ticker:
        await ctx.send("⚠️ Please provide a ticker symbol. Example: `!analyze CRSP`")
        return

    ticker = ticker.upper()
    
    # Try to find the actual company name from our lists
    all_dict = {t[0]: t[1] for t in TRAIN_TICKERS + HOLDOUT_TICKERS}
    company_name = all_dict.get(ticker, ticker)
    
    status_msg = await ctx.send(f"🔬 **Initiating Tatetuck vNext analysis for {ticker}**...\nBuilding the company-program-catalyst graph and scoring the thesis.")

    # Run data gathering in a separate thread so we don't block the async event loop
    loop = asyncio.get_event_loop()
    try:
        analysis = await loop.run_in_executor(None, platform.analyze_ticker, ticker, company_name, False)

        # Assemble the Tear Sheet Embed
        embed = discord.Embed(
            title=f"Tatetuck Analyst Tear Sheet: {ticker}",
            description="Event-driven biopharma alpha profile",
            color=0x2ecc71 if analysis.signal.expected_return > 0 else 0xe74c3c
        )

        embed.add_field(name="🧬 Confidence", value=f"{round(analysis.signal.confidence * 100, 1)}%", inline=True)
        embed.add_field(name="📊 Target Weight", value=f"{analysis.portfolio.target_weight}%", inline=True)
        embed.add_field(name="🛡️ Scenario", value=analysis.portfolio.scenario, inline=True)

        embed.add_field(name="🎯 Expected Return (90d)", value=f"{analysis.signal.expected_return * 100:+.1f}%", inline=True)
        embed.add_field(name="📅 Catalyst Success", value=f"{analysis.signal.catalyst_success_prob * 100:.1f}%", inline=True)
        embed.add_field(name="⚠️ Financing Risk", value=f"{analysis.signal.financing_risk:.2f}", inline=True)

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

        await status_msg.edit(content=None, embed=embed)

    except Exception as e:
        await status_msg.edit(content=f"❌ **System Error analyzing {ticker}**: {str(e)}")

async def generate_top5_embed():
    """Shared helper to generate the Top 5 scan embed."""
    loop = asyncio.get_event_loop()
    all_tickers = TRAIN_TICKERS + HOLDOUT_TICKERS
    
    results = []
    
    # We will gather and score all tickers
    for ticker, company_name in all_tickers:
        try:
            analysis = await loop.run_in_executor(None, platform.analyze_ticker, ticker, company_name, False)
            results.append({
                "ticker": ticker,
                "confidence": analysis.signal.confidence,
                "allocation": analysis.portfolio.target_weight,
                "expected_return": analysis.signal.expected_return,
                "scenario": analysis.portfolio.scenario,
            })
        except Exception:
            pass # Skip failed tickers during bulk scan
            
    if not results:
        return None
        
    # Sort by highest target weight
    results.sort(key=lambda x: x["allocation"], reverse=True)
    top_picks = results[:5]
    
    embed = discord.Embed(
        title="☀️ Morning Biotech Brief: Top 5 Alpha Signals",
        description="Highest ranked targets across the benchmark based on vNext event-driven position sizing.",
        color=0x3498db
    )
    
    for i, pick in enumerate(top_picks, 1):
        # Prevent TypeError if values are missing
        conv_val = float(pick['confidence']) * 100 if pick['confidence'] is not None else 0.0
        alloc_val = pick['allocation'] if pick['allocation'] is not None else 0.0
        
        embed.add_field(
            name=f"#{i} — {pick['ticker']}",
            value=(
                f"**Confidence**: {round(conv_val, 1)}% | **Target Allocation**: {alloc_val}%\n"
                f"**Expected Return**: {pick['expected_return'] * 100:+.1f}% | **Scenario**: {pick['scenario']}"
            ),
            inline=False
        )
        
    embed.set_footer(text="Tatetuck AutoResearch - vNext Event-Driven Platform")
    return embed

@bot.command(name="top5")
async def top5(ctx):
    """
    Scans the benchmark universe and returns the Top 5 highest conviction trades.
    """
    status_msg = await ctx.send("🚀 **Initiating Benchmark Scan** (This may take ~30-60 seconds)...")
    
    embed = await generate_top5_embed()
    
    if embed:
        await status_msg.edit(content=None, embed=embed)
    else:
        await status_msg.edit(content="❌ **Scan Failed**: Unable to retrieve any benchmark data.")

# Scheduled Daily Task: 9:00 AM EST is roughly 13:00 UTC (14:00 UTC during standard time, using 13:30 UTC generic)
@tasks.loop(time=time(hour=13, minute=30, tzinfo=timezone.utc))
async def morning_briefing():
    print("Executing automated morning briefing...")
    embed = await generate_top5_embed()
    if embed:
        channel = None
        if TARGET_CHANNEL_ID:
            try:
                channel = bot.get_channel(int(TARGET_CHANNEL_ID))
            except ValueError:
                pass
        
        # Fallback to the first text channel the bot can write to if ID is missing or invalid
        if not channel:
            for guild in bot.guilds:
                for t_channel in guild.text_channels:
                    if t_channel.permissions_for(guild.me).send_messages:
                        channel = t_channel
                        break
                if channel:
                    break
                    
        if channel:
            await channel.send("🔔 **Automated Market Scan Complete**", embed=embed)
        else:
            print("ERROR: Could not find a valid channel to broadcast the morning briefing.")

@bot.command(name="help")
async def custom_help(ctx):
    embed = discord.Embed(
        title="Tatetuck Analyst Commands",
        description="I am your event-driven quantitative biopharma research engine.",
        color=0x9b59b6
    )
    embed.add_field(name="`!analyze TICKER`", value="Builds the company-program-catalyst graph for one biotech ticker and returns its event-driven tear sheet.", inline=False)
    embed.add_field(name="`!top5`", value="Scans the benchmark universe and returns the best current event-driven ideas from the vNext platform.", inline=False)
    await ctx.send(embed=embed)

if __name__ == "__main__":
    bot.run(TOKEN)
