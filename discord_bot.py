import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
import asyncio

# Import the core biopharma engine
from prepare import gather_company_data, TRAIN_TICKERS, HOLDOUT_TICKERS
from strategy import score_company

# Load environment variables
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

if not TOKEN or TOKEN == "your_discord_bot_token_here":
    print("CRITICAL ERROR: DISCORD_TOKEN is not set in .env")
    print("Please add your token to .env and restart.")
    exit(1)

# Initialize bot with message intents
intents = discord.Intents.default()
intents.message_content = True

# We use the '!' prefix for commands
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

@bot.event
async def on_ready():
    print(f"✅ Tatetuck Analyst ({bot.user}) is online and ready for scans.")
    # Set custom status
    await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name="Phase 3 Trials"))

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
    
    status_msg = await ctx.send(f"🔬 **Initiating Alpha Stack Analysis for {ticker}**...\nGathering SEC, FDA, and PubMed data.")

    # Run data gathering in a separate thread so we don't block the async event loop
    loop = asyncio.get_event_loop()
    try:
        data = await loop.run_in_executor(None, gather_company_data, ticker, company_name)
        
        if not data or data.get("error"):
            await status_msg.edit(content=f"❌ **Analysis Failed for {ticker}**: {data.get('error', 'No data found.')}")
            return

        # Score the company using Alpha Stack v13
        score = score_company(data)

        # Assemble the Tear Sheet Embed
        embed = discord.Embed(
            title=f"Tatetuck Analyst Tear Sheet: {ticker}",
            description="Alpha Stack v13 Breakthrough Profile",
            color=0x2ecc71 if score["signal"] > 0 else 0xe74c3c
        )

        embed.add_field(name="🧬 Base Conviction", value=f"{round(score['conviction_weight'] * 100, 1)}%", inline=True)
        embed.add_field(name="📊 Recommended Allocation", value=f"{score['recommended_allocation']}%", inline=True)
        embed.add_field(name="🛡️ Risk-Parity Allocation", value=f"{score['risk_parity_allocation']}%", inline=True)

        embed.add_field(name="🔬 Fundamental DCF/rNPV", value=f"${score['rnpv']:,.0f}", inline=False)

        # Break down the signals
        brk = score.get("alpha_breakdown", {})
        
        def format_alpha(val):
            return f"{round(val, 3)} {'📈' if val > 0 else '📉'}"

        signal_str = (
            f"**Value (rNPV/MC)**: {format_alpha(brk.get('value', 0))}\n"
            f"**Clinical Momentum**: {format_alpha(brk.get('clinical', 0))}\n"
            f"**FDA Safety Risk**: {format_alpha(brk.get('safety', 0))}\n"
            f"**Financial Health**: {format_alpha(brk.get('financial', 0))}\n"
            f"**Macro Autocorrelation**: {format_alpha(brk.get('momentum', 0))}"
        )

        embed.add_field(name="Core Alpha Sub-Signals", value=signal_str, inline=False)
        embed.set_footer(text="Tatetuck Bot - Biopharma Quant Engine")

        await status_msg.edit(content=None, embed=embed)

    except Exception as e:
        await status_msg.edit(content=f"❌ **System Error analyzing {ticker}**: {str(e)}")

@bot.command(name="top5")
async def top5(ctx):
    """
    Scans the benchmark universe and returns the Top 5 highest conviction trades.
    """
    status_msg = await ctx.send("🚀 **Initiating Benchmark Scan** (This may take ~30-60 seconds)...")

    loop = asyncio.get_event_loop()
    all_tickers = TRAIN_TICKERS + HOLDOUT_TICKERS
    
    results = []
    
    # We will gather and score all tickers
    for ticker, company_name in all_tickers:
        try:
            data = await loop.run_in_executor(None, gather_company_data, ticker, company_name)
            if data and not data.get("error"):
                score = score_company(data)
                
                results.append({
                    "ticker": ticker,
                    "conviction": score["conviction_weight"],
                    "allocation": score["risk_parity_allocation"],
                    "rnpv": score["rnpv"]
                })
        except Exception:
            pass # Skip failed tickers during bulk scan
            
    if not results:
        await status_msg.edit(content="❌ **Scan Failed**: Unable to retrieve any benchmark data.")
        return
        
    # Sort by highest conviction_weight
    results.sort(key=lambda x: x["conviction"], reverse=True)
    top_picks = results[:5]
    
    embed = discord.Embed(
        title="Morning Biotech Brief: Top 5 Conviction Profiles",
        description="Highest ranked targets across the 41-ticker benchmark based on risk-parity allocations.",
        color=0x3498db
    )
    
    for i, pick in enumerate(top_picks, 1):
        embed.add_field(
            name=f"#{i} — {pick['ticker']}",
            value=f"**Conviction**: {round(pick['conviction'] * 100, 1)}% | **Target Allocation**: {pick['allocation']}%\n**Est. rNPV**: ${pick['rnpv']:,.0f}",
            inline=False
        )
        
    embed.set_footer(text="Tatetuck Bot - Alpha Stack v13")
    await status_msg.edit(content=None, embed=embed)

@bot.command(name="help")
async def custom_help(ctx):
    embed = discord.Embed(
        title="Tatetuck Analyst Commands",
        description="I am your quantitative biopharma research engine.",
        color=0x9b59b6
    )
    embed.add_field(name="`!analyze TICKER`", value="Runs the Alpha Stack across a single biotech ticker and returns its tear sheet.", inline=False)
    embed.add_field(name="`!top5`", value="Scans the 41-ticker benchmark and returns the absolute best 5 risk-parity configurations.", inline=False)
    await ctx.send(embed=embed)

if __name__ == "__main__":
    bot.run(TOKEN)
