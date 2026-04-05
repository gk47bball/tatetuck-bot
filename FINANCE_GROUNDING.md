# Finance Grounding Backbone

Tatetuck should not reason about markets as a pure biotech heuristic engine.
It must stay grounded in real market data, peer behavior, and finance-native
comparisons.

## Installed Agent Skills

The local Codex environment now has these finance skills installed from
[`himself65/finance-skills`](https://github.com/himself65/finance-skills):

- `yfinance-data`
- `stock-correlation`

These skills are agent-side capabilities, not runtime Python modules inside the
repo. They should be treated as part of Tatetuck's operating backbone for
research, validation, and PM-facing outputs.

## Operating Rules

1. Use `yfinance-data` whenever Tatetuck work depends on real market or company
   data from Yahoo Finance:
   - price history
   - OHLCV
   - financial statements
   - earnings data
   - analyst targets / recommendations
   - options chains
   - holders / insider context

2. Use `stock-correlation` whenever Tatetuck needs:
   - peer discovery
   - pair candidates
   - beta / co-movement context
   - sector clustering
   - correlation sanity checks for a thesis

3. Never present a PM-facing “price target” or “market setup” as if it were
   Street-grounded unless it has been cross-checked against actual market data
   or analyst data.

4. Never treat “catalyst success” and “stock went up” as interchangeable.
   When the label is market-reaction-based, say so explicitly.

5. Never use static peer lists when a dynamic market-based peer or correlation
   process is available.

## What This Means For Tatetuck

- [market_profile.py](/Users/gkornblatt/Desktop/TATETUCK_BOT/biopharma_agent/vnext/market_profile.py)
  should be treated as Tatetuck's internal lens, not the final source of market
  truth.
- [universe.py](/Users/gkornblatt/Desktop/TATETUCK_BOT/biopharma_agent/vnext/universe.py)
  and [evaluation.py](/Users/gkornblatt/Desktop/TATETUCK_BOT/biopharma_agent/vnext/evaluation.py)
  should increasingly be validated against externally grounded peer and market
  context.
- [discord_bot.py](/Users/gkornblatt/Desktop/TATETUCK_BOT/discord_bot.py)
  and [trade_vnext.py](/Users/gkornblatt/Desktop/TATETUCK_BOT/trade_vnext.py)
  should present market framing conservatively unless the claim is tied to real
  fetched finance data.

## Near-Term Follow-Ons

- Add a market-reality layer that fetches Yahoo Finance quote, analyst-target,
  and peer-correlation context before PM-facing outputs are finalized.
- Use correlation-based peer sets instead of only archived-universe stage peers
  when discussing relative valuation and pair ideas.
- Add a PM sanity-check step that flags when Tatetuck's internal view differs
  materially from market-implied or peer-relative behavior.
