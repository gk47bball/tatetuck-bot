# Tatetuck Bot Guide

## What The Bot Does

Tatetuck Bot is a biotech research bot built for catalyst-driven investing.
It looks at each company’s drug programs, approved products, cash runway, and
upcoming events, then turns that into a simple ranking of which setups look
most attractive over the next 1-6 months.

## How To Think About It

The easiest way to think about the bot is this:

- Every biotech company is really a bundle of bets.
- Some bets are clinical readouts.
- Some bets are regulatory decisions.
- Some bets are commercial updates.
- Some are weak because the company may need to raise cash first.

The bot tries to sort through that and answer:

- Which event matters most?
- How promising is it?
- How big is the upside if it works?
- How dangerous is dilution or balance-sheet stress?

## What The Main Outputs Mean

- `Confidence`
  The overall strength of the setup.

- `Expected Return (90d)`
  The model’s directional view over the next roughly three months.

- `Catalyst Success`
  The estimated chance that the next important catalyst lands favorably.

- `Target Weight`
  A paper-portfolio sizing suggestion. Bigger does not mean certain; it means
  the bot thinks the risk/reward is better relative to other names.

- `Scenario`
  A simple label for how the bot sees the setup.
  Examples: `pre-catalyst long`, `pairs candidate`, `commercial compounder`.

- `Financing Risk`
  A warning signal that a company may need to raise capital, which can hurt the
  stock even if the science is good.

## What The Bot Is Not

- It is not a guarantee.
- It is not a replacement for reading primary source data.
- It is not pure “AI magic.”

It is best used as a fast, structured research assistant that helps surface the
strongest biotech setups and explain why they rank where they do.

## Discord Commands

- `!analyze TICKER`
  Build a tear sheet for one company.

- `!top5`
  Show the current top-ranked benchmark ideas.

- `!guide`
  Show the plain-English version of how the bot works.

- `!setup`
  Post the guide in the current Discord channel, try to pin it, and show the
  channel ID for configuration.

- `!channelid`
  Show the current Discord channel ID.

- `!status`
  Show research-engine readiness and health.
