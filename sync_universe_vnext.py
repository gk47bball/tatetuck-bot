"""
sync_universe_vnext.py — sync active and delisted US listings from EODHD into
the local universe-membership table for survivorship-aware validation.
"""

from __future__ import annotations

import argparse

from biopharma_agent.vnext.eodhd import EODHDUniverseClient
from biopharma_agent.vnext.settings import VNextSettings
from biopharma_agent.vnext.storage import LocalResearchStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync active and delisted exchange symbol lists from EODHD.")
    parser.add_argument(
        "--exchanges",
        default="NASDAQ,NYSE,AMEX",
        help="Comma-separated EODHD exchange codes to sync.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = VNextSettings.from_env()
    store = LocalResearchStore(settings.store_dir)
    client = EODHDUniverseClient(store=store, api_key=settings.eodhd_api_key)
    exchanges = tuple(item.strip().upper() for item in args.exchanges.split(",") if item.strip())
    summary = client.sync_universe_membership(exchanges=exchanges)

    print("=" * 72)
    print("  TATETUCK BOT vNEXT — UNIVERSE SYNC")
    print("=" * 72)
    print(f"exchanges_requested: {summary.exchanges_requested}")
    print(f"active_rows:         {summary.active_rows}")
    print(f"delisted_rows:       {summary.delisted_rows}")
    print(f"total_rows:          {summary.total_rows}")


if __name__ == "__main__":
    main()
