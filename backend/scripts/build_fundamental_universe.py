"""Offline batch script — builds the fundamental reference universes.

The universes are the cross-sectional reference (yields, ratios, growth metrics per company) the
fundamental module normalizes each ticker against. They are committed as frozen seed CSVs in
engine/data/ and refreshed weekly by the scheduler; run this manually to regenerate the seed.

The MSCI World build needs engine/data/URTH_holdings.csv (the official iShares holdings export).

Usage (from backend/):
    uv run python scripts/build_fundamental_universe.py             # both universes
    uv run python scripts/build_fundamental_universe.py --only sp500
    uv run python scripts/build_fundamental_universe.py --only msci_world
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure backend/ is on sys.path so `app` is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from app.services.fundamental.engine.msci_world import (  # noqa: E402
    MSCI_UNIVERSE_CSV,
    build_msci_world_universe,
)
from app.services.fundamental.engine.universe import (  # noqa: E402
    CACHE_PATH,
    build_universe,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fundamental_universe")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the fundamental reference universes.")
    parser.add_argument(
        "--only",
        choices=["sp500", "msci_world"],
        default=None,
        help="Build only the given universe (default: both).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.6,
        help="Seconds between yfinance requests to avoid rate-limiting (default: 0.6).",
    )
    args = parser.parse_args()

    if args.only in (None, "sp500"):
        logger.info("Building S&P 500 universe → %s", CACHE_PATH)
        df = build_universe(delay=args.delay)
        logger.info("S&P 500 done: %d rows, %d columns", len(df), len(df.columns))

    if args.only in (None, "msci_world"):
        logger.info("Building MSCI World universe → %s", MSCI_UNIVERSE_CSV)
        df = build_msci_world_universe(delay=args.delay)
        logger.info("MSCI World done: %d rows, %d columns", len(df), len(df.columns))


if __name__ == "__main__":
    main()
