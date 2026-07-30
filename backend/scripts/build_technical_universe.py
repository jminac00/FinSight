"""Offline batch script — builds the technical price-universe snapshots.

The snapshots are the cross-sectional reference (daily closes, plus OHLCV for the S&P 500) the
technical blocks normalize each ticker against. They are committed as seed data under
engine/data/ and refreshed daily by the scheduler; run this manually to regenerate them (e.g.
before a demo, or before committing a fresher snapshot).

Usage (from backend/):
    uv run python -m scripts.build_technical_universe               # both universes
    uv run python -m scripts.build_technical_universe --only sp500
    uv run python -m scripts.build_technical_universe --only msci_world
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure backend/ is on sys.path so `app` is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from app.services.technical.engine.universe_manager import build_snapshot  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("technical_universe")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the technical price-universe snapshots.")
    parser.add_argument(
        "--only",
        choices=["sp500", "msci_world"],
        default=None,
        help="Build only the given universe (default: both).",
    )
    args = parser.parse_args()

    universes = [args.only] if args.only else ["sp500", "msci_world"]
    for universe in universes:
        logger.info("Building technical %s snapshot...", universe)
        closes = build_snapshot(universe)
        logger.info("%s done: %d tickers, %d sessions", universe, closes.shape[1], closes.shape[0])


if __name__ == "__main__":
    main()
