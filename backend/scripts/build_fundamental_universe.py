"""Offline batch script — builds the S&P 500 fundamental universe snapshot.

The snapshot is the cross-sectional reference (yields, ratios, growth metrics per company) the
fundamental module normalizes each ticker against. It takes ~6-8 minutes (yfinance) and is
refreshed daily by the scheduler; run it manually before a demo on a fresh deploy.

Usage (from backend/):
    uv run python scripts/build_fundamental_universe.py
    uv run python scripts/build_fundamental_universe.py --output path/to/universe.csv
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure backend/ is on sys.path so `app` is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from app.core.config import get_settings  # noqa: E402
from app.services.fundamental.engine.universe import (  # noqa: E402
    DEFAULT_UNIVERSE_PATH,
    build_universe,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fundamental_universe")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the S&P 500 fundamental universe snapshot.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (defaults to FUNDAMENTAL_UNIVERSE_PATH or the engine default).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.6,
        help="Seconds between yfinance requests to avoid rate-limiting (default: 0.6).",
    )
    args = parser.parse_args()

    settings = get_settings()
    output = args.output or (
        Path(settings.fundamental_universe_path)
        if settings.fundamental_universe_path
        else DEFAULT_UNIVERSE_PATH
    )

    logger.info("Building fundamental universe → %s", output)
    df = build_universe(output, delay=args.delay)
    logger.info("Done: %d rows, %d columns saved to %s", len(df), len(df.columns), output)


if __name__ == "__main__":
    main()
