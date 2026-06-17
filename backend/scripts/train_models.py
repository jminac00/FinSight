"""Offline batch trainer for the deep learning module.

Trains one GRU per ticker from live end-of-day prices and writes the
``{ticker}.pt`` + ``{ticker}.json`` artifacts the runtime consumes. History is
fetched through the shared ``app.core.market_data`` module (yfinance), the same
price source the rest of the app uses. Reads only an explicit ``--tickers`` list:
choosing the production universe (curated set vs on-demand) is the job of the
runtime service and the scheduler, not this script.

Usage (from backend/):
    uv run python -m scripts.train_models --tickers AAPL,NVDA
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from app.core.market_data import get_price_history
from app.services.deep_learning.artifacts import DEFAULT_MODELS_DIR
from app.services.deep_learning.preprocessing import InsufficientHistoryError, clean_ohlc
from app.services.deep_learning.recipe import load_frozen_recipe
from app.services.deep_learning.training.pipeline import train_ticker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_models")


def _fetch_ohlc(ticker: str) -> pd.DataFrame:
    """Download a ticker's full adjusted history and clean it for training.

    ``get_price_history`` returns capitalized OHLCV columns; ``clean_ohlc`` expects
    lowercase OHLC, so lower-case the columns at this boundary before cleaning.
    """
    history = get_price_history(ticker, period="max")
    history.columns = history.columns.str.lower()
    return clean_ohlc(history)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train per-ticker GRU models from live prices.")
    parser.add_argument("--tickers", required=True, help="Comma-separated tickers, e.g. AAPL,NVDA")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_MODELS_DIR, help="Artifacts output"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    recipe = load_frozen_recipe()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    trained = skipped = failed = 0
    for ticker in tickers:
        try:
            ohlc = _fetch_ohlc(ticker)
            artifacts = train_ticker(
                ohlc,
                ticker,
                recipe,
                seed=args.seed,
                max_epochs=args.max_epochs,
                patience=args.patience,
            )
            artifacts.save(args.output_dir)
            logger.info(
                "%s: trained on %d samples through %s — %s",
                ticker,
                artifacts.metadata.n_samples,
                artifacts.metadata.data_through,
                artifacts.metadata.metrics,
            )
            trained += 1
        except InsufficientHistoryError as exc:
            logger.warning("%s: %s — skipping", ticker, exc)
            skipped += 1
        except ValueError as exc:
            logger.warning("%s: no usable price data (%s) — skipping", ticker, exc)
            skipped += 1
        except Exception:
            logger.exception("%s: training failed", ticker)
            failed += 1

    logger.info("Done — trained=%d skipped=%d failed=%d", trained, skipped, failed)


if __name__ == "__main__":
    main()
