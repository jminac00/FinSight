"""Offline batch trainer for the deep learning module.

Trains one GRU per ticker from the local research CSVs and writes the
``{ticker}.pt`` + ``{ticker}.json`` artifacts the runtime consumes. Reads only an
explicit ``--tickers`` list: choosing the production universe (curated set vs
on-demand) is the job of the runtime service and the scheduler, not this script.

Usage:
    uv run python -m scripts.train_models --tickers AAPL,NVDA
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from app.services.deep_learning.artifacts import DEFAULT_MODELS_DIR
from app.services.deep_learning.preprocessing import FEATURES, InsufficientHistoryError, clean_ohlc
from app.services.deep_learning.recipe import load_frozen_recipe
from app.services.deep_learning.training.pipeline import train_ticker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_models")

DEFAULT_CSV_DIR = Path(__file__).resolve().parents[2] / "data" / "nasdaq_prices"


def _read_ohlc_csv(path: Path) -> pd.DataFrame:
    """Read one ticker CSV (columns ticker,date,open,high,low,close) into a clean,
    date-indexed OHLC frame."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"])
    return clean_ohlc(df.set_index("date")[FEATURES])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train per-ticker GRU models from local CSVs.")
    parser.add_argument("--tickers", required=True, help="Comma-separated tickers, e.g. AAPL,NVDA")
    parser.add_argument(
        "--csv-dir", type=Path, default=DEFAULT_CSV_DIR, help="Directory of OHLC CSVs"
    )
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
        csv_path = args.csv_dir / f"{ticker}.csv"
        if not csv_path.exists():
            logger.warning("%s: no CSV at %s — skipping", ticker, csv_path)
            skipped += 1
            continue
        try:
            ohlc = _read_ohlc_csv(csv_path)
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
        except Exception:
            logger.exception("%s: training failed", ticker)
            failed += 1

    logger.info("Done — trained=%d skipped=%d failed=%d", trained, skipped, failed)


if __name__ == "__main__":
    main()
