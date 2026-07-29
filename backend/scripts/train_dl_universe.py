"""Offline batch trainer for the deep learning universe (S&P 500).

Trains one GRU per index constituent and lets the quality gate decide which
ones are published: a model that does not beat the naive zero-return predictor
leaves only its ``{ticker}.json``, recording why it is not served.

Tickers come from the same S&P 500 snapshot the rest of the app reads, through
``app.services.deep_learning.coverage``. Training goes through ``DLService``, so
the publication threshold lives in exactly one place.

Execution is sequential on purpose: on a 2 OCPU host parallel fits do not pay
off and they strain the yfinance rate limit. The run is resumable — tickers that
already have a published model, or that were already discarded, are skipped
unless ``--force`` is given.

Usage (from backend/):
    uv run python -m scripts.train_dl_universe
    uv run python -m scripts.train_dl_universe --limit 10
    uv run python -m scripts.train_dl_universe --force
    uv run python -m scripts.train_dl_universe --clean --yes
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from app.services.deep_learning.artifacts import DEFAULT_MODELS_DIR
from app.services.deep_learning.coverage import universe_tickers
from app.services.deep_learning.service import DLService, ModelQualityInsufficientError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_dl_universe")

_ARTIFACT_SUFFIXES = (".pt", ".json")


def clean_artifacts(models_dir: Path) -> int:
    """Delete every model artifact in *models_dir* and return how many were removed.

    Only ``.pt`` and ``.json`` files are touched, so ``.gitkeep`` — which keeps
    the directory tracked and carries no suffix — always survives.
    """
    removed = 0
    for path in sorted(models_dir.glob("*")):
        if path.is_file() and path.suffix in _ARTIFACT_SUFFIXES:
            path.unlink()
            removed += 1
    return removed


def is_resolved(models_dir: Path, ticker: str) -> bool:
    """Return True if *ticker* already has a verdict and can be skipped.

    A ticker is resolved when it has a published model on disk, or when the gate
    already discarded it. Unreadable metadata counts as unresolved: retraining
    is cheaper than reasoning about a corrupt file.
    """
    json_path = models_dir / f"{ticker}.json"
    if not json_path.is_file():
        return False
    try:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not metadata.get("published", True):
        return True
    return (models_dir / f"{ticker}.pt").is_file()


def pending_tickers(models_dir: Path, *, force: bool, limit: int | None) -> list[str]:
    """Return the tickers this run should train, in universe order."""
    tickers = list(universe_tickers())
    if not force:
        tickers = [t for t in tickers if not is_resolved(models_dir, t)]
    return tickers[:limit] if limit else tickers


async def train_all(service: DLService, tickers: list[str]) -> tuple[int, int, int]:
    """Train every ticker in order; return (published, discarded, errors).

    One ticker's failure never aborts the run: the batch is long and a single
    bad download must not cost the rest of it.
    """
    published = discarded = errors = 0
    total = len(tickers)
    for index, ticker in enumerate(tickers, start=1):
        try:
            artifacts = await service.train(ticker)
        except ModelQualityInsufficientError as exc:
            discarded += 1
            logger.info("[%d/%d] %s: discarded — %s", index, total, ticker, exc)
        except Exception as exc:  # noqa: BLE001 — one ticker must not stop the batch
            errors += 1
            logger.warning("[%d/%d] %s: failed — %s", index, total, ticker, exc)
        else:
            published += 1
            logger.info(
                "[%d/%d] %s: published (skill_ratio=%.3f)",
                index,
                total,
                ticker,
                artifacts.metadata.skill_ratio,
            )
    return published, discarded, errors


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a GRU per S&P 500 constituent, publishing only the models "
        "that beat the naive predictor."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Train at most N tickers in this run (for smoke tests and chunked runs)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain every ticker, including those already published or discarded",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete all existing .pt and .json artifacts before training",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt for --clean",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args(sys.argv[1:])
    models_dir = DEFAULT_MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    removed = 0
    if args.clean:
        doomed = sum(1 for p in models_dir.glob("*") if p.suffix in _ARTIFACT_SUFFIXES)
        if not args.yes:
            answer = input(f"Delete {doomed} artifact files in {models_dir}? [y/N] ")
            if answer.strip().lower() not in {"y", "yes"}:
                logger.info("Aborted: nothing was deleted")
                return
        removed = clean_artifacts(models_dir)
        logger.info("Removed %d artifact files from %s", removed, models_dir)

    tickers = pending_tickers(models_dir, force=args.force, limit=args.limit)
    if not tickers:
        logger.info("Nothing to train — every ticker in the universe is already resolved")
        return

    logger.info("Training %d tickers sequentially into %s", len(tickers), models_dir)
    service = DLService(models_dir=models_dir, max_models=1)
    published, discarded, errors = asyncio.run(train_all(service, tickers))

    logger.info(
        "Done — published=%d discarded=%d errors=%d removed=%d",
        published,
        discarded,
        errors,
        removed,
    )


if __name__ == "__main__":
    main()
