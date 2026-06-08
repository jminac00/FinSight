"""Parser for the FinMarBa CSV dataset."""

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_SENTIMENT_MAP = {1: "Positive", 0: "Neutral", -1: "Negative"}


@dataclass
class FinMarBaRecord:
    title: str
    date: str  # ISO date string "YYYY-MM-DD"
    tickers: list[str] = field(default_factory=list)
    sentiment: dict[str, str] = field(default_factory=dict)  # ticker → label
    sentiment_score: dict[str, int] = field(default_factory=dict)  # ticker → -1/0/1
    pct_change: dict[str, float] = field(default_factory=dict)


def _parse_tickers(raw: str) -> list[str]:
    """Parse the double-escaped tickers column: ["['A', 'B']"] → ['A', 'B']."""
    outer = ast.literal_eval(raw)
    return ast.literal_eval(outer[0])


def _parse_dict(raw: str) -> dict:
    return ast.literal_eval(raw)


def load(path: Path) -> list[FinMarBaRecord]:
    """Load and parse the FinMarBa CSV dataset."""
    df = pd.read_csv(path)
    records: list[FinMarBaRecord] = []
    errors = 0
    for _, row in df.iterrows():
        try:
            tickers = _parse_tickers(str(row["Tickers"]))
            sent_raw: dict[str, int] = _parse_dict(str(row["Sentiment"]))
            pct_raw: dict[str, float] = _parse_dict(str(row["Pct_Change"]))
            records.append(
                FinMarBaRecord(
                    title=str(row["Title"]),
                    date=str(row["Date"]),
                    tickers=tickers,
                    sentiment={t: _SENTIMENT_MAP.get(v, "Neutral") for t, v in sent_raw.items()},
                    sentiment_score={t: int(v) for t, v in sent_raw.items()},
                    pct_change={t: float(v) for t, v in pct_raw.items()},
                )
            )
        except Exception as exc:
            errors += 1
            logger.warning("FinMarBa: skipping row (parse error: %s)", exc)
    logger.info("FinMarBa: loaded %d records from %s (%d skipped)", len(records), path.name, errors)
    return records
