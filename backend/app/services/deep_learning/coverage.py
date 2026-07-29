"""Which tickers the deep learning module covers.

Coverage is the S&P 500 universe the fundamental engine already maintains, read
straight from its cached snapshot: one source of truth for index membership, and
no second download. The recipe was frozen on US large caps (ADR-0008), so a
ticker outside that universe has no validated model to offer.
"""

from __future__ import annotations

from functools import lru_cache

from app.services.fundamental.engine.universe import load_universe


@lru_cache(maxsize=1)
def universe_tickers() -> tuple[str, ...]:
    """Return the S&P 500 tickers, in snapshot order.

    Cached for the process lifetime: the snapshot only changes when the weekly
    refresh job rebuilds it, and ``predict`` consults this on every request.
    """
    return tuple(load_universe()["ticker"].dropna().astype(str))


def is_covered(ticker: str) -> bool:
    """Return True if *ticker* belongs to the covered universe."""
    return ticker in universe_tickers()
