"""Resolve asset names ↔ tickers and infer asset_type via yfinance."""

import logging
import re
from typing import Optional

import yfinance as yf

logger = logging.getLogger(__name__)

# Reuters inline ticker pattern: e.g. <JNJ.N>, <AAPL.O>
_REUTERS_RE = re.compile(r"<([A-Z0-9^=]+)\.[A-Z]+>")

# yfinance quoteType → our asset_type enum
_QUOTE_TYPE_MAP: dict[str, str] = {
    "EQUITY": "STOCK",
    "ETF": "ETF",
    "INDEX": "INDEX",
    "FUTURE": "FUTURE",
    "MUTUALFUND": "MUTUALFUND",
    "CURRENCY": "CURRENCY",
    "CRYPTOCURRENCY": "CRYPTOCURRENCY",
    "COMMODITY": "COMMODITY",
}

# Cache: key → (ticker, name, asset_type)
_cache: dict[str, tuple[Optional[str], str, str]] = {}


def _yf_info(ticker: str) -> tuple[str, str]:
    """Return (name, asset_type) for a ticker via yfinance."""
    try:
        full = yf.Ticker(ticker).info
        name = full.get("longName") or full.get("shortName") or ticker
        quote_type = full.get("quoteType", "EQUITY")
        asset_type = _QUOTE_TYPE_MAP.get(quote_type, "STOCK")
        return name, asset_type
    except Exception:
        return ticker, "STOCK"


def resolve_from_name(entity_name: str, text: str) -> tuple[Optional[str], str, str]:
    """
    Resolve (ticker, name, asset_type) for a FinEntity annotation.

    Tries Reuters pattern in text first; falls back to yfinance search.
    Returns (None, entity_name, 'STOCK') if unresolvable.
    """
    if entity_name in _cache:
        return _cache[entity_name]

    # 1. Reuters pattern in text
    match = _REUTERS_RE.search(text)
    if match:
        ticker = match.group(1)
        name, asset_type = _yf_info(ticker)
        result = (ticker, name, asset_type)
        _cache[entity_name] = result
        return result

    # 2. yfinance search
    try:
        hits = yf.Search(entity_name, max_results=1).quotes
        if hits:
            ticker = hits[0].get("symbol", "")
            name = hits[0].get("longname") or hits[0].get("shortname") or entity_name
            quote_type = hits[0].get("quoteType", "EQUITY")
            asset_type = _QUOTE_TYPE_MAP.get(quote_type, "STOCK")
            result = (ticker or None, name, asset_type)
            _cache[entity_name] = result
            return result
    except Exception as exc:
        logger.debug("yfinance search failed for %r: %s", entity_name, exc)

    result = (None, entity_name, "STOCK")
    _cache[entity_name] = result
    return result


def resolve_from_ticker(ticker: str) -> tuple[str, str, str]:
    """
    Resolve (ticker, name, asset_type) for a FinMarBa ticker.

    Returns (ticker, ticker, 'STOCK') if yfinance lookup fails.
    """
    if ticker in _cache:
        return _cache[ticker]  # type: ignore[return-value]

    name, asset_type = _yf_info(ticker)
    result = (ticker, name, asset_type)
    _cache[ticker] = result
    return result
