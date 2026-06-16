"""Shared market-data access: single-ticker OHLC download via yfinance.

Single source of truth for end-of-day price history, consumed by the technical
engine and the deep-learning module so both fetch prices the same way. Prices are
adjusted for splits and dividends (``auto_adjust=True``).
"""

import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def get_price_history(ticker: str, period: str = "3y") -> pd.DataFrame:
    """Download the adjusted OHLCV history for a single ticker.

    Args:
        ticker: Stock symbol (e.g. ``"AAPL"``).
        period: yfinance period string (e.g. ``"3y"``, ``"max"``).

    Returns:
        A date-indexed DataFrame with flat ``Open, High, Low, Close, Volume``
        columns, adjusted for splits and dividends.

    Raises:
        ValueError: If no data is returned or no valid close prices remain.
    """
    logger.debug("Fetching price history for %s (period=%s)", ticker, period)
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data.empty:
        logger.warning("No price data returned for '%s'", ticker)
        raise ValueError(f"No price data found for '{ticker}'.")

    # yfinance returns MultiIndex (field, ticker) columns for some queries; flatten them.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    data = data.dropna(subset=["Close"])
    if data.empty:
        logger.warning("Price history for '%s' is empty after dropping NaN closes", ticker)
        raise ValueError(f"Price history for '{ticker}' is empty after removing NaN closes.")

    logger.debug(
        "Fetched %d sessions for %s [%s -> %s]",
        len(data),
        ticker,
        data.index[0].date(),
        data.index[-1].date(),
    )
    return data
