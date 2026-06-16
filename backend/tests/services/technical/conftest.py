"""Shared fixtures for the technical engine tests.

The engine normalizes every indicator against a reference universe, so the tests need a
deterministic, network-free universe. ``synthetic_universe`` builds one with a seeded RNG: a panel
of OHLCV-coherent daily series long enough (and cross-sectionally dispersed enough) for the four
blocks to produce non-null, normalizable scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Universe shape: enough sessions for momentum (needs >= 280) and enough tickers for the robust
# z-score minimum sample (>= 10).
_N_SESSIONS = 400
_N_TICKERS = 30
_BASE_SEED = 20260616


def _make_ticker_ohlcv(seed: int, drift: float, n: int) -> pd.DataFrame:
    """Build one OHLCV-coherent daily series via geometric brownian motion."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.018, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = close * (1.0 + rng.normal(0.0, 0.003, n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.004, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.004, n)))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume})


@pytest.fixture
def synthetic_universe() -> dict:
    """Deterministic universe payload matching ``_get_cached_universe``'s shape.

    Returns a dict with ``tickers`` (list), ``closes`` (DataFrame, tickers in columns) and
    ``ohlcv`` (dict of High/Low/Close/Volume DataFrames). Drift varies per ticker to create the
    cross-sectional dispersion the robust median/MAD normalization needs.
    """
    dates = pd.bdate_range("2014-01-01", periods=_N_SESSIONS)
    tickers = [f"T{i:02d}" for i in range(_N_TICKERS)]

    fields = ("Open", "High", "Low", "Close", "Volume")
    frames: dict[str, dict[str, pd.Series]] = {f: {} for f in fields}
    for i, ticker in enumerate(tickers):
        drift = -0.0006 + 0.00008 * i  # spread from mild downtrend to mild uptrend
        ohlcv = _make_ticker_ohlcv(seed=_BASE_SEED + i, drift=drift, n=_N_SESSIONS)
        ohlcv.index = dates
        for f in fields:
            frames[f][ticker] = ohlcv[f]

    field_dfs = {f: pd.DataFrame(frames[f]) for f in fields}

    closes = field_dfs["Close"]
    closes.attrs["universe_name"] = "golden_test_universe"

    ohlcv_payload = {f: field_dfs[f] for f in ("High", "Low", "Close", "Volume")}
    ohlcv_payload["Close"].attrs["universe_name"] = "golden_test_universe"

    return {"tickers": tickers, "closes": closes, "ohlcv": ohlcv_payload}
