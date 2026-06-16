"""Tests for the per-ticker training CLI (network-free, mocked market data).

The CLI fetches history through ``app.core.market_data.get_price_history``; these
tests patch that boundary so nothing touches the network.
"""

import sys

import numpy as np
import pandas as pd

from scripts import train_models


def _yf_history(n: int = 6) -> pd.DataFrame:
    """A capitalized-column OHLCV frame shaped like ``get_price_history`` output."""
    base = np.linspace(100.0, 105.0, n)
    return pd.DataFrame(
        {
            "Open": base,
            "High": base + 1.0,
            "Low": base - 1.0,
            "Close": base + 0.5,
            "Volume": np.arange(1, n + 1, dtype=float),
        },
        index=pd.bdate_range("2020-01-01", periods=n),
    )


def _coherent_history(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """A long, OHLC-coherent capitalized frame, enough to fill train/val/test."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, n)))
    open_ = close * (1.0 + rng.normal(0.0, 0.005, n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.005, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.005, n)))
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": rng.integers(1, 100, n).astype(float),
        },
        index=pd.bdate_range("2018-01-01", periods=n),
    )


def test_fetch_ohlc_requests_full_history(monkeypatch):
    captured = {}

    def fake(ticker, period):
        captured["ticker"] = ticker
        captured["period"] = period
        return _yf_history()

    monkeypatch.setattr(train_models, "get_price_history", fake)

    train_models._fetch_ohlc("AAPL")

    assert captured == {"ticker": "AAPL", "period": "max"}


def test_fetch_ohlc_returns_clean_lowercase_ohlc(monkeypatch):
    monkeypatch.setattr(train_models, "get_price_history", lambda ticker, period: _yf_history())

    out = train_models._fetch_ohlc("AAPL")

    assert list(out.columns) == ["open", "high", "low", "close"]
    assert out.index.is_monotonic_increasing
    assert len(out) == 6


def test_main_skips_ticker_without_price_data(monkeypatch, tmp_path):
    called = {}

    def fake(ticker, period):
        called["ticker"] = ticker
        raise ValueError(f"No price data found for '{ticker}'.")

    monkeypatch.setattr(train_models, "get_price_history", fake)
    monkeypatch.setattr(
        sys, "argv", ["train_models", "--tickers", "NODATA", "--output-dir", str(tmp_path)]
    )

    train_models.main()  # must not raise

    assert called["ticker"] == "NODATA"
    assert list(tmp_path.glob("*.pt")) == []
    assert list(tmp_path.glob("*.json")) == []


def test_main_fetches_trains_and_writes_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(
        train_models, "get_price_history", lambda ticker, period: _coherent_history()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_models", "--tickers", "AAPL", "--output-dir", str(tmp_path), "--max-epochs", "3"],
    )

    train_models.main()

    assert (tmp_path / "AAPL.pt").exists()
    assert (tmp_path / "AAPL.json").exists()
