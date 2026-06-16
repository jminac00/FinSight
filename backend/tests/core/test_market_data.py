"""Tests for the shared market-data fetch (network-free, mocked yfinance)."""

import numpy as np
import pandas as pd
import pytest

from app.core import market_data


def _index(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=n)


def test_flattens_multiindex_columns_to_ohlcv(monkeypatch):
    fields = ["Open", "High", "Low", "Close", "Volume"]
    columns = pd.MultiIndex.from_product([fields, ["AAPL"]])
    values = np.arange(3 * 5, dtype=float).reshape(3, 5)
    raw = pd.DataFrame(values, index=_index(3), columns=columns)
    monkeypatch.setattr(market_data.yf, "download", lambda *a, **k: raw)

    out = market_data.get_price_history("AAPL")

    assert list(out.columns) == fields
    assert len(out) == 3


def test_passes_expected_download_arguments(monkeypatch):
    raw = pd.DataFrame(
        {
            "Open": [1.0, 2.0],
            "High": [1.0, 2.0],
            "Low": [1.0, 2.0],
            "Close": [1.0, 2.0],
            "Volume": [10.0, 20.0],
        },
        index=_index(2),
    )
    calls = {}

    def fake_download(ticker, **kwargs):
        calls["ticker"] = ticker
        calls["kwargs"] = kwargs
        return raw

    monkeypatch.setattr(market_data.yf, "download", fake_download)

    market_data.get_price_history("MSFT", period="6mo")

    assert calls["ticker"] == "MSFT"
    assert calls["kwargs"]["period"] == "6mo"
    assert calls["kwargs"]["auto_adjust"] is True
    assert calls["kwargs"]["progress"] is False


def test_raises_on_empty_download(monkeypatch):
    monkeypatch.setattr(market_data.yf, "download", lambda *a, **k: pd.DataFrame())

    with pytest.raises(ValueError):
        market_data.get_price_history("AAPL")


def test_drops_rows_with_nan_close(monkeypatch):
    raw = pd.DataFrame(
        {
            "Open": [1.0, 2.0, 3.0],
            "High": [1.0, 2.0, 3.0],
            "Low": [1.0, 2.0, 3.0],
            "Close": [1.0, np.nan, 3.0],
            "Volume": [10.0, 20.0, 30.0],
        },
        index=_index(3),
    )
    monkeypatch.setattr(market_data.yf, "download", lambda *a, **k: raw)

    out = market_data.get_price_history("AAPL")

    assert len(out) == 2
    assert out["Close"].notna().all()


def test_raises_when_all_close_nan(monkeypatch):
    raw = pd.DataFrame(
        {
            "Open": [1.0, 2.0],
            "High": [1.0, 2.0],
            "Low": [1.0, 2.0],
            "Close": [np.nan, np.nan],
            "Volume": [10.0, 20.0],
        },
        index=_index(2),
    )
    monkeypatch.setattr(market_data.yf, "download", lambda *a, **k: raw)

    with pytest.raises(ValueError):
        market_data.get_price_history("AAPL")
