"""Unit tests for DLService.predict and is_model_available."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import numpy as np
import pytest

from app.services.deep_learning.preprocessing import InsufficientHistoryError
from app.services.deep_learning.recipe import load_frozen_recipe
from app.services.deep_learning.service import DLService, ModelNotAvailableError
from app.services.deep_learning.training.pipeline import train_ticker

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def recipe():
    return load_frozen_recipe()


@pytest.fixture
def fake_artifacts(make_ohlc, tmp_path, recipe):
    """Train a minimal model (3 epochs) and save artifacts to tmp_path."""
    artifacts = train_ticker(make_ohlc(n=400, seed=7), "AAPL", recipe, seed=42, max_epochs=3)
    artifacts.save(tmp_path)
    return tmp_path


@pytest.fixture
def service(fake_artifacts):
    return DLService(models_dir=fake_artifacts, max_models=5)


def _make_ohlc_df(n: int = 60, seed: int = 0):
    """Return a synthetic OHLC DataFrame with uppercase column names (as yfinance returns)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.02, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = close * (1.0 + rng.normal(0.0, 0.005, n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.005, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.005, n)))
    import pandas as pd

    dates = pd.bdate_range("2025-01-01", periods=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1e6},
        index=dates,
    )


# ---------------------------------------------------------------------------
# is_model_available
# ---------------------------------------------------------------------------


async def test_is_model_available_true_when_both_files_exist(fake_artifacts):
    service = DLService(models_dir=fake_artifacts, max_models=2)
    assert await service.is_model_available("AAPL") is True


async def test_is_model_available_false_when_pt_missing(fake_artifacts):
    (fake_artifacts / "AAPL.pt").unlink()
    service = DLService(models_dir=fake_artifacts, max_models=2)
    assert await service.is_model_available("AAPL") is False


async def test_is_model_available_false_when_json_missing(fake_artifacts):
    (fake_artifacts / "AAPL.json").unlink()
    service = DLService(models_dir=fake_artifacts, max_models=2)
    assert await service.is_model_available("AAPL") is False


# ---------------------------------------------------------------------------
# predict — error paths
# ---------------------------------------------------------------------------


async def test_predict_raises_when_no_artifacts(tmp_path):
    service = DLService(models_dir=tmp_path, max_models=2)
    with pytest.raises(ModelNotAvailableError):
        await service.predict("AAPL")


async def test_predict_raises_insufficient_history_when_few_candles(service):
    """10-row frame is below lookback=24 — must raise InsufficientHistoryError."""

    short_df = _make_ohlc_df(n=10)
    with patch("app.services.deep_learning.service.get_price_history", return_value=short_df):
        with pytest.raises(InsufficientHistoryError):
            await service.predict("AAPL")


async def test_predict_raises_value_error_when_yfinance_returns_no_data(service):
    with patch(
        "app.services.deep_learning.service.get_price_history",
        side_effect=ValueError("No price data found for 'XX'."),
    ):
        with pytest.raises(ValueError, match="No price data"):
            await service.predict("AAPL")


# ---------------------------------------------------------------------------
# predict — trend classification (mock _forward to control return value)
# ---------------------------------------------------------------------------


def _patched_predict(service, fake_artifacts, return_value: float):
    """Return a predict coroutine with _forward patched to a fixed return value."""
    df = _make_ohlc_df(n=60)
    return (
        patch("app.services.deep_learning.service.get_price_history", return_value=df),
        patch.object(DLService, "_forward", return_value=return_value),
    )


async def test_predict_trend_alcista_above_band(service):
    df = _make_ohlc_df(n=60)
    with patch("app.services.deep_learning.service.get_price_history", return_value=df):
        with patch.object(DLService, "_forward", return_value=2.5):
            result = await service.predict("AAPL")
    assert result.trend == "alcista"


async def test_predict_trend_bajista_below_band(service):
    df = _make_ohlc_df(n=60)
    with patch("app.services.deep_learning.service.get_price_history", return_value=df):
        with patch.object(DLService, "_forward", return_value=-2.5):
            result = await service.predict("AAPL")
    assert result.trend == "bajista"


async def test_predict_trend_neutral_within_band(service):
    df = _make_ohlc_df(n=60)
    with patch("app.services.deep_learning.service.get_price_history", return_value=df):
        with patch.object(DLService, "_forward", return_value=0.5):
            result = await service.predict("AAPL")
    assert result.trend == "neutral"


async def test_predict_neutral_at_positive_boundary(service):
    """Exactly 1.5 is within the neutral band (not alcista)."""
    df = _make_ohlc_df(n=60)
    with patch("app.services.deep_learning.service.get_price_history", return_value=df):
        with patch.object(DLService, "_forward", return_value=1.5):
            result = await service.predict("AAPL")
    assert result.trend == "neutral"


async def test_predict_neutral_at_negative_boundary(service):
    """Exactly -1.5 is within the neutral band (not bajista)."""
    df = _make_ohlc_df(n=60)
    with patch("app.services.deep_learning.service.get_price_history", return_value=df):
        with patch.object(DLService, "_forward", return_value=-1.5):
            result = await service.predict("AAPL")
    assert result.trend == "neutral"


# ---------------------------------------------------------------------------
# predict — derived fields and metadata passthrough
# ---------------------------------------------------------------------------


async def test_predict_derives_predicted_price_correctly(service):
    df = _make_ohlc_df(n=60)
    with patch("app.services.deep_learning.service.get_price_history", return_value=df):
        with patch.object(DLService, "_forward", return_value=10.0):
            result = await service.predict("AAPL")
    expected_price = result.current_price * 1.1
    assert abs(result.predicted_price - expected_price) < 1e-6


async def test_predict_populates_metrics_from_metadata(service):
    df = _make_ohlc_df(n=60)
    with patch("app.services.deep_learning.service.get_price_history", return_value=df):
        with patch.object(DLService, "_forward", return_value=0.0):
            result = await service.predict("AAPL")
    assert isinstance(result.metrics.rmse, float)
    assert isinstance(result.metrics.mae, float)
    assert 0.0 <= result.metrics.directional_accuracy <= 1.0


async def test_predict_returns_correct_horizon_days(service):
    df = _make_ohlc_df(n=60)
    with patch("app.services.deep_learning.service.get_price_history", return_value=df):
        with patch.object(DLService, "_forward", return_value=0.0):
            result = await service.predict("AAPL")
    assert result.horizon_days == 10


async def test_predict_trained_at_is_datetime(service):
    df = _make_ohlc_df(n=60)
    with patch("app.services.deep_learning.service.get_price_history", return_value=df):
        with patch.object(DLService, "_forward", return_value=0.0):
            result = await service.predict("AAPL")
    assert isinstance(result.trained_at, datetime)


# ---------------------------------------------------------------------------
# LRU cache — second call should not re-load from disk
# ---------------------------------------------------------------------------


async def test_second_predict_call_hits_cache(service):
    df = _make_ohlc_df(n=60)
    with patch("app.services.deep_learning.service.get_price_history", return_value=df):
        with patch.object(DLService, "_forward", return_value=0.0):
            with patch.object(service, "_get_model", wraps=service._get_model) as spy:
                await service.predict("AAPL")
                await service.predict("AAPL")
                # _get_model called twice but the inner disk load only happens once
                # (second call reads from self._cache)
                assert spy.call_count == 2
                assert len(service._cache) == 1
