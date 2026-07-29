"""predict() tells the four availability states apart.

The universe is always stubbed: resolving coverage must read the cached
fundamental universe, never reach the network.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd
import pytest

from app.services.deep_learning.service import (
    DLService,
    DLUnavailableError,
    ModelNotAvailableError,
    ModelQualityInsufficientError,
    OutOfCoverageError,
)

_LOAD_UNIVERSE = "app.services.deep_learning.coverage.load_universe"


@pytest.fixture(autouse=True)
def stub_universe():
    """Cover AAPL and NVDA; anything else is outside the universe."""
    from app.services.deep_learning import coverage

    coverage.universe_tickers.cache_clear()
    frame = pd.DataFrame({"ticker": ["AAPL", "NVDA"], "sector": ["Tech", "Tech"]})
    with patch(_LOAD_UNIVERSE, return_value=frame) as loader:
        yield loader
    coverage.universe_tickers.cache_clear()


@pytest.fixture
def service(tmp_path):
    return DLService(models_dir=tmp_path, max_models=2)


def _write_discarded_metadata(models_dir, ticker: str) -> None:
    """Write the metadata-only artifact the gate leaves for a discarded model."""
    (models_dir / f"{ticker}.json").write_text(
        json.dumps(
            {
                "ticker": ticker,
                "trained_at": "2026-07-20T22:00:00+00:00",
                "horizon_days": 10,
                "lookback": 24,
                "input_size": 9,
                "recipe": {},
                "metrics": {"rmse": 6.0, "mae": 4.8, "directional_accuracy": 0.48,
                            "rmse_naive": 5.0},
                "n_samples": 300,
                "data_through": "2026-07-20",
                "skill_ratio": 1.2,
                "published": False,
            }
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# The three unavailability reasons
# ---------------------------------------------------------------------------


async def test_ticker_outside_the_universe_is_out_of_coverage(service):
    with pytest.raises(OutOfCoverageError) as exc:
        await service.predict("REP.MC")
    assert exc.value.reason == "out_of_coverage"


async def test_ticker_in_the_universe_without_artifacts_is_not_trained(service):
    with pytest.raises(ModelNotAvailableError) as exc:
        await service.predict("NVDA")
    assert exc.value.reason == "not_trained"


async def test_discarded_model_is_insufficient_quality(service, tmp_path):
    _write_discarded_metadata(tmp_path, "NVDA")
    with pytest.raises(ModelQualityInsufficientError) as exc:
        await service.predict("NVDA")
    assert exc.value.reason == "insufficient_quality"


async def test_discarded_model_wins_over_coverage(service, tmp_path):
    """A discarded ticker reports why it was discarded, not that it is uncovered."""
    _write_discarded_metadata(tmp_path, "REP.MC")
    with pytest.raises(ModelQualityInsufficientError) as exc:
        await service.predict("REP.MC")
    assert exc.value.reason == "insufficient_quality"


async def test_all_reasons_share_one_catchable_base(service):
    with pytest.raises(DLUnavailableError):
        await service.predict("REP.MC")


# ---------------------------------------------------------------------------
# A published model is still served
# ---------------------------------------------------------------------------


async def test_published_model_is_not_blocked_by_the_universe(make_ohlc, tmp_path):
    """A ticker with a published model is served even if it left the index."""
    from app.services.deep_learning.recipe import load_frozen_recipe
    from app.services.deep_learning.training.pipeline import train_ticker

    artifacts = train_ticker(make_ohlc(n=400, seed=9), "REP.MC", load_frozen_recipe(), max_epochs=3)
    artifacts.metadata.published = True
    artifacts.save(tmp_path)
    service = DLService(models_dir=tmp_path, max_models=2)

    raw = make_ohlc(n=80, seed=9)
    raw.columns = [c.capitalize() for c in raw.columns]
    with patch("app.services.deep_learning.service.get_price_history", return_value=raw):
        result = await service.predict("REP.MC")

    assert result.trend in {"alcista", "bajista", "neutral"}


# ---------------------------------------------------------------------------
# Coverage source
# ---------------------------------------------------------------------------


async def test_coverage_reads_the_cached_universe_loader(service, stub_universe):
    with pytest.raises(DLUnavailableError):
        await service.predict("ZZZZ")
    assert stub_universe.called


async def test_universe_is_loaded_once_and_cached(service, stub_universe):
    for ticker in ("ZZZZ", "YYYY", "XXXX"):
        with pytest.raises(DLUnavailableError):
            await service.predict(ticker)
    assert stub_universe.call_count == 1


def test_universe_tickers_returns_the_ticker_column():
    from app.services.deep_learning import coverage

    coverage.universe_tickers.cache_clear()
    frame = pd.DataFrame({"ticker": ["AAPL", "MSFT"], "sector": ["Tech", "Tech"]})
    try:
        with patch(_LOAD_UNIVERSE, return_value=frame):
            assert coverage.universe_tickers() == ("AAPL", "MSFT")
    finally:
        coverage.universe_tickers.cache_clear()


# ---------------------------------------------------------------------------
# Auto-training (local development) is unchanged
# ---------------------------------------------------------------------------


async def test_auto_train_still_runs_before_reporting_a_reason(tmp_path, make_ohlc):
    """Locally the service trains a missing model instead of reporting a reason."""
    service = DLService(models_dir=tmp_path, max_models=2, auto_train=True)

    async def _fake_train(ticker, max_epochs=50):
        from app.services.deep_learning.recipe import load_frozen_recipe
        from app.services.deep_learning.training.pipeline import train_ticker

        artifacts = train_ticker(make_ohlc(n=400, seed=1), ticker, load_frozen_recipe(),
                                 max_epochs=3)
        artifacts.metadata.published = True
        artifacts.save(tmp_path)
        return artifacts

    raw = make_ohlc(n=80, seed=1)
    raw.columns = [c.capitalize() for c in raw.columns]
    with patch.object(DLService, "train", side_effect=_fake_train):
        with patch("app.services.deep_learning.service.get_price_history", return_value=raw):
            result = await service.predict("NVDA")

    assert result.trend in {"alcista", "bajista", "neutral"}
