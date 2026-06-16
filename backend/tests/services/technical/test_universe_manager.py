"""Tests for the technical universe manager: snapshot round-trip, readiness and universe routing."""

from __future__ import annotations

import numpy as np
import pytest

from app.services.technical.engine.universe_manager import (
    TechnicalUniverseManager,
    TechnicalUniverseNotReadyError,
    save_snapshot,
)


def test_snapshot_round_trip_with_ohlcv(synthetic_universe: dict, tmp_path) -> None:
    closes = synthetic_universe["closes"]
    ohlcv = synthetic_universe["ohlcv"]
    save_snapshot("sp500", closes, ohlcv, data_dir=tmp_path)

    payload = TechnicalUniverseManager(data_dir=tmp_path).load("sp500")

    loaded_closes = payload["closes"]
    assert loaded_closes.shape == closes.shape
    assert list(loaded_closes.columns) == list(closes.columns)
    assert np.allclose(loaded_closes.to_numpy(), closes.to_numpy())
    assert loaded_closes.attrs["universe_name"] == "technical_sp500"

    assert payload["ohlcv"] is not None
    assert set(payload["ohlcv"]) == {"Close", "High", "Low", "Volume"}
    # Close shares the closes object so the per-block cache keys stay consistent.
    assert payload["ohlcv"]["Close"] is loaded_closes


def test_snapshot_round_trip_closes_only(synthetic_universe: dict, tmp_path) -> None:
    closes = synthetic_universe["closes"]
    save_snapshot("msci_world", closes, None, data_dir=tmp_path)

    payload = TechnicalUniverseManager(data_dir=tmp_path).load("msci_world")

    assert payload["ohlcv"] is None
    assert payload["closes"].attrs["universe_name"] == "technical_msci_world"


def test_load_missing_snapshot_raises(tmp_path) -> None:
    with pytest.raises(TechnicalUniverseNotReadyError):
        TechnicalUniverseManager(data_dir=tmp_path).load("sp500")


def test_load_is_cached(synthetic_universe: dict, tmp_path) -> None:
    save_snapshot(
        "sp500", synthetic_universe["closes"], synthetic_universe["ohlcv"], data_dir=tmp_path
    )
    manager = TechnicalUniverseManager(data_dir=tmp_path)
    assert manager.load("sp500") is manager.load("sp500")


def test_resolve_universe_modes() -> None:
    manager = TechnicalUniverseManager()
    assert manager.resolve_universe("AAPL", "domestic") == "sp500"
    assert manager.resolve_universe("AAPL", "global") == "msci_world"
    # A symbol that is not an S&P 500 member routes to the global universe in auto mode.
    assert manager.resolve_universe("ZZZZ", "auto") == "msci_world"
