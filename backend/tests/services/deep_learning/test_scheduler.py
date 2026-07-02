"""Unit tests for daily_model_update scheduler job."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.scheduler.jobs import daily_model_update

# get_dl_service is imported lazily inside daily_model_update, so we patch it
# at its source module so the lazy import picks up the mock.
_DL_SERVICE_PATCH = "app.api.v1.deep_learning.get_dl_service"
_MODELS_DIR_PATCH = "app.scheduler.jobs.DEFAULT_MODELS_DIR"


async def test_daily_model_update_calls_train_for_each_ticker(tmp_path):
    """Both tickers with .pt + .json pairs are retrained."""
    (tmp_path / "AAPL.pt").touch()
    (tmp_path / "AAPL.json").touch()
    (tmp_path / "NVDA.pt").touch()
    (tmp_path / "NVDA.json").touch()

    mock_service = MagicMock()
    mock_service.train = AsyncMock(return_value=MagicMock())

    with patch(_MODELS_DIR_PATCH, tmp_path):
        with patch(_DL_SERVICE_PATCH, return_value=mock_service):
            await daily_model_update()

    assert mock_service.train.await_count == 2
    called_tickers = {call.args[0] for call in mock_service.train.await_args_list}
    assert called_tickers == {"AAPL", "NVDA"}


async def test_daily_model_update_skips_ticker_missing_json(tmp_path):
    """A .pt without a matching .json is not retrained."""
    (tmp_path / "AAPL.pt").touch()
    (tmp_path / "AAPL.json").touch()
    (tmp_path / "ORPHAN.pt").touch()  # no .json counterpart

    mock_service = MagicMock()
    mock_service.train = AsyncMock(return_value=MagicMock())

    with patch(_MODELS_DIR_PATCH, tmp_path):
        with patch(_DL_SERVICE_PATCH, return_value=mock_service):
            await daily_model_update()

    assert mock_service.train.await_count == 1
    mock_service.train.assert_awaited_once_with("AAPL")


async def test_daily_model_update_continues_after_single_failure(tmp_path):
    """A failure on one ticker does not prevent processing the rest."""
    (tmp_path / "AAPL.pt").touch()
    (tmp_path / "AAPL.json").touch()
    (tmp_path / "NVDA.pt").touch()
    (tmp_path / "NVDA.json").touch()

    mock_service = MagicMock()
    mock_service.train = AsyncMock(side_effect=[RuntimeError("network error"), MagicMock()])

    with patch(_MODELS_DIR_PATCH, tmp_path):
        with patch(_DL_SERVICE_PATCH, return_value=mock_service):
            await daily_model_update()  # must not raise

    assert mock_service.train.await_count == 2


async def test_daily_model_update_no_op_when_empty_dir(tmp_path):
    """When ml_models/ is empty, service.train is never called."""
    mock_service = MagicMock()
    mock_service.train = AsyncMock()

    with patch(_MODELS_DIR_PATCH, tmp_path):
        with patch(_DL_SERVICE_PATCH, return_value=mock_service):
            await daily_model_update()

    mock_service.train.assert_not_awaited()
