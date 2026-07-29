"""Unit tests for daily_model_update scheduler job."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.scheduler.jobs import daily_model_update
from app.services.deep_learning.service import ModelQualityInsufficientError

# get_dl_service is imported lazily inside daily_model_update, so we patch it
# at its source module so the lazy import picks up the mock.
_DL_SERVICE_PATCH = "app.api.v1.deep_learning.get_dl_service"
_MODELS_DIR_PATCH = "app.scheduler.jobs.DEFAULT_MODELS_DIR"
_UNIVERSE_PATCH = "app.scheduler.jobs.universe_tickers"


@pytest.fixture(autouse=True)
def empty_universe():
    """Default to an empty universe so tests opt in to new-member detection."""
    with patch(_UNIVERSE_PATCH, return_value=()):
        yield


def _discarded(models_dir, ticker: str) -> None:
    """Write the metadata-only artifact the quality gate leaves behind."""
    (models_dir / f"{ticker}.json").write_text(
        json.dumps({"ticker": ticker, "skill_ratio": 1.3, "published": False}), encoding="utf-8"
    )


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


# ---------------------------------------------------------------------------
# Newly added universe constituents
# ---------------------------------------------------------------------------


async def _run_update(tmp_path, mock_service, universe: tuple[str, ...]):
    with patch(_MODELS_DIR_PATCH, tmp_path):
        with patch(_UNIVERSE_PATCH, return_value=universe):
            with patch(_DL_SERVICE_PATCH, return_value=mock_service):
                await daily_model_update()


async def test_universe_members_without_artifacts_are_trained(tmp_path):
    """A company that just joined the index gets its first model."""
    mock_service = MagicMock()
    mock_service.train = AsyncMock(return_value=MagicMock())

    await _run_update(tmp_path, mock_service, ("AAPL", "NEWCO"))

    trained = {call.args[0] for call in mock_service.train.await_args_list}
    assert trained == {"AAPL", "NEWCO"}


async def test_discarded_tickers_are_not_retried(tmp_path):
    """A model rejected on quality is only recomputed by the script with --force."""
    _discarded(tmp_path, "BADCO")
    mock_service = MagicMock()
    mock_service.train = AsyncMock(return_value=MagicMock())

    await _run_update(tmp_path, mock_service, ("BADCO", "NEWCO"))

    trained = {call.args[0] for call in mock_service.train.await_args_list}
    assert trained == {"NEWCO"}


async def test_existing_models_are_still_retrained_alongside_new_members(tmp_path):
    (tmp_path / "AAPL.pt").touch()
    (tmp_path / "AAPL.json").touch()
    _discarded(tmp_path, "BADCO")
    mock_service = MagicMock()
    mock_service.train = AsyncMock(return_value=MagicMock())

    await _run_update(tmp_path, mock_service, ("AAPL", "BADCO", "NEWCO"))

    trained = {call.args[0] for call in mock_service.train.await_args_list}
    assert trained == {"AAPL", "NEWCO"}


async def test_a_failing_new_member_does_not_abort_the_job(tmp_path):
    mock_service = MagicMock()
    mock_service.train = AsyncMock(side_effect=[RuntimeError("yfinance timeout"), MagicMock()])

    await _run_update(tmp_path, mock_service, ("ONE", "TWO"))

    assert mock_service.train.await_count == 2


async def test_a_discarded_new_member_does_not_abort_the_job(tmp_path):
    """The gate rejecting a fresh model is an expected outcome, not a crash."""
    mock_service = MagicMock()
    mock_service.train = AsyncMock(
        side_effect=[ModelQualityInsufficientError("skill_ratio=1.310"), MagicMock()]
    )

    await _run_update(tmp_path, mock_service, ("ONE", "TWO"))

    assert mock_service.train.await_count == 2


async def test_log_reports_how_many_new_members_were_trained(tmp_path, caplog):
    mock_service = MagicMock()
    mock_service.train = AsyncMock(
        side_effect=[MagicMock(), ModelQualityInsufficientError("skill_ratio=1.31")]
    )

    with caplog.at_level("INFO"):
        await _run_update(tmp_path, mock_service, ("ONE", "TWO"))

    messages = [r.getMessage() for r in caplog.records]
    assert any("1" in m and "new" in m.lower() for m in messages)
