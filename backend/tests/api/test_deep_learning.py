"""Contract tests for the deep learning endpoint."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.api.v1.deep_learning import get_dl_service
from app.core.config import get_settings
from app.main import app
from app.models.deep_learning import DLResult, ModelMetrics
from app.services.deep_learning.preprocessing import InsufficientHistoryError
from app.services.deep_learning.service import (
    ModelNotAvailableError,
    ModelQualityInsufficientError,
    OutOfCoverageError,
)

_RESULT = DLResult(
    trend="alcista",
    predicted_return_pct=7.24,
    predicted_price=195.50,
    current_price=182.30,
    horizon_days=10,
    trained_at=datetime(2026, 6, 3, 22, 0, 0, tzinfo=UTC),
    metrics=ModelMetrics(rmse=5.61, mae=4.39, directional_accuracy=0.60),
)


@pytest.fixture
def mock_dl_service():
    service = MagicMock()
    service.predict = AsyncMock(return_value=_RESULT)
    app.dependency_overrides[get_dl_service] = lambda: service
    yield service
    app.dependency_overrides.pop(get_dl_service, None)


# ---------------------------------------------------------------------------
# Existing contract tests (kept unchanged)
# ---------------------------------------------------------------------------


def test_prediction_returns_return_based_contract(client, mock_dl_service):
    response = client.get("/api/v1/prediction/AAPL")
    assert response.status_code == 200
    data = response.json()

    assert data["trend"] in {"alcista", "bajista", "neutral"}
    assert isinstance(data["predicted_return_pct"], float)
    assert isinstance(data["predicted_price"], float)
    assert isinstance(data["current_price"], float)
    assert data["horizon_days"] == 10
    assert "trained_at" in data

    metrics = data["metrics"]
    assert set(metrics) == {"rmse", "mae", "directional_accuracy"}

    # Price-prediction leftovers must be gone.
    assert "pct_change" not in data
    assert "mape" not in metrics
    assert "r2" not in metrics


def test_prediction_rejects_invalid_ticker(client):
    response = client.get("/api/v1/prediction/toolongtickerxyz")  # 16 chars > limit
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# New endpoint tests
# ---------------------------------------------------------------------------


def test_get_prediction_delegates_to_service(client, mock_dl_service):
    response = client.get("/api/v1/prediction/AAPL")
    assert response.status_code == 200
    mock_dl_service.predict.assert_awaited_once_with("AAPL")


def test_get_prediction_lowercase_ticker_normalised(client, mock_dl_service):
    response = client.get("/api/v1/prediction/aapl")
    assert response.status_code == 200
    mock_dl_service.predict.assert_awaited_once_with("AAPL")


def test_get_prediction_model_not_available_returns_404(client):
    service = MagicMock()
    service.predict = AsyncMock(side_effect=ModelNotAvailableError("No trained model for 'FAKE'"))
    app.dependency_overrides[get_dl_service] = lambda: service
    try:
        response = client.get("/api/v1/prediction/FAKE")
        assert response.status_code == 404
    finally:
        app.dependency_overrides.pop(get_dl_service, None)


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (OutOfCoverageError("'REP.MC' is outside the S&P 500 universe"), "out_of_coverage"),
        (ModelNotAvailableError("No trained model for 'NVDA'"), "not_trained"),
        (
            ModelQualityInsufficientError("Model for 'XOM' does not beat the naive predictor"),
            "insufficient_quality",
        ),
    ],
)
def test_get_prediction_404_body_carries_the_reason(client, error, reason):
    """Each unavailability state is a 404 whose body states why."""
    service = MagicMock()
    service.predict = AsyncMock(side_effect=error)
    app.dependency_overrides[get_dl_service] = lambda: service
    try:
        response = client.get("/api/v1/prediction/AAPL")
        assert response.status_code == 404
        detail = response.json()["detail"]
        assert detail["reason"] == reason
        assert detail["message"] == str(error)
    finally:
        app.dependency_overrides.pop(get_dl_service, None)


def test_get_prediction_insufficient_history_returns_422(client):
    service = MagicMock()
    service.predict = AsyncMock(
        side_effect=InsufficientHistoryError("Need at least 24 clean candles, got 10")
    )
    app.dependency_overrides[get_dl_service] = lambda: service
    try:
        response = client.get("/api/v1/prediction/AAPL")
        assert response.status_code == 422
    finally:
        app.dependency_overrides.pop(get_dl_service, None)


def test_get_prediction_yfinance_no_data_returns_422(client):
    service = MagicMock()
    service.predict = AsyncMock(side_effect=ValueError("No price data found for 'XX'."))
    app.dependency_overrides[get_dl_service] = lambda: service
    try:
        response = client.get("/api/v1/prediction/XX")
        assert response.status_code == 422
    finally:
        app.dependency_overrides.pop(get_dl_service, None)


# ---------------------------------------------------------------------------
# POST /dl/train/{ticker} — dev-only training endpoint
# ---------------------------------------------------------------------------

_FAKE_ARTIFACTS = MagicMock()
_FAKE_ARTIFACTS.metadata.trained_at = "2026-07-01T22:00:00+00:00"
_FAKE_ARTIFACTS.metadata.metrics = {"rmse": 5.61, "mae": 4.39, "directional_accuracy": 0.60}
_FAKE_ARTIFACTS.metadata.data_through = "2026-07-01"


def test_train_returns_403_in_production(client, mock_dl_service):
    mock_settings = MagicMock()
    mock_settings.environment = "production"
    app.dependency_overrides[get_settings] = lambda: mock_settings
    try:
        response = client.post("/api/v1/train/AAPL")
        assert response.status_code == 403
    finally:
        app.dependency_overrides.pop(get_settings, None)


def test_train_returns_422_for_invalid_ticker(client, mock_dl_service):
    response = client.post("/api/v1/train/toolongtickerxyz")  # 16 chars > limit
    assert response.status_code == 422


def test_train_delegates_to_service_train(client, mock_dl_service):
    mock_dl_service.train = AsyncMock(return_value=_FAKE_ARTIFACTS)
    response = client.post("/api/v1/train/aapl")
    assert response.status_code == 200
    mock_dl_service.train.assert_awaited_once_with("AAPL")


def test_train_returns_200_with_dl_train_result_schema(client, mock_dl_service):
    mock_dl_service.train = AsyncMock(return_value=_FAKE_ARTIFACTS)
    response = client.post("/api/v1/train/AAPL")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "AAPL"
    assert "trained_at" in data
    assert "data_through" in data
    assert set(data["metrics"]) == {"rmse", "mae", "directional_accuracy"}


# ---------------------------------------------------------------------------
# get_dl_service — auto_train wiring from environment
# ---------------------------------------------------------------------------


def test_get_dl_service_enables_auto_train_outside_production(monkeypatch):
    """Locally (development) the service trains missing models on demand."""
    from app.api.v1 import deep_learning as dl_mod

    dl_mod.get_dl_service.cache_clear()
    monkeypatch.setattr(
        dl_mod,
        "get_settings",
        lambda: MagicMock(environment="development", lru_cache_max_models=10),
    )
    try:
        service = dl_mod.get_dl_service()
        assert service._auto_train is True
    finally:
        dl_mod.get_dl_service.cache_clear()


def test_get_dl_service_disables_auto_train_in_production(monkeypatch):
    """In production the service never trains on demand (no GPU, scarce CPU)."""
    from app.api.v1 import deep_learning as dl_mod

    dl_mod.get_dl_service.cache_clear()
    monkeypatch.setattr(
        dl_mod,
        "get_settings",
        lambda: MagicMock(environment="production", lru_cache_max_models=10),
    )
    try:
        service = dl_mod.get_dl_service()
        assert service._auto_train is False
    finally:
        dl_mod.get_dl_service.cache_clear()
