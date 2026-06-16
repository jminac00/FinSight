"""Contract tests for the technical endpoint. The technical service is mocked."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.api.v1.technical import get_technical_service
from app.main import app
from app.models.technical import TechnicalBlockScores, TechnicalResult
from app.services.technical.service import TechnicalAnalysisError, TechnicalUniverseNotReadyError

_RESULT = TechnicalResult(
    score=6.42,
    signal="neutral",
    block_scores=TechnicalBlockScores(
        momentum=7.1, trend=6.0, risk_stability=5.5, confirmation=6.8
    ),
    indicators={"universe": "sp500", "score_reliable": True},
    llm_analysis="Análisis técnico en español.",
    calculated_at=datetime.now(tz=UTC),
)


def _override(service: MagicMock) -> None:
    app.dependency_overrides[get_technical_service] = lambda: service


@pytest.fixture
def mock_service():
    service = MagicMock()
    service.analyze = AsyncMock(return_value=_RESULT)
    _override(service)
    yield service
    app.dependency_overrides.pop(get_technical_service, None)


def test_get_technical_returns_analysis(client, mock_service):
    response = client.get("/api/v1/technical/AAPL")

    assert response.status_code == 200
    body = response.json()
    assert body["score"] == 6.42
    assert body["signal"] == "neutral"
    assert body["block_scores"]["momentum"] == 7.1
    assert body["indicators"]["universe"] == "sp500"
    mock_service.analyze.assert_awaited_once_with("AAPL", mode="auto", force_refresh=False)


def test_mode_and_force_refresh_are_forwarded(client, mock_service):
    response = client.get("/api/v1/technical/ASML?mode=global&force_refresh=true")

    assert response.status_code == 200
    mock_service.analyze.assert_awaited_once_with("ASML", mode="global", force_refresh=True)


def test_invalid_ticker_returns_422(client, mock_service):
    response = client.get("/api/v1/technical/@@@")

    assert response.status_code == 422
    mock_service.analyze.assert_not_awaited()


def test_universe_not_ready_returns_503(client):
    service = MagicMock()
    service.analyze = AsyncMock(side_effect=TechnicalUniverseNotReadyError("snapshot missing"))
    _override(service)
    try:
        response = client.get("/api/v1/technical/AAPL")
        assert response.status_code == 503
    finally:
        app.dependency_overrides.pop(get_technical_service, None)


def test_analysis_error_returns_404(client):
    service = MagicMock()
    service.analyze = AsyncMock(side_effect=TechnicalAnalysisError("Insufficient technical data"))
    _override(service)
    try:
        response = client.get("/api/v1/technical/AAPL")
        assert response.status_code == 404
    finally:
        app.dependency_overrides.pop(get_technical_service, None)
