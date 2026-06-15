"""Contract tests for the fundamental endpoint (service dependency overridden)."""

from datetime import UTC, datetime

import pytest

from app.api.v1.fundamental import get_fundamental_service
from app.main import app
from app.models.fundamental import FundamentalResult
from app.services.fundamental.service import UniverseNotReadyError, UnknownTickerError


class _FakeService:
    def __init__(self, result: FundamentalResult | None = None, exc: Exception | None = None):
        self._result = result
        self._exc = exc

    async def analyze(self, ticker: str, force_refresh: bool = False) -> FundamentalResult:
        if self._exc is not None:
            raise self._exc
        return self._result


def _result() -> FundamentalResult:
    return FundamentalResult(
        score=7.2,
        metrics={"scores": {"valoracion": 5.0, "calidad": 8.0}},
        llm_analysis="Análisis fundamental en español.",
        cached_at=datetime.now(tz=UTC),
    )


@pytest.fixture
def override():
    def _apply(service: _FakeService):
        app.dependency_overrides[get_fundamental_service] = lambda: service

    yield _apply
    app.dependency_overrides.pop(get_fundamental_service, None)


def test_returns_200_with_payload(client, override):
    override(_FakeService(result=_result()))
    response = client.get("/api/v1/fundamental/AAPL")
    assert response.status_code == 200
    data = response.json()
    assert data["score"] == 7.2
    assert data["llm_analysis"]
    assert "scores" in data["metrics"]


def test_invalid_ticker_returns_422(client, override):
    override(_FakeService(result=_result()))
    response = client.get("/api/v1/fundamental/TOOLONG")
    assert response.status_code == 422


def test_unknown_ticker_returns_404(client, override):
    override(_FakeService(exc=UnknownTickerError("not found")))
    response = client.get("/api/v1/fundamental/ZZZZ")
    assert response.status_code == 404


def test_universe_not_ready_returns_503(client, override):
    override(_FakeService(exc=UniverseNotReadyError("warming up")))
    response = client.get("/api/v1/fundamental/AAPL")
    assert response.status_code == 503
