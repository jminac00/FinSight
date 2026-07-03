"""Rate limiting tests — written before implementation (TDD red phase)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.testclient import TestClient

from app.api.v1.deep_learning import get_dl_service
from app.api.v1.fundamental import get_fundamental_service
from app.api.v1.sentiment import get_sentiment_service
from app.api.v1.technical import get_technical_service
from app.core.rate_limit import limiter
from app.llm.factory import get_llm_service
from app.main import app
from app.models.deep_learning import DLResult, ModelMetrics
from app.models.fundamental import FundamentalResult
from app.models.sentiment import SentimentResult
from app.models.technical import TechnicalBlockScores, TechnicalResult

# ── fixtures ──────────────────────────────────────────────────────────────────

_SENTIMENT = SentimentResult(
    label="neutral", score=0.0, confidence=0.5, explanation="ok", influential_news=[]
)
_DL = DLResult(
    trend="neutral",
    predicted_return_pct=0.0,
    predicted_price=100.0,
    current_price=100.0,
    horizon_days=10,
    trained_at=datetime(2026, 1, 1, tzinfo=UTC),
    metrics=ModelMetrics(rmse=1.0, mae=0.8, directional_accuracy=0.5),
)
_FUNDAMENTAL = FundamentalResult(
    score=5.0, metrics={}, llm_analysis="ok", cached_at=datetime.now(tz=UTC)
)
_TECHNICAL = TechnicalResult(
    score=5.0,
    signal="neutral",
    block_scores=TechnicalBlockScores(
        momentum=5.0, trend=5.0, risk_stability=5.0, confirmation=5.0
    ),
    indicators={},
    llm_analysis="ok",
    calculated_at=datetime.now(tz=UTC),
)


@pytest.fixture(autouse=True)
def reset_limiter():
    """Clear in-memory rate-limit counters before each test."""
    limiter._storage.reset()
    yield


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_services():
    """Fast mocks for all deps so tests never hit real external services."""
    svc_s = MagicMock()
    svc_s.analyze = AsyncMock(return_value=_SENTIMENT)
    svc_d = MagicMock()
    svc_d.predict = AsyncMock(return_value=_DL)
    svc_f = MagicMock()
    svc_f.analyze = AsyncMock(return_value=_FUNDAMENTAL)
    svc_t = MagicMock()
    svc_t.analyze = AsyncMock(return_value=_TECHNICAL)
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value="ok")

    app.dependency_overrides[get_sentiment_service] = lambda: svc_s
    app.dependency_overrides[get_dl_service] = lambda: svc_d
    app.dependency_overrides[get_fundamental_service] = lambda: svc_f
    app.dependency_overrides[get_technical_service] = lambda: svc_t
    app.dependency_overrides[get_llm_service] = lambda: mock_llm
    yield svc_s, svc_d, svc_f, svc_t, mock_llm
    for fn in (
        get_sentiment_service,
        get_dl_service,
        get_fundamental_service,
        get_technical_service,
        get_llm_service,
    ):
        app.dependency_overrides.pop(fn, None)


# ── tests ─────────────────────────────────────────────────────────────────────


def test_report_allows_requests_within_limit(client, mock_services):
    """First 10 requests to /report return 200."""
    for _ in range(10):
        assert client.get("/api/v1/report/AAPL").status_code == 200


def test_report_returns_429_when_limit_exceeded(client, mock_services):
    """11th request to /report within the minute window returns 429."""
    for _ in range(10):
        client.get("/api/v1/report/AAPL")
    r = client.get("/api/v1/report/AAPL")
    assert r.status_code == 429
    assert "detail" in r.json()
    assert "Retry-After" in r.headers


def test_analysis_endpoint_returns_429_when_limit_exceeded(client, mock_services):
    """11th request to a module endpoint within the window returns 429."""
    for _ in range(10):
        client.get("/api/v1/sentiment/AAPL")
    assert client.get("/api/v1/sentiment/AAPL").status_code == 429


def test_rate_limits_are_independent_per_route(client, mock_services):
    """Exhausting the report limit does not affect the sentiment limit."""
    for _ in range(10):
        client.get("/api/v1/report/AAPL")
    # report exhausted — sentiment should still return 200
    assert client.get("/api/v1/sentiment/AAPL").status_code == 200
