"""Contract tests for the sentiment endpoint. The sentiment service is mocked."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.api.v1.sentiment import get_sentiment_service
from app.main import app
from app.models.sentiment import NewsItem, SentimentResult
from app.services.sentiment.news_client import NewsAPIQuotaError
from app.services.sentiment.service import NoRecentNewsError, SentimentAnalysisError

_RESULT = SentimentResult(
    label="positivo",
    score=0.6,
    confidence=0.8,
    explanation="Tendencia positiva.",
    influential_news=[
        NewsItem(title="Title 0", url="https://example.com/news/0", source="Source 0")
    ],
)


@pytest.fixture
def mock_service():
    service = MagicMock()
    service.analyze = AsyncMock(return_value=_RESULT)
    app.dependency_overrides[get_sentiment_service] = lambda: service
    yield service
    app.dependency_overrides.pop(get_sentiment_service, None)


def test_get_sentiment_returns_analysis(client, mock_service):
    response = client.get("/api/v1/sentiment/AAPL")

    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "positivo"
    assert body["score"] == 0.6
    assert body["influential_news"][0]["url"] == "https://example.com/news/0"
    mock_service.analyze.assert_awaited_once_with("AAPL", force_refresh=False)


def test_get_sentiment_passes_force_refresh(client, mock_service):
    response = client.get("/api/v1/sentiment/AAPL?force_refresh=true")

    assert response.status_code == 200
    mock_service.analyze.assert_awaited_once_with("AAPL", force_refresh=True)


def test_get_sentiment_lowercase_ticker_is_normalised(client, mock_service):
    response = client.get("/api/v1/sentiment/aapl")

    assert response.status_code == 200
    mock_service.analyze.assert_awaited_once_with("AAPL", force_refresh=False)


def test_get_sentiment_invalid_ticker_returns_422(client, mock_service):
    response = client.get("/api/v1/sentiment/TOOLONG99")

    assert response.status_code == 422
    mock_service.analyze.assert_not_awaited()


def test_get_sentiment_quota_exhausted_returns_503(client, mock_service):
    mock_service.analyze.side_effect = NewsAPIQuotaError("quota exhausted")

    response = client.get("/api/v1/sentiment/AAPL")

    assert response.status_code == 503


def test_get_sentiment_no_news_returns_404(client, mock_service):
    mock_service.analyze.side_effect = NoRecentNewsError("no recent news for AAPL")

    response = client.get("/api/v1/sentiment/AAPL")

    assert response.status_code == 404


def test_get_sentiment_analysis_failure_returns_502(client, mock_service):
    mock_service.analyze.side_effect = SentimentAnalysisError("LLM returned invalid JSON")

    response = client.get("/api/v1/sentiment/AAPL")

    assert response.status_code == 502
