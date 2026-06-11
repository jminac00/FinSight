"""Tests for the Finnhub news provider. The finnhub client is fully mocked."""

from datetime import date, timedelta
from unittest.mock import MagicMock

import pytest
from finnhub import FinnhubAPIException

from app.services.sentiment.news.base import NewsProviderError, NewsQuotaError
from app.services.sentiment.news.finnhub_provider import FinnhubNewsProvider


def _finnhub_article(i: int) -> dict:
    return {
        "category": "company",
        "datetime": 1765400400 + i,  # unix seconds
        "headline": f"Headline {i}",
        "id": i,
        "image": "",
        "related": "AAPL",
        "source": f"Source {i}",
        "summary": f"Summary {i}",
        "url": f"https://example.com/news/{i}",
    }


def _api_exception(status_code: int) -> FinnhubAPIException:
    response = MagicMock(status_code=status_code)
    response.json.return_value = {"error": "API limit reached."}
    return FinnhubAPIException(response)


async def test_fetch_news_maps_finnhub_fields():
    client = MagicMock()
    client.company_news.return_value = [_finnhub_article(1)]
    provider = FinnhubNewsProvider(client=client, max_articles=10)

    articles = await provider.fetch_news("AAPL")

    assert len(articles) == 1
    article = articles[0]
    assert article.title == "Headline 1"
    assert article.url == "https://example.com/news/1"
    assert article.source == "Source 1"
    assert article.description == "Summary 1"
    assert article.published_at.startswith("2025-12-10")  # unix 1765400401 → ISO date


async def test_fetch_news_uses_seven_day_window():
    client = MagicMock()
    client.company_news.return_value = []
    provider = FinnhubNewsProvider(client=client, max_articles=10)

    await provider.fetch_news("AAPL")

    call = client.company_news.call_args
    assert call.args[0] == "AAPL"
    assert call.kwargs["to"] == date.today().isoformat()
    assert call.kwargs["_from"] == (date.today() - timedelta(days=7)).isoformat()


async def test_fetch_news_caps_articles_to_max():
    client = MagicMock()
    client.company_news.return_value = [_finnhub_article(i) for i in range(20)]
    provider = FinnhubNewsProvider(client=client, max_articles=5)

    articles = await provider.fetch_news("AAPL")

    assert len(articles) == 5


async def test_fetch_news_empty_results():
    client = MagicMock()
    client.company_news.return_value = []
    provider = FinnhubNewsProvider(client=client, max_articles=10)

    assert await provider.fetch_news("ASML") == []


async def test_rate_limit_raises_quota_error():
    client = MagicMock()
    client.company_news.side_effect = _api_exception(429)
    provider = FinnhubNewsProvider(client=client, max_articles=10)

    with pytest.raises(NewsQuotaError):
        await provider.fetch_news("AAPL")


async def test_api_error_raises_provider_error():
    client = MagicMock()
    client.company_news.side_effect = _api_exception(500)
    provider = FinnhubNewsProvider(client=client, max_articles=10)

    with pytest.raises(NewsProviderError):
        await provider.fetch_news("AAPL")


async def test_network_error_raises_provider_error():
    client = MagicMock()
    client.company_news.side_effect = ConnectionError("connection refused")
    provider = FinnhubNewsProvider(client=client, max_articles=10)

    with pytest.raises(NewsProviderError):
        await provider.fetch_news("AAPL")
