"""Tests for the sentiment analysis pipeline. All external services are mocked."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.sentiment import SentimentResult
from app.services.sentiment import service as service_mod
from app.services.sentiment.graph_retriever import RetrievedNews
from app.services.sentiment.news_client import NewsArticle
from app.services.sentiment.service import (
    NoRecentNewsError,
    SentimentAnalysisError,
    SentimentService,
)

_LLM_RESPONSE = json.dumps(
    {
        "label": "positivo",
        "score": 0.6,
        "confidence": 0.8,
        "explanation": "Las noticias recientes muestran una tendencia positiva.",
        "influential_news_indices": [0],
    }
)


def _article(i: int) -> NewsArticle:
    return NewsArticle(
        title=f"Title {i}",
        url=f"https://example.com/news/{i}",
        source=f"Source {i}",
        published_at="2026-06-10T12:00:00Z",
        description=f"Description {i}",
    )


def _retrieved() -> RetrievedNews:
    return RetrievedNews(
        text="Reference news about strong earnings",
        title="Reference title",
        similarity=0.9,
        mentions=[
            {"ticker": "AAPL", "name": "Apple", "sentiment_label": "Positive", "sentiment_score": 1}
        ],
        topics=["earnings"],
        events=[{"type": "earnings_report", "description": "Q3 beat", "date": "2026-06-01"}],
    )


def _make_service(
    articles: list[NewsArticle] | None = None,
    llm_responses: list[str] | None = None,
) -> SentimentService:
    news_client = MagicMock()
    news_client.fetch_news = AsyncMock(return_value=articles if articles is not None else [])

    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=[_retrieved()])

    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=llm_responses or [_LLM_RESPONSE])

    return SentimentService(
        news_client=news_client,
        graph_retriever=retriever,
        llm_service=llm,
        cache_ttl=1800,
    )


def _patch_embeddings():
    return patch.object(
        service_mod, "embed_texts", new=AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
    )


async def test_analyze_returns_sentiment_result():
    service = _make_service(articles=[_article(0), _article(1)])

    with _patch_embeddings():
        result = await service.analyze("AAPL")

    assert isinstance(result, SentimentResult)
    assert result.label == "positivo"
    assert result.score == 0.6
    assert result.confidence == 0.8
    assert "positiva" in result.explanation
    # influential_news mapped from LLM indices to the fresh articles
    assert len(result.influential_news) == 1
    assert result.influential_news[0].title == "Title 0"
    assert result.influential_news[0].url == "https://example.com/news/0"


async def test_analyze_prompt_includes_news_and_graph_context():
    service = _make_service(articles=[_article(0)])

    with _patch_embeddings():
        await service.analyze("AAPL")

    call = service._llm.complete.call_args
    user_prompt = call.kwargs.get("user_prompt") or call.args[1]
    assert "Title 0" in user_prompt
    assert "Reference news about strong earnings" in user_prompt
    assert "earnings" in user_prompt


async def test_analyze_caches_result_within_ttl():
    service = _make_service(articles=[_article(0)])

    with _patch_embeddings():
        first = await service.analyze("AAPL")
        second = await service.analyze("AAPL")

    assert first is second
    service._news_client.fetch_news.assert_awaited_once()
    service._llm.complete.assert_awaited_once()


async def test_force_refresh_bypasses_cache():
    service = _make_service(articles=[_article(0)], llm_responses=[_LLM_RESPONSE, _LLM_RESPONSE])

    with _patch_embeddings():
        await service.analyze("AAPL")
        await service.analyze("AAPL", force_refresh=True)

    assert service._news_client.fetch_news.await_count == 2
    assert service._llm.complete.await_count == 2


async def test_analyze_raises_when_no_news_found():
    service = _make_service(articles=[])

    with _patch_embeddings(), pytest.raises(NoRecentNewsError):
        await service.analyze("AAPL")


async def test_malformed_llm_output_retries_once_then_raises():
    service = _make_service(articles=[_article(0)], llm_responses=["not json", "still not json"])

    with _patch_embeddings(), pytest.raises(SentimentAnalysisError):
        await service.analyze("AAPL")

    assert service._llm.complete.await_count == 2


async def test_malformed_then_valid_llm_output_recovers():
    service = _make_service(articles=[_article(0)], llm_responses=["garbage", _LLM_RESPONSE])

    with _patch_embeddings():
        result = await service.analyze("AAPL")

    assert result.label == "positivo"
    assert service._llm.complete.await_count == 2


async def test_llm_output_with_code_fences_is_parsed():
    fenced = f"```json\n{_LLM_RESPONSE}\n```"
    service = _make_service(articles=[_article(0)], llm_responses=[fenced])

    with _patch_embeddings():
        result = await service.analyze("AAPL")

    assert result.label == "positivo"


async def test_invalid_indices_from_llm_are_ignored():
    response = json.dumps(
        {
            "label": "neutral",
            "score": 0.0,
            "confidence": 0.5,
            "explanation": "Sin señal clara.",
            "influential_news_indices": [0, 7, -3],
        }
    )
    service = _make_service(articles=[_article(0)], llm_responses=[response])

    with _patch_embeddings():
        result = await service.analyze("AAPL")

    assert [n.title for n in result.influential_news] == ["Title 0"]
