import re
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.config import get_settings
from app.core.neo4j import get_neo4j_driver
from app.llm.factory import get_llm_service
from app.models.sentiment import SentimentResult
from app.services.sentiment.graph_retriever import GraphRetriever
from app.services.sentiment.news_client import NewsAPIClient, NewsAPIError, NewsAPIQuotaError
from app.services.sentiment.service import (
    NoRecentNewsError,
    SentimentAnalysisError,
    SentimentService,
)

router = APIRouter()

_TICKER_RE = re.compile(r"^[A-Z0-9]{2,5}$")


def _validate_ticker(ticker: str) -> str:
    t = ticker.upper().strip()
    if not _TICKER_RE.match(t):
        raise HTTPException(status_code=422, detail="ticker must be 2–5 alphanumeric characters")
    return t


@lru_cache
def get_sentiment_service() -> SentimentService:
    """Return the singleton SentimentService wired from settings."""
    settings = get_settings()
    return SentimentService(
        news_client=NewsAPIClient(
            api_key=settings.newsapi_key,
            max_articles=settings.max_news_articles,
        ),
        graph_retriever=GraphRetriever(
            driver=get_neo4j_driver(),
            database=settings.neo4j_database,
            hop_depth=settings.graph_hop_depth,
        ),
        llm_service=get_llm_service(),
        cache_ttl=settings.cache_ttl_sentiment,
    )


@router.get("/sentiment/{ticker}", response_model=SentimentResult)
async def get_sentiment(
    ticker: str,
    force_refresh: bool = Query(default=False),
    service: SentimentService = Depends(get_sentiment_service),
) -> SentimentResult:
    """Run the GraphRAG sentiment analysis for the given ticker."""
    ticker = _validate_ticker(ticker)
    try:
        return await service.analyze(ticker, force_refresh=force_refresh)
    except NewsAPIQuotaError as exc:
        raise HTTPException(
            status_code=503, detail="News provider daily quota exhausted; try again later"
        ) from exc
    except NoRecentNewsError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SentimentAnalysisError as exc:
        raise HTTPException(status_code=502, detail="Sentiment analysis failed") from exc
    except NewsAPIError as exc:
        raise HTTPException(status_code=502, detail="News provider request failed") from exc
