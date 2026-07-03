import re
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.core.config import get_settings
from app.core.finnhub import get_finnhub_client
from app.core.neo4j import get_neo4j_driver
from app.core.rate_limit import limiter
from app.llm.factory import get_llm_service
from app.models.sentiment import SentimentResult
from app.services.sentiment.graph_retriever import GraphRetriever
from app.services.sentiment.news.base import NewsProviderError, NewsQuotaError
from app.services.sentiment.news.factory import get_news_provider
from app.services.sentiment.service import (
    NoRecentNewsError,
    SentimentAnalysisError,
    SentimentService,
    UnknownTickerError,
)
from app.services.sentiment.ticker_validator import TickerValidator

router = APIRouter()

# International symbols carry exchange suffixes and class separators (e.g. REP.MC,
# ASML.AS, 7203.T, BRK-B), so the US-only 2-5 alphanumeric rule (CLAUDE.md §3.8) is
# relaxed here for the universal (MSCI World) coverage.
_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,14}$")


def _validate_ticker(ticker: str) -> str:
    t = ticker.upper().strip()
    if not _TICKER_RE.match(t):
        raise HTTPException(status_code=422, detail="invalid ticker symbol")
    return t


@lru_cache
def get_sentiment_service() -> SentimentService:
    """Return the singleton SentimentService wired from settings."""
    settings = get_settings()
    return SentimentService(
        news_client=get_news_provider(),
        graph_retriever=GraphRetriever(
            driver=get_neo4j_driver(),
            database=settings.neo4j_database,
            hop_depth=settings.graph_hop_depth,
        ),
        llm_service=get_llm_service(),
        ticker_validator=TickerValidator(client=get_finnhub_client()),
        cache_ttl=settings.cache_ttl_sentiment,
    )


@router.get("/sentiment/{ticker}", response_model=SentimentResult)
@limiter.limit(lambda: get_settings().rate_limit_analysis)
async def get_sentiment(
    request: Request,
    ticker: str,
    force_refresh: bool = Query(default=False),
    service: SentimentService = Depends(get_sentiment_service),
) -> SentimentResult:
    """Run the GraphRAG sentiment analysis for the given ticker."""
    ticker = _validate_ticker(ticker)
    try:
        return await service.analyze(ticker, force_refresh=force_refresh)
    except NewsQuotaError as exc:
        raise HTTPException(
            status_code=503, detail="News providers' quota exhausted; try again later"
        ) from exc
    except UnknownTickerError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except NoRecentNewsError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SentimentAnalysisError as exc:
        raise HTTPException(status_code=502, detail="Sentiment analysis failed") from exc
    except NewsProviderError as exc:
        raise HTTPException(status_code=502, detail="News provider request failed") from exc
