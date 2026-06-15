import re
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.config import get_settings
from app.core.finnhub import get_finnhub_client
from app.llm.factory import get_llm_service
from app.models.fundamental import FundamentalResult
from app.services.fundamental.service import (
    FundamentalAnalysisError,
    FundamentalService,
    UniverseNotReadyError,
    UniverseProvider,
    UnknownTickerError,
)
from app.services.sentiment.ticker_validator import TickerValidator

router = APIRouter()

_TICKER_RE = re.compile(r"^[A-Z0-9]{2,5}$")


def _validate_ticker(ticker: str) -> str:
    t = ticker.upper().strip()
    if not _TICKER_RE.match(t):
        raise HTTPException(status_code=422, detail="ticker must be 2–5 alphanumeric characters")
    return t


@lru_cache
def get_fundamental_service() -> FundamentalService:
    """Return the singleton FundamentalService wired from settings."""
    settings = get_settings()
    path = Path(settings.fundamental_universe_path) if settings.fundamental_universe_path else None
    return FundamentalService(
        universe_provider=UniverseProvider(path),
        llm_service=get_llm_service(),
        ticker_validator=TickerValidator(client=get_finnhub_client()),
        cache_ttl=settings.cache_ttl_fundamental,
    )


@router.get("/fundamental/{ticker}", response_model=FundamentalResult)
async def get_fundamental(
    ticker: str,
    force_refresh: bool = Query(default=False),
    service: FundamentalService = Depends(get_fundamental_service),
) -> FundamentalResult:
    """Return the fundamental analysis for the given ticker."""
    ticker = _validate_ticker(ticker)
    try:
        return await service.analyze(ticker, force_refresh=force_refresh)
    except UnknownTickerError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except UniverseNotReadyError as exc:
        raise HTTPException(
            status_code=503, detail="Fundamental universe is being prepared; try again shortly"
        ) from exc
    except FundamentalAnalysisError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
