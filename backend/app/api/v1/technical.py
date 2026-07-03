import re
from functools import lru_cache
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.core.config import get_settings
from app.core.rate_limit import limiter
from app.llm.factory import get_llm_service
from app.models.technical import TechnicalResult
from app.services.technical.engine.universe_manager import TechnicalUniverseManager
from app.services.technical.service import (
    TechnicalAnalysisError,
    TechnicalService,
    TechnicalUniverseNotReadyError,
)

router = APIRouter()

# International symbols carry exchange suffixes and class separators
_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,14}$")


def _validate_ticker(ticker: str) -> str:
    t = ticker.upper().strip()
    if not _TICKER_RE.match(t):
        raise HTTPException(status_code=422, detail="invalid ticker symbol")
    return t


@lru_cache
def get_technical_service() -> TechnicalService:
    """Return the singleton TechnicalService wired from settings."""
    settings = get_settings()
    data_dir = settings.technical_data_dir.strip()
    manager = TechnicalUniverseManager(data_dir=data_dir) if data_dir else None
    return TechnicalService(
        llm_service=get_llm_service(),
        cache_ttl=settings.cache_ttl_technical,
        manager=manager,
    )


@router.get("/technical/{ticker}", response_model=TechnicalResult)
@limiter.limit(lambda: get_settings().rate_limit_analysis)
async def get_technical(
    request: Request,
    ticker: str,
    mode: Literal["auto", "domestic", "global"] = Query(default="auto"),
    force_refresh: bool = Query(default=False),
    service: TechnicalService = Depends(get_technical_service),
) -> TechnicalResult:
    """Return the technical analysis for the given ticker.

    `mode` selects the reference universe: auto (S&P 500 if the ticker is in the index, otherwise
    MSCI World), domestic (S&P 500) or global (MSCI World).
    """
    ticker = _validate_ticker(ticker)
    try:
        return await service.analyze(ticker, mode=mode, force_refresh=force_refresh)
    except TechnicalUniverseNotReadyError as exc:
        raise HTTPException(
            status_code=503, detail="Technical universe is being prepared; try again shortly"
        ) from exc
    except TechnicalAnalysisError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
