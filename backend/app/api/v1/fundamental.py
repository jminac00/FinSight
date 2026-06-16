import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.config import get_settings
from app.llm.factory import get_llm_service
from app.models.fundamental import FundamentalResult
from app.services.fundamental.engine.universe_manager import UniverseManager
from app.services.fundamental.service import (
    FundamentalAnalysisError,
    FundamentalService,
    UniverseNotReadyError,
)

router = APIRouter()

# International symbols carry exchange suffixes and class separators (e.g. ASML.AS,
# NOVO-B.CO, 7203.T, BRK-B), so the US-only 2-5 alphanumeric rule (CLAUDE.md §3.8) is
# relaxed here for the universal (MSCI World) coverage.
_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,14}$")


def _validate_ticker(ticker: str) -> str:
    t = ticker.upper().strip()
    if not _TICKER_RE.match(t):
        raise HTTPException(status_code=422, detail="invalid ticker symbol")
    return t


@lru_cache
def get_fundamental_service() -> FundamentalService:
    """Return the singleton FundamentalService wired from settings."""
    settings = get_settings()
    data_dir = settings.fundamental_data_dir.strip()
    manager = UniverseManager(data_dir=Path(data_dir)) if data_dir else None
    return FundamentalService(
        llm_service=get_llm_service(),
        cache_ttl=settings.cache_ttl_fundamental,
        manager=manager,
    )


@router.get("/fundamental/{ticker}", response_model=FundamentalResult)
async def get_fundamental(
    ticker: str,
    mode: Literal["auto", "domestic", "global"] = Query(default="auto"),
    force_refresh: bool = Query(default=False),
    service: FundamentalService = Depends(get_fundamental_service),
) -> FundamentalResult:
    """Return the fundamental analysis for the given ticker.

    `mode` selects the normalization universe: auto (S&P 500 if the ticker is in the index,
    otherwise MSCI World), domestic (S&P 500) or global (MSCI World).
    """
    ticker = _validate_ticker(ticker)
    try:
        return await service.analyze(ticker, mode=mode, force_refresh=force_refresh)
    except UniverseNotReadyError as exc:
        raise HTTPException(
            status_code=503, detail="Fundamental universe is being prepared; try again shortly"
        ) from exc
    except FundamentalAnalysisError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
