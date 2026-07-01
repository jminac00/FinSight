import re
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException

from app.core.config import get_settings
from app.models.deep_learning import DLResult
from app.services.deep_learning.artifacts import DEFAULT_MODELS_DIR
from app.services.deep_learning.preprocessing import InsufficientHistoryError
from app.services.deep_learning.service import DLService, ModelNotAvailableError

router = APIRouter()

_TICKER_RE = re.compile(r"^[A-Z0-9]{2,5}$")


def _validate_ticker(ticker: str) -> str:
    t = ticker.upper().strip()
    if not _TICKER_RE.match(t):
        raise HTTPException(status_code=422, detail="ticker must be 2–5 alphanumeric characters")
    return t


@lru_cache
def get_dl_service() -> DLService:
    """Return the singleton DLService wired from settings."""
    settings = get_settings()
    return DLService(models_dir=DEFAULT_MODELS_DIR, max_models=settings.lru_cache_max_models)


@router.get("/prediction/{ticker}", response_model=DLResult)
async def get_prediction(
    ticker: str,
    service: DLService = Depends(get_dl_service),
) -> DLResult:
    """Return a GRU-based 10-day return prediction for the given ticker."""
    ticker = _validate_ticker(ticker)
    try:
        return await service.predict(ticker)
    except ModelNotAvailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (InsufficientHistoryError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
