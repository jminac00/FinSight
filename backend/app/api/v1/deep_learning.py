import re
from datetime import datetime
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException

from app.core.config import Settings, get_settings
from app.models.deep_learning import DLResult, DLTrainResult, ModelMetrics
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


@router.post("/train/{ticker}", response_model=DLTrainResult)
async def train_model(
    ticker: str,
    service: DLService = Depends(get_dl_service),
    settings: Settings = Depends(get_settings),
) -> DLTrainResult:
    """On-demand GRU retraining — development environment only."""
    if settings.environment == "production":
        raise HTTPException(status_code=403, detail="Training endpoint is disabled in production")
    ticker = _validate_ticker(ticker)
    try:
        artifacts = await service.train(ticker)
    except (InsufficientHistoryError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return DLTrainResult(
        ticker=ticker,
        trained_at=datetime.fromisoformat(artifacts.metadata.trained_at),
        metrics=ModelMetrics(
            rmse=artifacts.metadata.metrics["rmse"],
            mae=artifacts.metadata.metrics["mae"],
            directional_accuracy=artifacts.metadata.metrics["directional_accuracy"],
        ),
        data_through=artifacts.metadata.data_through,
    )
