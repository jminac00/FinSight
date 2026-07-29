import re
from datetime import datetime
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.config import Settings, get_settings
from app.core.rate_limit import limiter
from app.models.deep_learning import DLResult, DLTrainResult, DLUnavailable, ModelMetrics
from app.services.deep_learning.artifacts import DEFAULT_MODELS_DIR
from app.services.deep_learning.preprocessing import InsufficientHistoryError
from app.services.deep_learning.service import DLService, DLUnavailableError

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
def get_dl_service() -> DLService:
    """Return the singleton DLService wired from settings.

    On-demand training is enabled outside production only: locally the machine
    can train a missing model, while Render's free tier serves pre-trained
    artifacts exclusively.
    """
    settings = get_settings()
    return DLService(
        models_dir=DEFAULT_MODELS_DIR,
        max_models=settings.lru_cache_max_models,
        auto_train=settings.environment != "production",
    )


@router.get(
    "/prediction/{ticker}",
    response_model=DLResult,
    responses={404: {"model": DLUnavailable, "description": "No prediction available"}},
)
@limiter.limit(lambda: get_settings().rate_limit_analysis)
async def get_prediction(
    request: Request,
    ticker: str,
    service: DLService = Depends(get_dl_service),
) -> DLResult:
    """Return a GRU-based 10-day return prediction for the given ticker.

    When no prediction can be offered the response is a 404 whose ``detail``
    carries both the machine-readable reason and a human message.
    """
    ticker = _validate_ticker(ticker)
    try:
        return await service.predict(ticker)
    except DLUnavailableError as exc:
        raise HTTPException(
            status_code=404,
            detail=DLUnavailable(reason=exc.reason, message=str(exc)).model_dump(),
        ) from exc
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
