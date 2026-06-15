import re
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException

from app.models.deep_learning import DLResult, ModelMetrics

router = APIRouter()

_TICKER_RE = re.compile(r"^[A-Z0-9]{2,5}$")


def _validate_ticker(ticker: str) -> str:
    t = ticker.upper().strip()
    if not _TICKER_RE.match(t):
        raise HTTPException(status_code=422, detail="ticker must be 2–5 alphanumeric characters")
    return t


@router.get("/prediction/{ticker}", response_model=DLResult)
async def get_prediction(ticker: str) -> DLResult:
    """Return mock GRU prediction for the given ticker."""
    ticker = _validate_ticker(ticker)
    return DLResult(
        trend="alcista",
        predicted_return_pct=7.24,
        predicted_price=195.50,
        current_price=182.30,
        horizon_days=10,
        trained_at=datetime(2026, 6, 3, 22, 0, 0, tzinfo=UTC),
        metrics=ModelMetrics(rmse=5.61, mae=4.39, directional_accuracy=0.60),
    )
