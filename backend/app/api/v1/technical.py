import re
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.models.technical import TechnicalResult

router = APIRouter()

_TICKER_RE = re.compile(r"^[A-Z0-9]{2,5}$")


def _validate_ticker(ticker: str) -> str:
    t = ticker.upper().strip()
    if not _TICKER_RE.match(t):
        raise HTTPException(status_code=422, detail="ticker must be 2–5 alphanumeric characters")
    return t


@router.get("/technical/{ticker}", response_model=TechnicalResult)
async def get_technical(ticker: str) -> TechnicalResult:
    """Return mock technical analysis for the given ticker."""
    ticker = _validate_ticker(ticker)
    return TechnicalResult(
        score=6.5,
        indicators={
            "rsi_14": 58.3,
            "macd": 1.24,
            "macd_signal": 0.98,
            "macd_histogram": 0.26,
            "bb_upper": 194.20,
            "bb_middle": 182.30,
            "bb_lower": 170.40,
            "sma_20": 180.15,
            "sma_50": 175.80,
            "ema_20": 181.40,
        },
        llm_analysis=(
            f"El análisis técnico de {ticker} muestra señales moderadamente alcistas. "
            "El RSI en 58 indica momentum positivo sin estar sobrecomprado. "
            "El MACD presenta cruce alcista reciente por encima de la señal, "
            "confirmando la tendencia a corto plazo."
        ),
        calculated_at=datetime.now(tz=timezone.utc),
    )
