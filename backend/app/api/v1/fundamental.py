import re
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException

from app.models.fundamental import FundamentalResult

router = APIRouter()

_TICKER_RE = re.compile(r"^[A-Z0-9]{2,5}$")


def _validate_ticker(ticker: str) -> str:
    t = ticker.upper().strip()
    if not _TICKER_RE.match(t):
        raise HTTPException(status_code=422, detail="ticker must be 2–5 alphanumeric characters")
    return t


@router.get("/fundamental/{ticker}", response_model=FundamentalResult)
async def get_fundamental(ticker: str) -> FundamentalResult:
    """Return mock fundamental analysis for the given ticker."""
    ticker = _validate_ticker(ticker)
    return FundamentalResult(
        score=7.8,
        metrics={
            "per": 28.5,
            "roe": 0.312,
            "ev_ebitda": 21.3,
            "margen_operativo": 0.295,
            "margen_neto": 0.253,
            "deuda_capital": 1.74,
            "flujo_caja_libre": 89_500_000_000,
        },
        llm_analysis=(
            f"{ticker} presenta una situación financiera sólida con márgenes operativos "
            "superiores a la media del sector. El flujo de caja libre elevado proporciona "
            "capacidad para recompras y dividendos. El ratio PER refleja la prima de "
            "crecimiento que el mercado asigna a la empresa."
        ),
        cached_at=datetime.now(tz=UTC),
    )
