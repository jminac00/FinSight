import asyncio
import re
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from app.models.deep_learning import DLResult, ModelMetrics
from app.models.fundamental import FundamentalResult
from app.models.report import ReportResponse
from app.models.sentiment import NewsItem, SentimentResult
from app.models.technical import TechnicalResult

router = APIRouter()

_TICKER_RE = re.compile(r"^[A-Z0-9]{2,5}$")

_DISCLAIMER = (
    "AVISO LEGAL: El contenido de este informe es de carácter exclusivamente informativo "
    "y ha sido generado mediante sistemas de inteligencia artificial. No constituye "
    "asesoramiento financiero, recomendación de inversión ni oferta de compra o venta "
    "de valores. Consulte siempre a un asesor financiero cualificado antes de tomar "
    "decisiones de inversión."
)


def _validate_ticker(ticker: str) -> str:
    t = ticker.upper().strip()
    if not _TICKER_RE.match(t):
        raise HTTPException(status_code=422, detail="ticker must be 2–5 alphanumeric characters")
    return t


async def _mock_sentiment(ticker: str) -> SentimentResult:
    await asyncio.sleep(0)
    return SentimentResult(
        label="positivo",
        score=0.65,
        confidence=0.82,
        explanation=f"Las noticias recientes sobre {ticker} muestran una tendencia positiva.",
        influential_news=[
            NewsItem(title=f"{ticker} beats Q3 earnings", url="https://example.com/1", source="Reuters"),
        ],
    )


async def _mock_deep_learning(ticker: str) -> DLResult:
    await asyncio.sleep(0)
    return DLResult(
        trend="alcista",
        predicted_price=195.50,
        current_price=182.30,
        pct_change=7.24,
        horizon_days=10,
        trained_at=datetime(2026, 6, 3, 22, 0, 0, tzinfo=timezone.utc),
        metrics=ModelMetrics(rmse=3.12, mae=2.45, mape=1.35, r2=0.92),
    )


async def _mock_fundamental(ticker: str) -> FundamentalResult:
    await asyncio.sleep(0)
    return FundamentalResult(
        score=7.8,
        metrics={"per": 28.5, "roe": 0.312, "ev_ebitda": 21.3},
        llm_analysis=f"{ticker} presenta una situación financiera sólida con márgenes superiores a la media sectorial.",
        cached_at=datetime.now(tz=timezone.utc),
    )


async def _mock_technical(ticker: str) -> TechnicalResult:
    await asyncio.sleep(0)
    return TechnicalResult(
        score=6.5,
        indicators={"rsi_14": 58.3, "macd": 1.24, "sma_50": 175.80},
        llm_analysis=f"El análisis técnico de {ticker} muestra señales moderadamente alcistas.",
        calculated_at=datetime.now(tz=timezone.utc),
    )


@router.get("/report/{ticker}", response_model=ReportResponse)
async def get_report(
    ticker: str,
    force_refresh: bool = Query(default=False),
) -> ReportResponse:
    """Orchestrate the 4 analysis modules in parallel and return the consolidated report."""
    ticker = _validate_ticker(ticker)

    sentiment, dl, fundamental, technical = await asyncio.gather(
        _mock_sentiment(ticker),
        _mock_deep_learning(ticker),
        _mock_fundamental(ticker),
        _mock_technical(ticker),
    )

    # TODO: replace with real LLM call via get_llm_service().complete(...)
    global_conclusion = (
        f"El análisis consolidado de {ticker} refleja una perspectiva generalmente positiva. "
        "El sentimiento de mercado es favorable, respaldado por resultados sólidos. "
        "El modelo LSTM predice una tendencia alcista a 10 días. "
        "Los fundamentales son robustos y el análisis técnico confirma el momentum positivo. "
        "Sin embargo, recuerde que este análisis es informativo y no constituye recomendación de inversión."
    )

    return ReportResponse(
        ticker=ticker,
        generated_at=datetime.now(tz=timezone.utc),
        sentiment=sentiment,
        deep_learning=dl,
        fundamental=fundamental,
        technical=technical,
        global_conclusion=global_conclusion,
        disclaimer=_DISCLAIMER,
        partial_support=False,
    )
