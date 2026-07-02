import asyncio
import logging
import re
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.v1.deep_learning import get_dl_service
from app.api.v1.fundamental import get_fundamental_service
from app.api.v1.sentiment import get_sentiment_service
from app.api.v1.technical import get_technical_service
from app.llm.base import LLMService
from app.llm.factory import get_llm_service
from app.models.deep_learning import DLResult
from app.models.fundamental import FundamentalResult
from app.models.report import ReportResponse
from app.models.sentiment import SentimentResult
from app.models.technical import TechnicalResult
from app.services.deep_learning.service import DLService
from app.services.fundamental.service import FundamentalService
from app.services.sentiment.service import SentimentService
from app.services.technical.service import TechnicalService

router = APIRouter()
logger = logging.getLogger(__name__)

_TICKER_RE = re.compile(r"^[A-Z0-9]{2,5}$")

_DISCLAIMER = (
    "AVISO LEGAL: El contenido de este informe es de carácter exclusivamente informativo "
    "y ha sido generado mediante sistemas de inteligencia artificial. No constituye "
    "asesoramiento financiero, recomendación de inversión ni oferta de compra o venta "
    "de valores. Consulte siempre a un asesor financiero cualificado antes de tomar "
    "decisiones de inversión."
)

_FALLBACK_CONCLUSION = (
    "Se ha completado el análisis de los módulos disponibles. "
    "Consulte cada sección para obtener los detalles del análisis."
)

_SYNTHESIS_SYSTEM = (
    "You are a financial analyst writing a report summary for a retail investor. "
    "Write a concise objective summary in Spanish (3-5 sentences) that integrates "
    "the available analysis results. Highlight where modules agree or diverge. "
    "Never mention artificial intelligence, models, or automated systems. "
    "Do not give personalized investment advice."
)


def _validate_ticker(ticker: str) -> str:
    t = ticker.upper().strip()
    if not _TICKER_RE.match(t):
        raise HTTPException(status_code=422, detail="ticker must be 2–5 alphanumeric characters")
    return t


def _build_user_prompt(
    ticker: str,
    sentiment: SentimentResult | None,
    dl: DLResult | None,
    fundamental: FundamentalResult | None,
    technical: TechnicalResult | None,
) -> str:
    lines = [f"Ticker: {ticker}", ""]
    if sentiment:
        label_line = (
            f"  label={sentiment.label}, score={sentiment.score:.2f},"
            f" confidence={sentiment.confidence:.2f}"
        )
        lines += [
            "Sentiment analysis:",
            label_line,
            f"  {sentiment.explanation}",
            "",
        ]
    else:
        lines += ["Sentiment analysis: not available", ""]
    if dl:
        lines += [
            "Deep learning prediction (10-day):",
            f"  trend={dl.trend}, predicted_return={dl.predicted_return_pct:.2f}%",
            "",
        ]
    else:
        lines += ["Deep learning prediction: not available", ""]
    if fundamental:
        lines += [
            "Fundamental analysis:",
            f"  score={fundamental.score:.1f}/10",
            f"  {fundamental.llm_analysis}",
            "",
        ]
    else:
        lines += ["Fundamental analysis: not available", ""]
    if technical:
        lines += [
            "Technical analysis:",
            f"  score={technical.score:.1f}/10, signal={technical.signal}",
            f"  {technical.llm_analysis}",
            "",
        ]
    else:
        lines += ["Technical analysis: not available", ""]
    return "\n".join(lines)


@router.get("/report/{ticker}", response_model=ReportResponse)
async def get_report(
    ticker: str,
    force_refresh: bool = Query(default=False),
    sentiment_svc: SentimentService = Depends(get_sentiment_service),
    dl_svc: DLService = Depends(get_dl_service),
    fundamental_svc: FundamentalService = Depends(get_fundamental_service),
    technical_svc: TechnicalService = Depends(get_technical_service),
    llm: LLMService = Depends(get_llm_service),
) -> ReportResponse:
    """Orchestrate the 4 analysis modules in parallel and return the consolidated report."""
    ticker = _validate_ticker(ticker)

    results = await asyncio.gather(
        sentiment_svc.analyze(ticker, force_refresh=force_refresh),
        dl_svc.predict(ticker),
        fundamental_svc.analyze(ticker, mode="auto", force_refresh=force_refresh),
        technical_svc.analyze(ticker, mode="auto", force_refresh=force_refresh),
        return_exceptions=True,
    )

    sentiment = results[0] if not isinstance(results[0], BaseException) else None
    dl = results[1] if not isinstance(results[1], BaseException) else None
    fundamental = results[2] if not isinstance(results[2], BaseException) else None
    technical = results[3] if not isinstance(results[3], BaseException) else None

    for i, exc in enumerate(results):
        if isinstance(exc, BaseException):
            logger.warning("report %s: module %d failed: %s", ticker, i, exc)

    try:
        user_prompt = _build_user_prompt(ticker, sentiment, dl, fundamental, technical)
        global_conclusion = await llm.complete(_SYNTHESIS_SYSTEM, user_prompt)
    except Exception:
        logger.exception("report %s: LLM synthesis failed; using fallback", ticker)
        global_conclusion = _FALLBACK_CONCLUSION

    return ReportResponse(
        ticker=ticker,
        generated_at=datetime.now(tz=UTC),
        sentiment=sentiment,
        deep_learning=dl,
        fundamental=fundamental,
        technical=technical,
        global_conclusion=global_conclusion,
        disclaimer=_DISCLAIMER,
        partial_support=dl is None,
    )
