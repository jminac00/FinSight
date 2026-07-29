import asyncio
import logging
import re
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.api.v1.deep_learning import get_dl_service
from app.api.v1.fundamental import get_fundamental_service
from app.api.v1.search import get_symbol_search_service
from app.api.v1.sentiment import get_sentiment_service
from app.api.v1.technical import get_technical_service
from app.core.config import get_settings
from app.core.rate_limit import limiter
from app.llm.base import LLMService
from app.llm.factory import get_llm_service
from app.models.deep_learning import DLResult
from app.models.fundamental import FundamentalResult
from app.models.report import ReportResponse
from app.models.sentiment import SentimentResult
from app.models.technical import TechnicalResult
from app.services.deep_learning.service import DLService, DLUnavailableError
from app.services.fundamental.service import FundamentalService
from app.services.search.service import SymbolSearchService
from app.services.sentiment.service import SentimentService
from app.services.technical.service import TechnicalService

router = APIRouter()
logger = logging.getLogger(__name__)

# International symbols carry exchange suffixes and class separators (e.g. REP.MC,
# ASML.AS, 7203.T, BRK-B), so the US-only 2-5 alphanumeric rule (CLAUDE.md §3.8) is
# relaxed here for the universal (MSCI World) coverage.
_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,14}$")

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
    "When a module states a reason for being unavailable, explain that reason "
    "briefly in Spanish so the reader knows why the analysis is missing. "
    "Never mention artificial intelligence or automated systems; you may refer to "
    "the trend prediction model only when explaining why that module is missing. "
    "Do not give personalized investment advice."
)

# English rationales handed to the LLM, which renders them in Spanish. The
# insufficient-quality wording must convey that a model exists but underperforms.
_DL_UNAVAILABLE_REASONS: dict[str, str] = {
    "out_of_coverage": (
        "the ticker is outside the coverage of the trend prediction module, "
        "which only covers S&P 500 constituents"
    ),
    "not_trained": (
        "the ticker is within coverage but its trend prediction model has not been trained yet"
    ),
    "insufficient_quality": (
        "a trend prediction model exists for this ticker, but its predictive "
        "performance does not reach the minimum required, so the prediction "
        "is deliberately not offered"
    ),
}


def _validate_ticker(ticker: str) -> str:
    t = ticker.upper().strip()
    if not _TICKER_RE.match(t):
        raise HTTPException(status_code=422, detail="invalid ticker symbol")
    return t


def _build_user_prompt(
    ticker: str,
    sentiment: SentimentResult | None,
    dl: DLResult | None,
    fundamental: FundamentalResult | None,
    technical: TechnicalResult | None,
    dl_reason: str | None = None,
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
    elif dl_reason:
        lines += [
            f"Deep learning prediction: not available [{dl_reason}]",
            f"  Reason: {_DL_UNAVAILABLE_REASONS[dl_reason]}.",
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
@limiter.limit(lambda: get_settings().rate_limit_report)
async def get_report(
    request: Request,
    ticker: str,
    force_refresh: bool = Query(default=False),
    sentiment_svc: SentimentService = Depends(get_sentiment_service),
    dl_svc: DLService = Depends(get_dl_service),
    fundamental_svc: FundamentalService = Depends(get_fundamental_service),
    technical_svc: TechnicalService = Depends(get_technical_service),
    search_svc: SymbolSearchService = Depends(get_symbol_search_service),
    llm: LLMService = Depends(get_llm_service),
) -> ReportResponse:
    """Orchestrate the 4 analysis modules in parallel and return the consolidated report."""
    ticker = _validate_ticker(ticker)

    task_names = ("sentiment", "deep_learning", "fundamental", "technical", "search")
    results = await asyncio.gather(
        sentiment_svc.analyze(ticker, force_refresh=force_refresh),
        dl_svc.predict(ticker),
        fundamental_svc.analyze(ticker, mode="auto", force_refresh=force_refresh),
        technical_svc.analyze(ticker, mode="auto", force_refresh=force_refresh),
        search_svc.search(ticker),
        return_exceptions=True,
    )

    sentiment = results[0] if not isinstance(results[0], BaseException) else None
    dl = results[1] if not isinstance(results[1], BaseException) else None
    # gather() flattens every failure into "module missing"; the deep learning
    # module knows *why* it has nothing to offer, so keep that reason.
    dl_reason = results[1].reason if isinstance(results[1], DLUnavailableError) else None
    fundamental = results[2] if not isinstance(results[2], BaseException) else None
    technical = results[3] if not isinstance(results[3], BaseException) else None
    matches = results[4] if not isinstance(results[4], BaseException) else []

    company_name = next(
        (m.description for m in matches if m.symbol == ticker and m.description), None
    )

    for name, result in zip(task_names, results, strict=True):
        if isinstance(result, BaseException):
            logger.warning("report %s: %s failed: %s", ticker, name, result)

    missing_modules = [
        name
        for name, value in (
            ("sentiment", sentiment),
            ("deep_learning", dl),
            ("fundamental", fundamental),
            ("technical", technical),
        )
        if value is None
    ]

    try:
        user_prompt = _build_user_prompt(ticker, sentiment, dl, fundamental, technical, dl_reason)
        global_conclusion = await llm.complete(_SYNTHESIS_SYSTEM, user_prompt)
    except Exception:
        logger.exception("report %s: LLM synthesis failed; using fallback", ticker)
        global_conclusion = _FALLBACK_CONCLUSION

    return ReportResponse(
        ticker=ticker,
        company_name=company_name,
        generated_at=datetime.now(tz=UTC),
        sentiment=sentiment,
        deep_learning=dl,
        deep_learning_unavailable_reason=dl_reason,
        fundamental=fundamental,
        technical=technical,
        global_conclusion=global_conclusion,
        disclaimer=_DISCLAIMER,
        partial_support=bool(missing_modules),
        missing_modules=missing_modules,
    )
