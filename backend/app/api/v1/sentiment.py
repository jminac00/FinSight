import re

from fastapi import APIRouter, HTTPException, Query

from app.models.sentiment import NewsItem, SentimentResult

router = APIRouter()

_TICKER_RE = re.compile(r"^[A-Z0-9]{2,5}$")


def _validate_ticker(ticker: str) -> str:
    t = ticker.upper().strip()
    if not _TICKER_RE.match(t):
        raise HTTPException(status_code=422, detail="ticker must be 2–5 alphanumeric characters")
    return t


@router.get("/sentiment/{ticker}", response_model=SentimentResult)
async def get_sentiment(
    ticker: str,
    force_refresh: bool = Query(default=False),
) -> SentimentResult:
    """Return mock sentiment analysis for the given ticker."""
    ticker = _validate_ticker(ticker)
    return SentimentResult(
        label="positivo",
        score=0.65,
        confidence=0.82,
        explanation=(
            f"Las noticias recientes sobre {ticker} muestran una tendencia "
            "positiva impulsada por resultados trimestrales sólidos."
        ),
        influential_news=[
            NewsItem(
                title=f"{ticker} beats Q3 earnings expectations",
                url="https://example.com/news/1",
                source="Reuters",
            ),
            NewsItem(
                title=f"{ticker} announces expansion into new markets",
                url="https://example.com/news/2",
                source="Bloomberg",
            ),
        ],
    )
