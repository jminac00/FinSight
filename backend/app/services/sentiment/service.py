"""Sentiment analysis pipeline: NewsAPI → embeddings → GraphRAG retrieval → LLM."""

import asyncio
import json
import logging

from cachetools import TTLCache
from pydantic import ValidationError

from app.llm.base import LLMService
from app.llm.embeddings import embed_texts
from app.models.sentiment import NewsItem, SentimentResult
from app.services.sentiment.graph_retriever import GraphRetriever, RetrievedNews
from app.services.sentiment.news.base import NewsArticle, NewsProvider
from app.services.sentiment.ticker_validator import TickerValidator

logger = logging.getLogger(__name__)

_CACHE_MAX_TICKERS = 256
_RETRIEVAL_TOP_K = 3
_MAX_CONTEXT_NEWS = 10
_MAX_LLM_ATTEMPTS = 2

_SYSTEM_PROMPT = """You are a financial sentiment analyst for retail investors.

You receive (a) recent news about a stock and (b) historical reference context \
retrieved from a financial knowledge graph (similar past news with human-annotated \
sentiment, related assets, topics and events). Weigh the recent news as the primary \
signal and use the historical context to calibrate it.

Respond ONLY with valid JSON in this exact format:
{
  "label": "positivo" | "negativo" | "neutral",
  "score": <float in [-1, 1], -1 most negative, 1 most positive>,
  "confidence": <float in [0, 1]>,
  "explanation": "<2-4 sentences, written in Spanish, plain language for non-experts>",
  "influential_news_indices": [<indices of the recent news items that most influenced \
your assessment>]
}

The explanation MUST be written in Spanish even though the input is in English.
Do not include markdown, code fences or any text outside the JSON object."""


class SentimentAnalysisError(Exception):
    """Raised when the pipeline cannot produce a valid sentiment result."""


class NoRecentNewsError(SentimentAnalysisError):
    """Raised when no recent news is found for the requested ticker."""


class UnknownTickerError(SentimentAnalysisError):
    """Raised when the ticker does not match any listed symbol."""


def _format_news(articles: list[NewsArticle]) -> str:
    lines = []
    for i, a in enumerate(articles):
        line = f"{i}. [{a.source}, {a.published_at}] {a.title}"
        if a.description:
            line += f" — {a.description}"
        lines.append(line)
    return "\n".join(lines)


def _format_graph_context(context: list[RetrievedNews]) -> str:
    if not context:
        return "(no similar historical news found)"
    lines = []
    for item in context:
        lines.append(f'- "{item.text or item.title}" (similarity {item.similarity:.2f})')
        if item.mentions:
            mentions = ", ".join(
                f"{m.get('name')} ({m.get('ticker')}): {m.get('sentiment_label')}"
                for m in item.mentions
            )
            lines.append(f"  Assets: {mentions}")
        if item.topics:
            lines.append(f"  Topics: {', '.join(item.topics)}")
        for event in item.events:
            date = f" ({event.get('date')})" if event.get("date") else ""
            lines.append(f"  Event: {event.get('type')} — {event.get('description')}{date}")
        if item.affected_assets:
            lines.append(f"  Also affects: {', '.join(item.affected_assets)}")
    return "\n".join(lines)


def _strip_code_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[: -len("```")]
    return text.strip()


class SentimentService:
    """Orchestrates the GraphRAG sentiment analysis pipeline.

    Flow: NewsAPI → semantic embeddings → Neo4j k-hop retrieval → LLM classification.
    Results are cached per ticker with a TTL to protect the NewsAPI daily quota.
    """

    def __init__(
        self,
        news_client: NewsProvider,
        graph_retriever: GraphRetriever,
        llm_service: LLMService,
        ticker_validator: TickerValidator,
        cache_ttl: int,
    ) -> None:
        """Args:
        news_client: Provider (or chain of providers) for fetching recent news.
        graph_retriever: GraphRAG retriever over the knowledge graph.
        llm_service: LLM provider used for the final classification.
        ticker_validator: Existence check for the requested ticker.
        cache_ttl: Seconds a result stays cached per ticker.
        """
        self._news_client = news_client
        self._graph_retriever = graph_retriever
        self._llm = llm_service
        self._ticker_validator = ticker_validator
        self._cache: TTLCache[str, SentimentResult] = TTLCache(
            maxsize=_CACHE_MAX_TICKERS, ttl=cache_ttl
        )

    async def analyze(self, ticker: str, force_refresh: bool = False) -> SentimentResult:
        """Run the full sentiment analysis for a stock ticker.

        Args:
            ticker: Uppercase stock symbol (e.g. 'AAPL').
            force_refresh: Skip the cached result and recompute.

        Returns:
            SentimentResult with label, score, confidence, explanation and
            the list of most influential news articles.

        Raises:
            UnknownTickerError: If the ticker does not match any listed symbol.
            NoRecentNewsError: If no recent news is found for the ticker.
            SentimentAnalysisError: If the LLM cannot produce a valid result.
            NewsQuotaError: If every news provider's quota is exhausted.
        """
        if not force_refresh and ticker in self._cache:
            logger.info("Sentiment cache hit for %s", ticker)
            return self._cache[ticker]

        if not await self._ticker_validator.exists(ticker):
            raise UnknownTickerError(f"Ticker {ticker} does not match any listed symbol")

        articles = await self._news_client.fetch_news(ticker)
        if not articles:
            raise NoRecentNewsError(f"No recent news found for ticker {ticker}")

        texts = [f"{a.title}. {a.description}".strip(". ") for a in articles]
        embeddings = await embed_texts(texts)

        context = await self._retrieve_context(embeddings)
        user_prompt = (
            f"Stock ticker: {ticker}\n\n"
            f"Recent news (numbered):\n{_format_news(articles)}\n\n"
            f"Historical reference context from the knowledge graph:\n"
            f"{_format_graph_context(context)}"
        )

        result = await self._classify(user_prompt, articles)
        self._cache[ticker] = result
        return result

    async def _retrieve_context(self, embeddings: list[list[float]]) -> list[RetrievedNews]:
        """Retrieve graph context per article and deduplicate by news text."""
        per_article = await asyncio.gather(
            *[self._graph_retriever.retrieve(e, top_k=_RETRIEVAL_TOP_K) for e in embeddings]
        )
        deduped: dict[str, RetrievedNews] = {}
        for item in (news for batch in per_article for news in batch):
            existing = deduped.get(item.text)
            if existing is None or item.similarity > existing.similarity:
                deduped[item.text] = item
        ranked = sorted(deduped.values(), key=lambda n: n.similarity, reverse=True)
        return ranked[:_MAX_CONTEXT_NEWS]

    async def _classify(self, user_prompt: str, articles: list[NewsArticle]) -> SentimentResult:
        """Call the LLM and parse its JSON answer, retrying once on malformed output."""
        last_error: Exception | None = None
        for attempt in range(1, _MAX_LLM_ATTEMPTS + 1):
            raw = await self._llm.complete(system_prompt=_SYSTEM_PROMPT, user_prompt=user_prompt)
            try:
                return self._parse_response(raw, articles)
            except (json.JSONDecodeError, KeyError, TypeError, ValidationError) as exc:
                logger.warning(
                    "LLM returned invalid sentiment payload (attempt %d/%d): %s",
                    attempt,
                    _MAX_LLM_ATTEMPTS,
                    exc,
                )
                last_error = exc
        raise SentimentAnalysisError("LLM did not return a valid sentiment result") from last_error

    @staticmethod
    def _parse_response(raw: str, articles: list[NewsArticle]) -> SentimentResult:
        data = json.loads(_strip_code_fences(raw))
        indices = data.get("influential_news_indices") or []
        influential = [
            NewsItem(title=articles[i].title, url=articles[i].url, source=articles[i].source)
            for i in indices
            if isinstance(i, int) and 0 <= i < len(articles)
        ]
        return SentimentResult(
            label=data["label"],
            score=data["score"],
            confidence=data["confidence"],
            explanation=data["explanation"],
            influential_news=influential,
        )
