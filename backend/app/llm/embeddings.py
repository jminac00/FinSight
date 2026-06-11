"""Async embedding generation via the OpenAI API.

Shared by the runtime sentiment pipeline and the offline knowledge-graph builder.
Model and API key come from application settings (never hardcoded).
"""

import asyncio
import logging

from openai import AsyncOpenAI

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_BATCH_SIZE = 512
_MAX_RETRIES = 4
_RETRY_BASE_DELAY = 2.0  # seconds


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return embeddings for all texts, processed in batches with retries.

    Args:
        texts: Texts to embed, in order.

    Returns:
        One embedding vector per input text, in the same order.
    """
    if not texts:
        return []

    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    model = settings.openai_embedding_model
    results: list[list[float]] = []

    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        results.extend(await _embed_batch(client, batch, model))
        logger.debug("Embedded %d/%d texts", min(i + _BATCH_SIZE, len(texts)), len(texts))

    return results


async def _embed_batch(
    client: AsyncOpenAI,
    texts: list[str],
    model: str,
) -> list[list[float]]:
    for attempt in range(_MAX_RETRIES):
        try:
            response = await client.embeddings.create(input=texts, model=model)
            return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        except Exception as exc:
            if attempt == _MAX_RETRIES - 1:
                raise
            delay = _RETRY_BASE_DELAY * (2**attempt)
            logger.warning("Embedding batch failed (%s), retrying in %.1fs…", exc, delay)
            await asyncio.sleep(delay)
    return []  # unreachable
