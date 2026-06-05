"""Batch embedding generation via the OpenAI API."""

import asyncio
import logging
import time
from typing import List

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_BATCH_SIZE = 512
_MAX_RETRIES = 4
_RETRY_BASE_DELAY = 2.0  # seconds


async def embed_texts(
    texts: List[str],
    api_key: str,
    model: str,
) -> List[List[float]]:
    """Return embeddings for all texts, processed in batches."""
    client = AsyncOpenAI(api_key=api_key)
    results: List[List[float]] = []

    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        results.extend(await _embed_batch(client, batch, model))
        logger.debug("Embedded %d/%d texts", min(i + _BATCH_SIZE, len(texts)), len(texts))

    return results


async def _embed_batch(
    client: AsyncOpenAI,
    texts: List[str],
    model: str,
) -> List[List[float]]:
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
