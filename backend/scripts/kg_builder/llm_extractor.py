"""Extract events and topics from news text using the LLM service."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from .topics import EVENT_TYPES, TOPICS

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = f"""You are a financial news analyser. Given a news text, extract:
1. Financial events described (if any). Each event must have:
   - type: one of {EVENT_TYPES}
   - description: one concise sentence
   - date: ISO date string if determinable, otherwise null
2. Relevant topics from this controlled list: {TOPICS}

Respond ONLY with valid JSON in this exact format:
{{
  "events": [{{"type": "string", "description": "string", "date": "YYYY-MM-DD or null"}}],
  "topics": ["string"]
}}
If no events are found, use an empty list. Topics must only contain values from the controlled list."""


@dataclass
class ExtractedEvent:
    type: str
    description: str
    date: Optional[str]


@dataclass
class ExtractionResult:
    events: list[ExtractedEvent] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)


async def extract(text: str, llm_service) -> ExtractionResult:
    """Run a single LLM call to extract events and topics from news text."""
    try:
        raw = await llm_service.complete(system_prompt=_SYSTEM_PROMPT, user_prompt=text)
        data = json.loads(raw)
        events = [
            ExtractedEvent(
                type=e.get("type", "other"),
                description=e.get("description", ""),
                date=e.get("date"),
            )
            for e in data.get("events", [])
            if e.get("type") in EVENT_TYPES
        ]
        topics = [t for t in data.get("topics", []) if t in TOPICS]
        return ExtractionResult(events=events, topics=topics)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("LLM extraction failed (%s) — skipping LLM nodes for this record.", exc)
        return ExtractionResult()


async def extract_batch(texts: list[str], llm_service, concurrency: int = 5) -> list[ExtractionResult]:
    """Extract events and topics for a list of texts with bounded concurrency."""
    semaphore = asyncio.Semaphore(concurrency)

    async def _run(text: str) -> ExtractionResult:
        async with semaphore:
            return await extract(text, llm_service)

    return await asyncio.gather(*[_run(t) for t in texts])
