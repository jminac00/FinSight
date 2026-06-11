"""Managed Finnhub client shared across modules (news, and later DL/technical/fundamental)."""

import logging
from functools import lru_cache

import finnhub

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def get_finnhub_client() -> finnhub.Client:
    """Return the singleton Finnhub client configured from settings."""
    settings = get_settings()
    logger.info("Creating Finnhub client")
    return finnhub.Client(api_key=settings.finnhub_api_key)
