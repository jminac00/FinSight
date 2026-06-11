"""Managed Neo4j driver for the API runtime (created once, closed on app shutdown).

The neo4j-graphrag retrievers require a synchronous neo4j.Driver, so this module
exposes a sync driver; callers run blocking searches in a thread.
"""

import logging
from functools import lru_cache

from neo4j import Driver, GraphDatabase

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def get_neo4j_driver() -> Driver:
    """Return the singleton Neo4j driver configured from settings."""
    settings = get_settings()
    logger.info("Creating Neo4j driver for %s", settings.neo4j_uri)
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )


def close_neo4j_driver() -> None:
    """Close the driver if it was created; safe to call when it never was."""
    if get_neo4j_driver.cache_info().currsize:
        get_neo4j_driver().close()
        get_neo4j_driver.cache_clear()
        logger.info("Neo4j driver closed")
