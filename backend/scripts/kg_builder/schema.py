"""Neo4j schema initialization: constraints and vector index."""

import logging

from neo4j import AsyncDriver

logger = logging.getLogger(__name__)

# text-embedding-3-large produces 3 072 dimensions
_EMBEDDING_DIMS = 3072

_CONSTRAINTS = [
    "CREATE CONSTRAINT asset_ticker IF NOT EXISTS FOR (a:Asset) REQUIRE a.ticker IS UNIQUE",
    "CREATE CONSTRAINT news_id IF NOT EXISTS FOR (n:News) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE",
]

_VECTOR_INDEX = (
    "CREATE VECTOR INDEX news_embedding IF NOT EXISTS "
    "FOR (n:News) ON n.embedding "
    f"OPTIONS {{indexConfig: {{"
    f"`vector.dimensions`: {_EMBEDDING_DIMS}, "
    f"`vector.similarity_function`: 'cosine'"
    f"}}}}"
)


async def init_schema(driver: AsyncDriver, database: str) -> None:
    """Create constraints and vector index if they do not already exist."""
    async with driver.session(database=database) as session:
        for stmt in _CONSTRAINTS:
            await session.run(stmt)
            logger.debug("Constraint applied: %s", stmt[:60])
        await session.run(_VECTOR_INDEX)
        logger.info("Schema initialised (constraints + vector index).")
