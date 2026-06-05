"""Neo4j connection and MERGE helpers for graph construction."""

import hashlib
import logging
import uuid
from typing import Optional

from neo4j import AsyncDriver, AsyncGraphDatabase

logger = logging.getLogger(__name__)


def create_driver(uri: str, username: str, password: str) -> AsyncDriver:
    return AsyncGraphDatabase.driver(uri, auth=(username, password))


def news_id(text: str) -> str:
    """Deterministic 32-char hex ID derived from normalised text."""
    normalised = text.lower().strip()
    return hashlib.sha256(normalised.encode()).hexdigest()[:32]


async def merge_asset(
    session,
    ticker: Optional[str],
    name: str,
    asset_type: str,
) -> None:
    if ticker:
        await session.run(
            """
            MERGE (a:Asset {ticker: $ticker})
            SET a.name = $name, a.asset_type = $asset_type
            """,
            ticker=ticker,
            name=name,
            asset_type=asset_type,
        )
    else:
        # Assets without a resolved ticker are stored with name as fallback key.
        # They won't be deduplicated across datasets but won't block the pipeline.
        await session.run(
            """
            MERGE (a:Asset {ticker: $name})
            SET a.name = $name, a.asset_type = $asset_type
            """,
            name=name,
            asset_type=asset_type,
        )


async def merge_news(
    session,
    node_id: str,
    text: str,
    title: Optional[str],
    embedding: list[float],
    published_at: Optional[str],
    source: str,
) -> None:
    await session.run(
        """
        MERGE (n:News {id: $id})
        SET n.text = $text,
            n.title = $title,
            n.embedding = $embedding,
            n.published_at = $published_at,
            n.source = $source
        """,
        id=node_id,
        text=text,
        title=title,
        embedding=embedding,
        published_at=published_at,
        source=source,
    )


async def merge_mentions(
    session,
    news_id: str,
    ticker: Optional[str],
    entity_name: str,
    sentiment_label: str,
    sentiment_score: int,
    pct_change: Optional[float] = None,
) -> None:
    asset_key = ticker if ticker else entity_name
    query = """
        MATCH (n:News {id: $news_id})
        MATCH (a:Asset {ticker: $asset_key})
        MERGE (n)-[r:MENTIONS]->(a)
        SET r.sentiment_label = $sentiment_label,
            r.sentiment_score = $sentiment_score
        """
    params: dict = {
        "news_id": news_id,
        "asset_key": asset_key,
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
    }
    if pct_change is not None:
        query = query.rstrip() + ", r.pct_change = $pct_change"
        params["pct_change"] = pct_change
    await session.run(query, **params)


async def merge_topic(session, name: str) -> None:
    await session.run("MERGE (t:Topic {name: $name})", name=name)


async def merge_tagged(session, news_id: str, topic_name: str) -> None:
    await session.run(
        """
        MATCH (n:News {id: $news_id})
        MATCH (t:Topic {name: $topic_name})
        MERGE (n)-[:TAGGED]->(t)
        """,
        news_id=news_id,
        topic_name=topic_name,
    )


async def merge_event(
    session,
    event_id: str,
    event_type: str,
    description: str,
    date: Optional[str],
) -> None:
    await session.run(
        """
        MERGE (e:Event {id: $id})
        SET e.type = $type, e.description = $description, e.date = $date
        """,
        id=event_id,
        type=event_type,
        description=description,
        date=date,
    )


async def merge_describes(session, news_id: str, event_id: str) -> None:
    await session.run(
        """
        MATCH (n:News {id: $news_id})
        MATCH (e:Event {id: $event_id})
        MERGE (n)-[:DESCRIBES]->(e)
        """,
        news_id=news_id,
        event_id=event_id,
    )


async def merge_affects(session, event_id: str, ticker: Optional[str], entity_name: str) -> None:
    asset_key = ticker if ticker else entity_name
    await session.run(
        """
        MATCH (e:Event {id: $event_id})
        MATCH (a:Asset {ticker: $asset_key})
        MERGE (e)-[:AFFECTS]->(a)
        """,
        event_id=event_id,
        asset_key=asset_key,
    )


def new_event_id() -> str:
    return str(uuid.uuid4())
