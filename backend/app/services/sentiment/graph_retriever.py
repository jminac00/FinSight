"""GraphRAG retrieval over the Neo4j knowledge graph.

Given a query embedding, finds the most similar reference News nodes via the
`news_embedding` vector index and expands their k-hop neighbourhood (assets with
sentiment-annotated MENTIONS, topics, events) to build LLM context.
"""

import asyncio
import logging
from dataclasses import dataclass, field

import neo4j
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem

logger = logging.getLogger(__name__)

_VECTOR_INDEX_NAME = "news_embedding"


def _build_retrieval_query(hop_depth: int) -> str:
    """Build the Cypher executed after vector search, honouring the hop depth.

    Depth 1 covers the direct neighbourhood of each News node (MENTIONS→Asset,
    TAGGED→Topic, DESCRIBES→Event); depth ≥ 2 adds the Event→AFFECTS→Asset hop.
    """
    legs = [
        "OPTIONAL MATCH (node)-[m:MENTIONS]->(a:Asset)",
        "OPTIONAL MATCH (node)-[:TAGGED]->(t:Topic)",
        "OPTIONAL MATCH (node)-[:DESCRIBES]->(e:Event)",
    ]
    affected_column = "[] AS affected_assets"
    if hop_depth >= 2:
        legs.append("OPTIONAL MATCH (e)-[:AFFECTS]->(ea:Asset)")
        affected_column = "collect(DISTINCT ea.name) AS affected_assets"

    return f"""
        {chr(10).join(legs)}
        RETURN node.text AS text,
               node.title AS title,
               score,
               collect(DISTINCT CASE WHEN a IS NULL THEN NULL ELSE {{
                   ticker: a.ticker,
                   name: a.name,
                   sentiment_label: m.sentiment_label,
                   sentiment_score: m.sentiment_score
               }} END) AS mentions,
               collect(DISTINCT t.name) AS topics,
               collect(DISTINCT CASE WHEN e IS NULL THEN NULL ELSE {{
                   type: e.type,
                   description: e.description,
                   date: e.date
               }} END) AS events,
               {affected_column}
        """


def _format_record(record: neo4j.Record) -> RetrieverResultItem:
    """Map a Cypher result record into a RetrieverResultItem with clean metadata."""
    return RetrieverResultItem(
        content=record.get("text") or "",
        metadata={
            "title": record.get("title"),
            "score": record.get("score"),
            "mentions": [m for m in record.get("mentions") or [] if m],
            "topics": [t for t in record.get("topics") or [] if t],
            "events": [e for e in record.get("events") or [] if e],
            "affected_assets": [a for a in record.get("affected_assets") or [] if a],
        },
    )


@dataclass
class RetrievedNews:
    """A reference news node retrieved from the graph with its neighbourhood."""

    text: str
    title: str | None
    similarity: float
    mentions: list[dict] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    affected_assets: list[str] = field(default_factory=list)


class GraphRetriever:
    """Vector similarity + k-hop expansion over the Neo4j knowledge graph."""

    def __init__(self, driver: neo4j.Driver, database: str, hop_depth: int) -> None:
        """Args:
        driver: Synchronous Neo4j driver (managed by app.core.neo4j).
        database: Neo4j database name.
        hop_depth: Traversal depth from each retrieved News node.
        """
        self._retriever = VectorCypherRetriever(
            driver=driver,
            index_name=_VECTOR_INDEX_NAME,
            retrieval_query=_build_retrieval_query(hop_depth),
            result_formatter=_format_record,
            neo4j_database=database,
        )

    async def retrieve(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievedNews]:
        """Return the top-k most similar news with their graph neighbourhood.

        The underlying neo4j-graphrag search is synchronous, so it runs in a
        worker thread to avoid blocking the event loop.
        """
        result = await asyncio.to_thread(
            self._retriever.search, query_vector=query_embedding, top_k=top_k
        )
        return [
            RetrievedNews(
                text=item.content,
                title=item.metadata.get("title"),
                similarity=item.metadata.get("score") or 0.0,
                mentions=item.metadata.get("mentions") or [],
                topics=item.metadata.get("topics") or [],
                events=item.metadata.get("events") or [],
                affected_assets=item.metadata.get("affected_assets") or [],
            )
            for item in result.items
        ]
