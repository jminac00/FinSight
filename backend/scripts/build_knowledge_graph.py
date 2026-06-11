"""
Offline batch script — builds the Neo4j knowledge graph from FinEntity and FinMarBa datasets.

Usage (from backend/):
    uv run python scripts/build_knowledge_graph.py \\
        --finentity scripts/kg_builder/data/FinEntity_dataset.json \\
        --finmarba  scripts/kg_builder/data/FinMarBa_dataset.csv

    uv run python scripts/build_knowledge_graph.py --init-schema   # schema only
    uv run python scripts/build_knowledge_graph.py --limit 20 ...  # smoke test
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Ensure backend/ is on sys.path so `app` and `scripts` packages are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from app.core.config import get_settings  # noqa: E402
from app.llm.factory import get_llm_service  # noqa: E402
from scripts.kg_builder import embeddings as emb  # noqa: E402
from scripts.kg_builder import entity_resolver as resolver  # noqa: E402
from scripts.kg_builder import llm_extractor as extractor  # noqa: E402
from scripts.kg_builder import neo4j_client as neo  # noqa: E402
from scripts.kg_builder import schema as schema_mod  # noqa: E402
from scripts.kg_builder.loaders import finentity as fe_loader  # noqa: E402
from scripts.kg_builder.loaders import finmarba as fm_loader  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("kg_builder")

_LOG_EVERY = 100


async def _process_finentity(
    records: list,
    driver,
    database: str,
    llm_service,
) -> None:
    texts = [r.text for r in records]
    logger.info("FinEntity: generating embeddings for %d records…", len(texts))
    embeddings = await emb.embed_texts(texts)

    logger.info("FinEntity: running LLM extraction…")
    extractions = await extractor.extract_batch(texts, llm_service, concurrency=3)

    async with driver.session(database=database) as session:
        for i, (record, embedding, extraction) in enumerate(
            zip(records, embeddings, extractions, strict=True), start=1
        ):
            node_id = neo.news_id(record.text)

            # Merge all assets referenced in this record
            resolved: list[tuple] = []
            for ann in record.annotations:
                ticker, name, asset_type = resolver.resolve_from_name(ann.value, record.text)
                await neo.merge_asset(session, ticker, name, asset_type)
                resolved.append((ann, ticker, name))

            # Merge News node
            await neo.merge_news(
                session,
                node_id=node_id,
                text=record.text,
                title=None,
                embedding=embedding,
                published_at=None,
                source="finentity",
            )

            # MENTIONS relations (sentiment from human annotations)
            for ann, ticker, name in resolved:
                score_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
                await neo.merge_mentions(
                    session,
                    news_id=node_id,
                    ticker=ticker,
                    entity_name=name,
                    sentiment_label=ann.sentiment_label,
                    sentiment_score=score_map.get(ann.sentiment_label, 0),
                )

            # Topics
            for topic in extraction.topics:
                await neo.merge_topic(session, topic)
                await neo.merge_tagged(session, node_id, topic)

            # Events
            asset_tuples = [(t, n) for _, t, n in resolved]
            for event in extraction.events:
                event_id = neo.new_event_id()
                await neo.merge_event(session, event_id, event.type, event.description, event.date)
                await neo.merge_describes(session, node_id, event_id)
                for ticker, name in asset_tuples:
                    await neo.merge_affects(session, event_id, ticker, name)

            if i % _LOG_EVERY == 0:
                logger.info("FinEntity: processed %d/%d records", i, len(records))

    logger.info("FinEntity: done.")


async def _process_finmarba(
    records: list,
    driver,
    database: str,
    llm_service,
) -> None:
    texts = [r.title for r in records]
    logger.info("FinMarBa: generating embeddings for %d records…", len(texts))
    embeddings = await emb.embed_texts(texts)

    logger.info("FinMarBa: running LLM extraction…")
    extractions = await extractor.extract_batch(texts, llm_service, concurrency=3)

    async with driver.session(database=database) as session:
        for i, (record, embedding, extraction) in enumerate(
            zip(records, embeddings, extractions, strict=True), start=1
        ):
            node_id = neo.news_id(record.title)

            # Merge all assets
            resolved: list[tuple] = []
            for ticker in record.tickers:
                t, name, asset_type = resolver.resolve_from_ticker(ticker)
                await neo.merge_asset(session, t, name, asset_type)
                resolved.append((ticker, t, name))

            # Merge News node
            await neo.merge_news(
                session,
                node_id=node_id,
                text=record.title,
                title=record.title,
                embedding=embedding,
                published_at=record.date,
                source="finmarba",
            )

            # MENTIONS with sentiment + pct_change (from structured data — never LLM)
            for orig_ticker, ticker, name in resolved:
                label = record.sentiment.get(orig_ticker, "Neutral")
                score = record.sentiment_score.get(orig_ticker, 0)
                pct = record.pct_change.get(orig_ticker)
                await neo.merge_mentions(
                    session,
                    news_id=node_id,
                    ticker=ticker,
                    entity_name=name,
                    sentiment_label=label,
                    sentiment_score=score,
                    pct_change=pct,
                )

            # Topics
            for topic in extraction.topics:
                await neo.merge_topic(session, topic)
                await neo.merge_tagged(session, node_id, topic)

            # Events
            asset_tuples = [(t, n) for _, t, n in resolved]
            for event in extraction.events:
                event_id = neo.new_event_id()
                await neo.merge_event(session, event_id, event.type, event.description, event.date)
                await neo.merge_describes(session, node_id, event_id)
                for ticker, name in asset_tuples:
                    await neo.merge_affects(session, event_id, ticker, name)

            if i % _LOG_EVERY == 0:
                logger.info("FinMarBa: processed %d/%d records", i, len(records))

    logger.info("FinMarBa: done.")


async def main(args: argparse.Namespace) -> None:
    settings = get_settings()
    driver = neo.create_driver(settings.neo4j_uri, settings.neo4j_username, settings.neo4j_password)
    database = settings.neo4j_database

    try:
        await schema_mod.init_schema(driver, database)

        if args.init_schema:
            logger.info("Schema initialised. Exiting (--init-schema flag).")
            return

        llm_service = get_llm_service()

        if args.finentity:
            records = fe_loader.load(Path(args.finentity))
            if args.limit:
                records = records[: args.limit]
            await _process_finentity(records, driver, database, llm_service)

        if args.finmarba:
            records = fm_loader.load(Path(args.finmarba))
            if args.limit:
                records = records[: args.limit]
            await _process_finmarba(records, driver, database, llm_service)

    finally:
        await driver.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the FinSight Neo4j knowledge graph.")
    parser.add_argument("--finentity", metavar="PATH", help="Path to FinEntity_dataset.json")
    parser.add_argument("--finmarba", metavar="PATH", help="Path to FinMarBa_dataset.csv")
    parser.add_argument(
        "--limit", type=int, metavar="N", help="Process only the first N records (for testing)"
    )
    parser.add_argument(
        "--init-schema", action="store_true", help="Only initialise schema, then exit"
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(_parse_args()))
