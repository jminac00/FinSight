"""Tests for the GraphRAG retriever. The neo4j-graphrag retriever is fully mocked."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services.sentiment import graph_retriever
from app.services.sentiment.graph_retriever import (
    GraphRetriever,
    RetrievedNews,
    _build_retrieval_query,
    _format_record,
)


def _result_item(metadata: dict, content: str = "some news text") -> SimpleNamespace:
    return SimpleNamespace(content=content, metadata=metadata)


def _make_retriever(
    search_result, hop_depth: int = 2
) -> tuple[GraphRetriever, MagicMock, MagicMock]:
    inner = MagicMock()
    inner.search.return_value = search_result
    with patch.object(graph_retriever, "VectorCypherRetriever", return_value=inner) as cls:
        retriever = GraphRetriever(driver=MagicMock(), database="neo4j", hop_depth=hop_depth)
    return retriever, inner, cls


async def test_retrieve_maps_results_to_retrieved_news():
    metadata = {
        "title": "Apple beats estimates",
        "score": 0.93,
        "mentions": [
            {"ticker": "AAPL", "name": "Apple", "sentiment_label": "Positive", "sentiment_score": 1}
        ],
        "topics": ["earnings"],
        "events": [{"type": "earnings_report", "description": "Q3 beat", "date": "2026-06-01"}],
        "affected_assets": ["Microsoft"],
    }
    result = SimpleNamespace(items=[_result_item(metadata)])
    retriever, inner, _ = _make_retriever(result)

    news = await retriever.retrieve([0.1, 0.2], top_k=3)

    assert len(news) == 1
    item = news[0]
    assert isinstance(item, RetrievedNews)
    assert item.text == "some news text"
    assert item.title == "Apple beats estimates"
    assert item.similarity == 0.93
    assert item.mentions[0]["ticker"] == "AAPL"
    assert item.topics == ["earnings"]
    assert item.events[0]["type"] == "earnings_report"
    assert item.affected_assets == ["Microsoft"]

    inner.search.assert_called_once_with(query_vector=[0.1, 0.2], top_k=3)


async def test_retrieve_handles_empty_results():
    result = SimpleNamespace(items=[])
    retriever, _, _ = _make_retriever(result)

    assert await retriever.retrieve([0.1], top_k=5) == []


def test_constructor_configures_vector_index_and_database():
    result = SimpleNamespace(items=[])
    _, _, cls = _make_retriever(result, hop_depth=2)

    kwargs = cls.call_args.kwargs
    assert kwargs["index_name"] == "news_embedding"
    assert kwargs["neo4j_database"] == "neo4j"
    assert "MENTIONS" in kwargs["retrieval_query"]


def test_retrieval_query_honours_hop_depth():
    shallow = _build_retrieval_query(1)
    deep = _build_retrieval_query(2)

    # Depth 1: direct neighbourhood only (assets, topics, events).
    assert "MENTIONS" in shallow
    assert "TAGGED" in shallow
    assert "DESCRIBES" in shallow
    assert "AFFECTS" not in shallow

    # Depth 2 adds the Event→AFFECTS→Asset second hop.
    assert "AFFECTS" in deep


def test_format_record_normalises_fields():
    record = {
        "text": "news body",
        "title": None,
        "score": 0.5,
        "mentions": None,
        "topics": None,
        "events": None,
        "affected_assets": None,
    }

    item = _format_record(record)

    assert item.content == "news body"
    assert item.metadata["mentions"] == []
    assert item.metadata["topics"] == []
    assert item.metadata["events"] == []
    assert item.metadata["affected_assets"] == []
