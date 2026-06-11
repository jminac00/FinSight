"""Tests for the runtime embedding service. The OpenAI client is fully mocked."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.llm import embeddings


def _embedding_response(vectors_by_index: list[tuple[int, list[float]]]) -> SimpleNamespace:
    """Build a fake OpenAI embeddings response with explicit (index, embedding) pairs."""
    return SimpleNamespace(
        data=[SimpleNamespace(index=i, embedding=vec) for i, vec in vectors_by_index]
    )


def _mock_openai_client(side_effect) -> MagicMock:
    client = MagicMock()
    client.embeddings.create = AsyncMock(side_effect=side_effect)
    return client


async def test_embed_texts_returns_embeddings_in_input_order():
    # Response arrives with indices out of order; result must follow input order.
    response = _embedding_response([(1, [0.2, 0.2]), (0, [0.1, 0.1])])
    client = _mock_openai_client([response])

    with patch.object(embeddings, "AsyncOpenAI", return_value=client):
        result = await embeddings.embed_texts(["first", "second"])

    assert result == [[0.1, 0.1], [0.2, 0.2]]
    call_kwargs = client.embeddings.create.call_args.kwargs
    assert call_kwargs["input"] == ["first", "second"]


async def test_embed_texts_processes_in_batches():
    n_texts = embeddings._BATCH_SIZE + 5
    texts = [f"text {i}" for i in range(n_texts)]

    def respond(input: list[str], model: str) -> SimpleNamespace:
        return _embedding_response([(i, [float(i)]) for i in range(len(input))])

    client = _mock_openai_client(respond)

    with patch.object(embeddings, "AsyncOpenAI", return_value=client):
        result = await embeddings.embed_texts(texts)

    assert len(result) == n_texts
    assert client.embeddings.create.call_count == 2


async def test_embed_texts_retries_on_transient_failure():
    response = _embedding_response([(0, [0.5])])
    client = _mock_openai_client([RuntimeError("rate limited"), response])

    with (
        patch.object(embeddings, "AsyncOpenAI", return_value=client),
        patch("asyncio.sleep", new=AsyncMock()),
    ):
        result = await embeddings.embed_texts(["only"])

    assert result == [[0.5]]
    assert client.embeddings.create.call_count == 2


async def test_embed_texts_raises_after_max_retries():
    client = _mock_openai_client(RuntimeError("permanent failure"))

    with (
        patch.object(embeddings, "AsyncOpenAI", return_value=client),
        patch("asyncio.sleep", new=AsyncMock()),
        pytest.raises(RuntimeError),
    ):
        await embeddings.embed_texts(["only"])


async def test_embed_texts_empty_input_returns_empty_list():
    client = _mock_openai_client([])

    with patch.object(embeddings, "AsyncOpenAI", return_value=client):
        result = await embeddings.embed_texts([])

    assert result == []
    client.embeddings.create.assert_not_called()
