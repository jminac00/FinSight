from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.rate_limit import limiter
from app.llm.base import LLMService
from app.main import app


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset in-memory rate-limit counters before each test."""
    limiter._storage.reset()
    yield


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_llm() -> LLMService:
    service = MagicMock(spec=LLMService)
    service.complete = AsyncMock(return_value="Respuesta de prueba del LLM.")
    return service
