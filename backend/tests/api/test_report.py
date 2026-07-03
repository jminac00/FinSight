"""Tests for the consolidated report endpoint."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.testclient import TestClient

from app.api.v1.deep_learning import get_dl_service
from app.api.v1.fundamental import get_fundamental_service
from app.api.v1.search import get_symbol_search_service
from app.api.v1.sentiment import get_sentiment_service
from app.api.v1.technical import get_technical_service
from app.llm.factory import get_llm_service
from app.main import app
from app.models.deep_learning import DLResult, ModelMetrics
from app.models.fundamental import FundamentalResult
from app.models.search import SymbolMatch
from app.models.sentiment import SentimentResult
from app.models.technical import TechnicalBlockScores, TechnicalResult
from app.services.deep_learning.service import ModelNotAvailableError

_SENTIMENT = SentimentResult(
    label="positivo",
    score=0.6,
    confidence=0.8,
    explanation="Noticias positivas.",
    influential_news=[],
)
_DL = DLResult(
    trend="alcista",
    predicted_return_pct=2.5,
    predicted_price=205.0,
    current_price=200.0,
    horizon_days=10,
    trained_at=datetime(2026, 7, 1, 22, 0, 0, tzinfo=UTC),
    metrics=ModelMetrics(rmse=3.1, mae=2.4, directional_accuracy=0.65),
)
_FUNDAMENTAL = FundamentalResult(
    score=7.5,
    metrics={"per": 28.0},
    llm_analysis="Sólido.",
    cached_at=datetime.now(tz=UTC),
)
_TECHNICAL = TechnicalResult(
    score=6.8,
    signal="alcista",
    block_scores=TechnicalBlockScores(
        momentum=7.0, trend=6.0, risk_stability=5.5, confirmation=6.8
    ),
    indicators={},
    llm_analysis="Tendencia positiva.",
    calculated_at=datetime.now(tz=UTC),
)
_LLM_CONCLUSION = "Conclusión global de prueba."
_FALLBACK_CONCLUSION = (
    "Se ha completado el análisis de los módulos disponibles. "
    "Consulte cada sección para obtener los detalles del análisis."
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_services():
    svc_s = MagicMock()
    svc_s.analyze = AsyncMock(return_value=_SENTIMENT)
    svc_d = MagicMock()
    svc_d.predict = AsyncMock(return_value=_DL)
    svc_f = MagicMock()
    svc_f.analyze = AsyncMock(return_value=_FUNDAMENTAL)
    svc_t = MagicMock()
    svc_t.analyze = AsyncMock(return_value=_TECHNICAL)
    svc_search = MagicMock()
    svc_search.search = AsyncMock(
        return_value=[
            SymbolMatch(
                symbol="AAPL", description="Apple Inc", type="Common Stock", display_symbol="AAPL"
            )
        ]
    )
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value=_LLM_CONCLUSION)

    app.dependency_overrides[get_sentiment_service] = lambda: svc_s
    app.dependency_overrides[get_dl_service] = lambda: svc_d
    app.dependency_overrides[get_fundamental_service] = lambda: svc_f
    app.dependency_overrides[get_technical_service] = lambda: svc_t
    app.dependency_overrides[get_symbol_search_service] = lambda: svc_search
    app.dependency_overrides[get_llm_service] = lambda: mock_llm
    yield svc_s, svc_d, svc_f, svc_t, svc_search, mock_llm
    for fn in (
        get_sentiment_service,
        get_dl_service,
        get_fundamental_service,
        get_technical_service,
        get_symbol_search_service,
        get_llm_service,
    ):
        app.dependency_overrides.pop(fn, None)


def test_get_report_returns_200_with_full_response(client, mock_services):
    response = client.get("/api/v1/report/AAPL")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "AAPL"
    assert data["company_name"] == "Apple Inc"
    assert data["sentiment"] is not None
    assert data["deep_learning"] is not None
    assert data["fundamental"] is not None
    assert data["technical"] is not None
    assert data["global_conclusion"] == _LLM_CONCLUSION
    assert data["partial_support"] is False
    assert data["missing_modules"] == []
    assert "disclaimer" in data


def test_get_report_invalid_ticker_returns_422(client, mock_services):
    response = client.get("/api/v1/report/TOOLONGX")
    assert response.status_code == 422


def test_get_report_ticker_normalized_to_uppercase(client, mock_services):
    svc_s, svc_d, svc_f, svc_t, svc_search, mock_llm = mock_services
    response = client.get("/api/v1/report/aapl")
    assert response.status_code == 200
    svc_s.analyze.assert_awaited_once_with("AAPL", force_refresh=False)
    svc_d.predict.assert_awaited_once_with("AAPL")


def test_get_report_partial_support_when_dl_unavailable(client, mock_services):
    svc_s, svc_d, svc_f, svc_t, svc_search, mock_llm = mock_services
    svc_d.predict = AsyncMock(side_effect=ModelNotAvailableError("no model for FAKE"))
    response = client.get("/api/v1/report/FAKE")
    assert response.status_code == 200
    data = response.json()
    assert data["deep_learning"] is None
    assert data["partial_support"] is True
    assert data["missing_modules"] == ["deep_learning"]
    assert data["sentiment"] is not None


def test_get_report_force_refresh_forwarded_to_services(client, mock_services):
    svc_s, svc_d, svc_f, svc_t, svc_search, mock_llm = mock_services
    response = client.get("/api/v1/report/AAPL?force_refresh=true")
    assert response.status_code == 200
    svc_s.analyze.assert_awaited_once_with("AAPL", force_refresh=True)
    svc_f.analyze.assert_awaited_once_with("AAPL", mode="auto", force_refresh=True)
    svc_t.analyze.assert_awaited_once_with("AAPL", mode="auto", force_refresh=True)
    svc_d.predict.assert_awaited_once_with("AAPL")


def test_get_report_includes_disclaimer(client, mock_services):
    response = client.get("/api/v1/report/AAPL")
    assert response.status_code == 200
    data = response.json()
    assert data["disclaimer"]


def test_get_report_llm_fallback_when_complete_raises(client, mock_services):
    svc_s, svc_d, svc_f, svc_t, svc_search, mock_llm = mock_services
    mock_llm.complete = AsyncMock(side_effect=RuntimeError("LLM down"))
    response = client.get("/api/v1/report/AAPL")
    assert response.status_code == 200
    data = response.json()
    assert data["global_conclusion"] == _FALLBACK_CONCLUSION


def test_get_report_sentiment_failure_returns_null_sentiment(client, mock_services):
    svc_s, svc_d, svc_f, svc_t, svc_search, mock_llm = mock_services
    svc_s.analyze = AsyncMock(side_effect=Exception("network error"))
    response = client.get("/api/v1/report/AAPL")
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] is None
    assert data["partial_support"] is True
    assert data["missing_modules"] == ["sentiment"]
    assert data["deep_learning"] is not None
    assert data["fundamental"] is not None
    assert data["technical"] is not None


def test_get_report_company_name_null_when_no_exact_match(client, mock_services):
    svc_s, svc_d, svc_f, svc_t, svc_search, mock_llm = mock_services
    svc_search.search = AsyncMock(return_value=[])
    response = client.get("/api/v1/report/AAPL")
    assert response.status_code == 200
    assert response.json()["company_name"] is None


def test_get_report_company_name_null_when_search_fails(client, mock_services):
    svc_s, svc_d, svc_f, svc_t, svc_search, mock_llm = mock_services
    svc_search.search = AsyncMock(side_effect=Exception("finnhub down"))
    response = client.get("/api/v1/report/AAPL")
    assert response.status_code == 200
    assert response.json()["company_name"] is None
