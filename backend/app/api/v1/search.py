from functools import lru_cache

from fastapi import APIRouter, Depends, Query, Request

from app.core.config import get_settings
from app.core.finnhub import get_finnhub_client
from app.core.rate_limit import limiter
from app.models.search import SymbolSearchResponse
from app.services.search.service import SymbolSearchService

router = APIRouter()


@lru_cache
def get_symbol_search_service() -> SymbolSearchService:
    """Return the singleton symbol search service wired from settings."""
    settings = get_settings()
    return SymbolSearchService(
        client=get_finnhub_client(),
        cache_ttl=settings.cache_ttl_search,
    )


@router.get("/search", response_model=SymbolSearchResponse)
@limiter.limit(lambda: get_settings().rate_limit_search)
async def search_symbols(
    request: Request,
    q: str = Query(min_length=1, max_length=64),
    service: SymbolSearchService = Depends(get_symbol_search_service),
) -> SymbolSearchResponse:
    """Search listed symbols by company name or ticker (fuzzy, global coverage)."""
    results = await service.search(q)
    return SymbolSearchResponse(query=q, results=results)
