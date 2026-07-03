from pydantic import BaseModel


class SymbolMatch(BaseModel):
    """A single symbol/company match from the search provider."""

    symbol: str
    description: str
    type: str
    display_symbol: str


class SymbolSearchResponse(BaseModel):
    """Response for a symbol/company search query."""

    query: str
    results: list[SymbolMatch]
