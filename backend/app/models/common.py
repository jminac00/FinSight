import re

from pydantic import BaseModel, field_validator

# International symbols carry exchange suffixes and class separators
# (e.g. REP.MC, ASML.AS, 7203.T, BRK-B), so the US-only 2-5 alphanumeric
# rule is relaxed for universal (MSCI World) coverage.
_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,14}$")


class TickerRequest(BaseModel):
    """Validated stock ticker symbol."""

    ticker: str

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        v = v.upper().strip()
        if not _TICKER_RE.match(v):
            raise ValueError("invalid ticker symbol")
        return v


class ErrorResponse(BaseModel):
    detail: str
