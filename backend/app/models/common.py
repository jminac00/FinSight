import re

from pydantic import BaseModel, field_validator


class TickerRequest(BaseModel):
    """Validated stock ticker symbol."""

    ticker: str

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        v = v.upper().strip()
        if not re.match(r"^[A-Z0-9]{2,5}$", v):
            raise ValueError("ticker must be 2–5 alphanumeric characters (e.g. AAPL)")
        return v


class ErrorResponse(BaseModel):
    detail: str
