from datetime import datetime

from pydantic import BaseModel, Field


class FundamentalResult(BaseModel):
    score: float = Field(ge=0.0, le=10.0)
    metrics: dict
    llm_analysis: str
    cached_at: datetime
