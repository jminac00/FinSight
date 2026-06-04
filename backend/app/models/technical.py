from datetime import datetime

from pydantic import BaseModel, Field


class TechnicalResult(BaseModel):
    score: float = Field(ge=0.0, le=10.0)
    indicators: dict
    llm_analysis: str
    calculated_at: datetime
