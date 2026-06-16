from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TechnicalBlockScores(BaseModel):
    """0-10 score of each technical block (None if the block could not be computed)."""

    momentum: float | None
    trend: float | None
    risk_stability: float | None
    confirmation: float | None


class TechnicalResult(BaseModel):
    score: float = Field(ge=0.0, le=10.0)  # composite technical score
    signal: Literal["alcista", "bajista", "neutral"]
    block_scores: TechnicalBlockScores
    indicators: dict  # per-block detail, effective weights, universe and reliability metadata
    llm_analysis: str
    calculated_at: datetime
