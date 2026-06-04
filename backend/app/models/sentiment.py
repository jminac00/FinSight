from typing import Literal

from pydantic import BaseModel, Field


class NewsItem(BaseModel):
    title: str
    url: str
    source: str


class SentimentResult(BaseModel):
    label: Literal["positivo", "negativo", "neutral"]
    score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str
    influential_news: list[NewsItem]
