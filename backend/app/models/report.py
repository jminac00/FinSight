from datetime import datetime

from pydantic import BaseModel

from app.models.deep_learning import DLResult
from app.models.fundamental import FundamentalResult
from app.models.sentiment import SentimentResult
from app.models.technical import TechnicalResult


class ReportResponse(BaseModel):
    ticker: str
    company_name: str | None
    generated_at: datetime
    sentiment: SentimentResult | None
    deep_learning: DLResult | None
    fundamental: FundamentalResult | None
    technical: TechnicalResult | None
    global_conclusion: str
    disclaimer: str
    partial_support: bool
    missing_modules: list[str]
