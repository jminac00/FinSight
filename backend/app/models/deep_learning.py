from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class ModelMetrics(BaseModel):
    """GRU quality metrics on the held-out test split (10-day return, in % points)."""

    rmse: float
    mae: float
    directional_accuracy: float  # [0, 1] sign-agreement rate


class DLResult(BaseModel):
    trend: Literal["alcista", "bajista", "neutral"]
    predicted_return_pct: float  # model output: cumulative 10-day return, in %
    predicted_price: float  # derived: current_price * (1 + predicted_return_pct / 100)
    current_price: float
    horizon_days: int
    trained_at: datetime
    metrics: ModelMetrics


class DLTrainResult(BaseModel):
    """Response from the on-demand training endpoint."""

    ticker: str
    trained_at: datetime
    metrics: ModelMetrics
    data_through: str
