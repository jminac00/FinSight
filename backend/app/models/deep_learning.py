from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class ModelMetrics(BaseModel):
    rmse: float
    mae: float
    mape: float
    r2: float


class DLResult(BaseModel):
    trend: Literal["alcista", "bajista"]
    predicted_price: float
    current_price: float
    pct_change: float
    horizon_days: int
    trained_at: datetime
    metrics: ModelMetrics
