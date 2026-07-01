"""DLService: GRU inference with LRU model cache."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from cachetools import LRUCache

from app.core.market_data import get_price_history
from app.models.deep_learning import DLResult, ModelMetrics
from app.services.deep_learning.artifacts import DEFAULT_MODELS_DIR, ModelArtifacts, ModelMetadata
from app.services.deep_learning.model import GRURegressor
from app.services.deep_learning.preprocessing import (
    FEATURES,
    INPUT_SIZE,
    InsufficientHistoryError,
    clean_ohlc,
    engineered_features,
)

logger = logging.getLogger(__name__)


class ModelNotAvailableError(Exception):
    """Raised when no trained GRU artifact exists for a ticker."""


class DLService:
    """Manages GRU model loading (lazy + LRU cache) and inference.

    Flow: yfinance EOD data → lookback window → GRU inference → 10-day return
          → trend classification (neutral band ±1.5 %).
    """

    def __init__(
        self,
        models_dir: Path = DEFAULT_MODELS_DIR,
        max_models: int = 10,
    ) -> None:
        self._models_dir = models_dir
        self._cache: LRUCache[str, tuple[nn.Module, ModelMetadata]] = LRUCache(maxsize=max_models)

    async def is_model_available(self, ticker: str) -> bool:
        """Return True if both .pt and .json artifact files exist on disk."""
        return (self._models_dir / f"{ticker}.pt").is_file() and (
            self._models_dir / f"{ticker}.json"
        ).is_file()

    def _get_model(self, ticker: str) -> tuple[nn.Module, ModelMetadata]:
        """Load model and metadata from LRU cache or disk. Blocking — call in thread."""
        if ticker not in self._cache:
            artifacts = ModelArtifacts.load(ticker, self._models_dir)
            model = GRURegressor(input_size=INPUT_SIZE, **artifacts.metadata.recipe)
            model.load_state_dict(artifacts.state_dict)
            model.eval()
            self._cache[ticker] = (model, artifacts.metadata)
            logger.debug(
                "Loaded model for %s into cache (%d/%d)",
                ticker,
                len(self._cache),
                self._cache.maxsize,
            )
        return self._cache[ticker]

    @staticmethod
    def _build_window(ohlc_clean: pd.DataFrame, lookback: int) -> np.ndarray:
        """Build and z-score the most recent inference window.

        Returns an array of shape (1, lookback, INPUT_SIZE), applying the same
        per-window normalisation as the training pipeline.
        """
        arr = ohlc_clean[FEATURES].to_numpy(dtype=np.float64)
        features = np.concatenate([arr, engineered_features(arr)], axis=1)
        window = features[-lookback:].copy()
        mu = window.mean(axis=0)
        sd = window.std(axis=0)
        sd[sd < 1e-8] = 1.0
        return ((window - mu) / sd).astype(np.float32)[np.newaxis, :, :]

    @staticmethod
    def _forward(model: nn.Module, X: np.ndarray) -> float:
        """Run the GRU forward pass for a single window. Blocking — call in thread."""
        with torch.no_grad():
            return float(model(torch.from_numpy(X)).item())

    async def predict(self, ticker: str) -> DLResult:
        """Generate a 10-day GRU return prediction for the given ticker.

        Args:
            ticker: Uppercase stock symbol (e.g. 'AAPL').

        Returns:
            DLResult with trend, predicted return, derived price, current price,
            horizon days, training date and model quality metrics.

        Raises:
            ModelNotAvailableError: No trained artifact exists for ticker.
            InsufficientHistoryError: yfinance returned fewer than lookback clean candles.
            ValueError: yfinance returned no data for ticker.
        """
        if not await self.is_model_available(ticker):
            raise ModelNotAvailableError(f"No trained model for '{ticker}'")

        model, metadata = await asyncio.to_thread(self._get_model, ticker)

        raw_df = await asyncio.to_thread(get_price_history, ticker, "3mo")
        df = raw_df.rename(columns=str.lower)
        ohlc = clean_ohlc(df)

        if len(ohlc) < metadata.lookback:
            raise InsufficientHistoryError(
                f"Need at least {metadata.lookback} clean candles for '{ticker}', got {len(ohlc)}"
            )

        X = self._build_window(ohlc, metadata.lookback)
        raw_return = await asyncio.to_thread(self._forward, model, X)

        current_price = float(ohlc["close"].iloc[-1])
        predicted_price = current_price * (1.0 + raw_return / 100.0)

        if raw_return > 1.5:
            trend = "alcista"
        elif raw_return < -1.5:
            trend = "bajista"
        else:
            trend = "neutral"

        logger.info("Prediction for %s: trend=%s return=%.2f%%", ticker, trend, raw_return)

        return DLResult(
            trend=trend,
            predicted_return_pct=raw_return,
            predicted_price=predicted_price,
            current_price=current_price,
            horizon_days=metadata.horizon_days,
            trained_at=datetime.fromisoformat(metadata.trained_at),
            metrics=ModelMetrics(
                rmse=metadata.metrics["rmse"],
                mae=metadata.metrics["mae"],
                directional_accuracy=metadata.metrics["directional_accuracy"],
            ),
        )
