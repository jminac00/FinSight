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

from app.core.config import get_settings
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
from app.services.deep_learning.recipe import load_frozen_recipe
from app.services.deep_learning.training.pipeline import train_ticker

logger = logging.getLogger(__name__)


def _period_for_days(days: int) -> str:
    """Convert a number of days into the smallest yfinance period string that covers it."""
    if days <= 365:
        return "1y"
    if days <= 730:
        return "2y"
    if days <= 1095:
        return "3y"
    return "max"


class ModelNotAvailableError(Exception):
    """Raised when no trained GRU artifact exists for a ticker."""


class ModelQualityInsufficientError(Exception):
    """Raised when a trained GRU does not beat the naive predictor by enough.

    The model is not published: its weights are discarded and only the metadata
    is kept, recording the skill ratio that ruled it out.
    """


class DLService:
    """Manages GRU model loading (lazy + LRU cache) and inference.

    Flow: yfinance EOD data → lookback window → GRU inference → 10-day return
          → trend classification (neutral band ±1.5 %).
    """

    def __init__(
        self,
        models_dir: Path = DEFAULT_MODELS_DIR,
        max_models: int = 10,
        auto_train: bool = False,
    ) -> None:
        self._models_dir = models_dir
        self._cache: LRUCache[str, tuple[nn.Module, ModelMetadata]] = LRUCache(maxsize=max_models)
        # When enabled (local only), predict() trains a missing model on demand
        # instead of failing. Kept off in production, where CPU is scarce.
        self._auto_train = auto_train

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
            ModelNotAvailableError: No trained artifact exists and auto_train is off.
            InsufficientHistoryError: yfinance returned fewer than lookback clean candles.
            ValueError: yfinance returned no data for ticker.
        """
        if not await self.is_model_available(ticker):
            if not self._auto_train:
                raise ModelNotAvailableError(f"No trained model for '{ticker}'")
            logger.info("No trained model for %s; training on demand (local)", ticker)
            await self.train(ticker)

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

    def evict(self, ticker: str) -> None:
        """Remove a model from the LRU cache (no-op if not cached)."""
        self._cache.pop(ticker, None)

    async def train(self, ticker: str, max_epochs: int = 50) -> ModelArtifacts:
        """Retrain the GRU for *ticker* and persist fresh artifacts.

        Warm-starts from existing weights when a trained model is available;
        trains from scratch otherwise. The LRU cache entry is evicted after
        saving so the next prediction loads the updated model.

        Args:
            ticker: Uppercase stock symbol.
            max_epochs: Training budget (fewer than cold-start default since
                warm-start converges faster).

        A model is only published when it beats the naive zero-return predictor
        by the margin ``DL_MAX_SKILL_RATIO`` demands. Otherwise the weights are
        dropped and only the metadata is written, so the ticker keeps a record of
        why it is not served.

        Returns:
            The newly trained :class:`ModelArtifacts`, published.

        Raises:
            ModelQualityInsufficientError: The model did not beat the baseline.
            InsufficientHistoryError: yfinance returned too few clean candles.
            ValueError: yfinance returned no data for the ticker.
        """
        from datetime import date

        recipe = load_frozen_recipe()
        initial_state_dict: dict | None = None
        period = "3y"

        was_published = await self.is_model_available(ticker)
        if was_published:
            current = await asyncio.to_thread(ModelArtifacts.load, ticker, self._models_dir)
            initial_state_dict = current.state_dict
            days_gap = (date.today() - date.fromisoformat(current.metadata.data_through)).days
            period = _period_for_days(max(days_gap + 365, 365))

        raw_df = await asyncio.to_thread(get_price_history, ticker, period)
        ohlc = clean_ohlc(raw_df.rename(columns=str.lower))

        new_artifacts = await asyncio.to_thread(
            train_ticker,
            ohlc,
            ticker,
            recipe,
            initial_state_dict=initial_state_dict,
            max_epochs=max_epochs,
        )
        threshold = get_settings().dl_max_skill_ratio
        skill_ratio = new_artifacts.metadata.skill_ratio

        if skill_ratio >= threshold:
            if was_published:
                # A worse retraining does not retire a model that already earned
                # its place: both artifacts are left exactly as they were.
                logger.warning(
                    "Retrained %s scored skill_ratio=%.3f (threshold %.2f); "
                    "keeping the previously published model",
                    ticker,
                    skill_ratio,
                    threshold,
                )
            else:
                new_artifacts.metadata.published = False
                await asyncio.to_thread(new_artifacts.save_metadata, self._models_dir)
                logger.info(
                    "Discarded %s: skill_ratio=%.3f (threshold %.2f) — metadata only",
                    ticker,
                    skill_ratio,
                    threshold,
                )
            raise ModelQualityInsufficientError(
                f"Model for '{ticker}' does not beat the naive predictor "
                f"(skill_ratio={skill_ratio:.3f}, threshold={threshold:.2f})"
            )

        new_artifacts.metadata.published = True
        await asyncio.to_thread(new_artifacts.save, self._models_dir)
        self.evict(ticker)
        logger.info(
            "Trained %s: rmse=%.4f da=%.2f%% skill_ratio=%.3f data_through=%s",
            ticker,
            new_artifacts.metadata.metrics["rmse"],
            new_artifacts.metadata.metrics["directional_accuracy"] * 100,
            skill_ratio,
            new_artifacts.metadata.data_through,
        )
        return new_artifacts
