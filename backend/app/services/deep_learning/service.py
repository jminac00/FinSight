import logging

from app.models.deep_learning import DLResult

logger = logging.getLogger(__name__)


class DLService:
    """Manages LSTM model loading (lazy + LRU cache) and inference.

    Flow: Finnhub EOD data → VMD preprocessing (params from .json metadata)
          → LSTM inference → trend classification.
    """

    async def predict(self, ticker: str) -> DLResult:
        """Generate a price trend prediction for the given ticker.

        Args:
            ticker: Uppercase stock symbol (e.g. 'AAPL').

        Returns:
            DLResult with trend, predicted/current price, pct_change,
            horizon days, training date and model quality metrics.
        """
        # TODO: check ml_models/{ticker}.pt and {ticker}.json exist
        # TODO: load model lazily with LRU cache (max LRU_CACHE_MAX_MODELS)
        # TODO: fetch last N EOD candles from Finnhub
        # TODO: apply VMD preprocessing with params from metadata JSON
        # TODO: run LSTM forward pass (torch.no_grad())
        # TODO: compute trend and pct_change vs current close
        raise NotImplementedError("DLService.predict is not yet implemented")

    async def is_model_available(self, ticker: str) -> bool:
        """Return True if a trained LSTM model exists for the ticker.

        Args:
            ticker: Uppercase stock symbol.
        """
        # TODO: check existence of ml_models/{ticker}.pt
        raise NotImplementedError("DLService.is_model_available is not yet implemented")
