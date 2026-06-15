import logging

from app.models.deep_learning import DLResult

logger = logging.getLogger(__name__)


class DLService:
    """Manages GRU model loading (lazy + LRU cache) and inference.

    Flow: Finnhub EOD data → lookback window → GRU inference → 10-day return
          → trend classification (neutral band).
    """

    async def predict(self, ticker: str) -> DLResult:
        """Generate a 10-day return prediction for the given ticker.

        Args:
            ticker: Uppercase stock symbol (e.g. 'AAPL').

        Returns:
            DLResult with trend, predicted return and derived price, current
            price, horizon days, training date and model quality metrics.
        """
        # TODO: check ml_models/{ticker}.pt and {ticker}.json exist
        # TODO: load model lazily with LRU cache (max LRU_CACHE_MAX_MODELS)
        # TODO: fetch last `lookback` EOD candles from Finnhub
        # TODO: run GRU forward pass (torch.no_grad()) → 10-day return
        # TODO: classify trend with the neutral band; derive predicted_price
        raise NotImplementedError("DLService.predict is not yet implemented")

    async def is_model_available(self, ticker: str) -> bool:
        """Return True if a trained GRU model exists for the ticker.

        Args:
            ticker: Uppercase stock symbol.
        """
        # TODO: check existence of ml_models/{ticker}.pt
        raise NotImplementedError("DLService.is_model_available is not yet implemented")
