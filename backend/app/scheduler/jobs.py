import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)

_scheduler = AsyncIOScheduler(timezone="Europe/Madrid")


async def daily_model_update() -> None:
    """Retrain all available GRU models with the latest EOD market data.

    Runs daily at 22:00 CET, after the US market close.
    Uses exclusively real market data — never model-generated predictions.
    Logs outcome (tickers updated, metrics, errors) for monitoring.
    """
    # TODO: list all .pt files in ml_models/
    # TODO: for each ticker, fetch latest EOD data from Finnhub
    # TODO: refit the GRU on the updated history (frozen recipe)
    # TODO: replace old .pt and update .json metadata
    # TODO: log success/failure per ticker with updated metrics
    logger.info("daily_model_update triggered — implementation pending")


def start_scheduler() -> None:
    """Register the daily retraining job and start the APScheduler."""
    _scheduler.add_job(
        daily_model_update,
        trigger=CronTrigger(hour=22, minute=0, timezone="Europe/Madrid"),
        id="daily_model_update",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info("APScheduler started — daily_model_update scheduled at 22:00 CET")


def stop_scheduler() -> None:
    """Gracefully shut down the APScheduler."""
    _scheduler.shutdown(wait=False)
    logger.info("APScheduler stopped")
