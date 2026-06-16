import asyncio
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.services.fundamental.engine.msci_world import build_msci_world_universe
from app.services.fundamental.engine.universe import build_universe

logger = logging.getLogger(__name__)

_scheduler = AsyncIOScheduler(timezone="Europe/Madrid")


async def weekly_fundamental_refresh() -> None:
    """Rebuild the fundamental universe snapshots (S&P 500 + MSCI World) with fresh data.

    Runs weekly (Sunday ~04:00 CET): fundamentals change at a quarterly cadence and index
    membership only a few times a year, so a weekly rebuild keeps the snapshots current without
    abusing the data source. Each rebuild is resilient: on failure the committed/last-good CSV is
    kept. Afterwards the cached FundamentalService is dropped so new requests read the fresh data.
    """
    logger.info("weekly_fundamental_refresh starting")
    try:
        df = await asyncio.to_thread(build_universe)
        logger.info("S&P 500 universe rebuilt: %d companies", len(df))
    except Exception:
        logger.exception("S&P 500 universe rebuild failed; keeping existing snapshot")
    try:
        mdf = await asyncio.to_thread(build_msci_world_universe)
        logger.info("MSCI World universe rebuilt: %d companies", len(mdf))
    except Exception:
        logger.exception("MSCI World universe rebuild failed; keeping existing snapshot")
    try:
        from app.api.v1.fundamental import get_fundamental_service

        get_fundamental_service.cache_clear()
    except Exception:
        logger.exception("Could not clear the fundamental service cache")


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
    """Register the scheduled jobs and start the APScheduler."""
    _scheduler.add_job(
        daily_model_update,
        trigger=CronTrigger(hour=22, minute=0, timezone="Europe/Madrid"),
        id="daily_model_update",
        replace_existing=True,
    )
    _scheduler.add_job(
        weekly_fundamental_refresh,
        trigger=CronTrigger(day_of_week="sun", hour=4, minute=0, timezone="Europe/Madrid"),
        id="weekly_fundamental_refresh",
        replace_existing=True,
    )
    # NOTE: the daily technical-universe (prices) refresh belongs to the technical module (Phase 2)
    # and will be registered here when that module is integrated.
    _scheduler.start()
    logger.info(
        "APScheduler started — daily_model_update at 22:00 CET, "
        "weekly_fundamental_refresh on Sunday 04:00 CET"
    )


def stop_scheduler() -> None:
    """Gracefully shut down the APScheduler."""
    _scheduler.shutdown(wait=False)
    logger.info("APScheduler stopped")
