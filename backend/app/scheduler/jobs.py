import asyncio
import logging
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.core.config import get_settings
from app.services.fundamental.engine.universe import DEFAULT_UNIVERSE_PATH, build_universe

logger = logging.getLogger(__name__)

_scheduler = AsyncIOScheduler(timezone="Europe/Madrid")


def _universe_path() -> Path:
    """Resolve the fundamental universe snapshot path from settings (or the engine default)."""
    settings = get_settings()
    return (
        Path(settings.fundamental_universe_path)
        if settings.fundamental_universe_path
        else DEFAULT_UNIVERSE_PATH
    )


async def daily_universe_refresh() -> None:
    """Rebuild the S&P 500 fundamental universe snapshot with the latest market data.

    Runs daily at 22:30 CET (after the US market close, staggered behind the model update).
    Drops the cached FundamentalService so the next request reads the fresh snapshot.
    """
    path = _universe_path()
    logger.info("daily_universe_refresh starting → %s", path)
    try:
        df = await asyncio.to_thread(build_universe, path)
        from app.api.v1.fundamental import get_fundamental_service

        get_fundamental_service.cache_clear()
        logger.info("daily_universe_refresh completed: %d companies", len(df))
    except Exception:
        logger.exception("daily_universe_refresh failed")


async def warm_universe_if_missing() -> None:
    """Build the universe snapshot in the background if it is absent (fresh deploy).

    Render's filesystem is ephemeral per deploy, so the snapshot may not exist until the next
    scheduled refresh. Building it on startup keeps the fundamental endpoint usable sooner;
    until it finishes, the endpoint returns 503 (UniverseNotReadyError).
    """
    path = _universe_path()
    if path.exists():
        return
    logger.info("Universe snapshot missing at %s — building in background", path)
    await daily_universe_refresh()


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
    """Register the daily jobs and start the APScheduler."""
    _scheduler.add_job(
        daily_model_update,
        trigger=CronTrigger(hour=22, minute=0, timezone="Europe/Madrid"),
        id="daily_model_update",
        replace_existing=True,
    )
    _scheduler.add_job(
        daily_universe_refresh,
        trigger=CronTrigger(hour=22, minute=30, timezone="Europe/Madrid"),
        id="daily_universe_refresh",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info(
        "APScheduler started — daily_model_update at 22:00 CET, daily_universe_refresh at 22:30 CET"
    )


def stop_scheduler() -> None:
    """Gracefully shut down the APScheduler."""
    _scheduler.shutdown(wait=False)
    logger.info("APScheduler stopped")
