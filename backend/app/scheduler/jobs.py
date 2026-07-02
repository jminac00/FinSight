import asyncio
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.core.config import get_settings
from app.services.deep_learning.artifacts import DEFAULT_MODELS_DIR
from app.services.fundamental.engine.msci_world import build_msci_world_universe
from app.services.fundamental.engine.universe import build_universe
from app.services.technical.engine.universe_manager import build_snapshot

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


async def daily_technical_refresh() -> None:
    """Rebuild the technical price-universe snapshots (S&P 500 + MSCI World) with fresh EOD data.

    Runs daily at 23:00 CET, after the US market close, so the technical blocks normalize against an
    up-to-date cross-section. Each universe is rebuilt independently: on failure the last-good
    snapshot is kept. build_snapshot already drops the universe manager's in-memory cache.
    """
    logger.info("daily_technical_refresh starting")
    for universe in ("sp500", "msci_world"):
        try:
            closes = await asyncio.to_thread(build_snapshot, universe)
            logger.info("Technical %s universe rebuilt: %d tickers", universe, closes.shape[1])
        except Exception:
            logger.exception(
                "Technical %s universe rebuild failed; keeping existing snapshot", universe
            )
    try:
        from app.api.v1.technical import get_technical_service

        get_technical_service.cache_clear()
    except Exception:
        logger.exception("Could not clear the technical service cache")


async def daily_model_update() -> None:
    """Retrain all available GRU models with the latest EOD market data.

    Runs daily at 22:00 CET, after the US market close.
    Uses exclusively real market data — never model-generated predictions.
    Logs outcome (tickers updated, metrics, errors) for monitoring.
    Each ticker is warm-started from its current weights; on failure the
    existing artifact is kept intact.
    """
    from app.api.v1.deep_learning import get_dl_service

    models_dir = DEFAULT_MODELS_DIR
    tickers = sorted(
        p.stem
        for p in models_dir.glob("*.pt")
        if (models_dir / f"{p.stem}.json").exists()
    )
    if not tickers:
        logger.info("daily_model_update: no models found; nothing to update")
        return

    service = get_dl_service()
    for ticker in tickers:
        try:
            artifacts = await service.train(ticker)
            logger.info(
                "daily_model_update %s: rmse=%.4f da=%.2f%% data_through=%s",
                ticker,
                artifacts.metadata.metrics["rmse"],
                artifacts.metadata.metrics["directional_accuracy"] * 100,
                artifacts.metadata.data_through,
            )
        except Exception:
            logger.exception(
                "daily_model_update: failed for %s; keeping existing artifact", ticker
            )


def start_scheduler() -> None:
    """Register the scheduled jobs and start the APScheduler — production only.

    The refresh jobs rebuild the universes in place from external data sources.
    Outside production (local dev, tests) that would burn API quotas and overwrite
    the committed reference snapshots, so the scheduler stays down there; seed data
    manually with the ``scripts/build_*`` commands instead.
    """
    settings = get_settings()
    if settings.environment != "production":
        logger.info(
            "APScheduler disabled (ENVIRONMENT=%s) — refresh jobs run in production only",
            settings.environment,
        )
        return

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
    _scheduler.add_job(
        daily_technical_refresh,
        trigger=CronTrigger(hour=23, minute=0, timezone="Europe/Madrid"),
        id="daily_technical_refresh",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info(
        "APScheduler started — daily_model_update at 22:00 CET, "
        "daily_technical_refresh at 23:00 CET, weekly_fundamental_refresh on Sunday 04:00 CET"
    )


def stop_scheduler() -> None:
    """Gracefully shut down the APScheduler if it is running (no-op otherwise)."""
    if not _scheduler.running:
        return
    _scheduler.shutdown(wait=False)
    logger.info("APScheduler stopped")
