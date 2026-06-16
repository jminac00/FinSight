"""The scheduler must only run its data-mutating refresh jobs in production.

Outside production (local dev, tests) the refresh jobs would burn external quotas
and rewrite the committed reference universes in place, so the scheduler must stay
down. Teardown must remain a safe no-op when it never started.
"""

import asyncio
from types import SimpleNamespace

from app.scheduler import jobs

_JOB_IDS = {"daily_model_update", "weekly_fundamental_refresh", "daily_technical_refresh"}


async def test_start_scheduler_disabled_outside_production(monkeypatch):
    monkeypatch.setattr(jobs, "get_settings", lambda: SimpleNamespace(environment="development"))

    jobs.start_scheduler()
    try:
        assert not jobs._scheduler.running
        assert jobs._scheduler.get_jobs() == []
    finally:
        jobs.stop_scheduler()


async def test_stop_scheduler_is_noop_when_never_started(monkeypatch):
    monkeypatch.setattr(jobs, "get_settings", lambda: SimpleNamespace(environment="development"))

    jobs.start_scheduler()
    jobs.stop_scheduler()  # must not raise even though the scheduler never started


async def test_start_scheduler_runs_in_production(monkeypatch):
    monkeypatch.setattr(jobs, "get_settings", lambda: SimpleNamespace(environment="production"))

    try:
        jobs.start_scheduler()
        assert jobs._scheduler.running
        assert {job.id for job in jobs._scheduler.get_jobs()} == _JOB_IDS
    finally:
        jobs.stop_scheduler()

    # AsyncIOScheduler.shutdown completes on the next loop turn; yield then verify.
    await asyncio.sleep(0)
    assert not jobs._scheduler.running
