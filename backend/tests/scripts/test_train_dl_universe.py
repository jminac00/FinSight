"""Tests for the S&P 500 batch trainer (network-free).

The universe is stubbed at ``load_universe`` and training is stubbed at
``DLService.train``: nothing here downloads prices or fits a GRU.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from app.services.deep_learning.service import ModelQualityInsufficientError
from scripts import train_dl_universe as script

_LOAD_UNIVERSE = "app.services.deep_learning.coverage.load_universe"


@pytest.fixture(autouse=True)
def stub_universe():
    """A five-ticker universe, rebuilt for each test."""
    from app.services.deep_learning import coverage

    coverage.universe_tickers.cache_clear()
    frame = pd.DataFrame({"ticker": ["AAPL", "MSFT", "NVDA", "PEP", "XOM"]})
    with patch(_LOAD_UNIVERSE, return_value=frame):
        yield
    coverage.universe_tickers.cache_clear()


def _published(models_dir: Path, ticker: str, skill_ratio: float = 0.8) -> None:
    """Lay down a published artifact pair."""
    (models_dir / f"{ticker}.pt").write_bytes(b"weights")
    (models_dir / f"{ticker}.json").write_text(
        json.dumps({"ticker": ticker, "skill_ratio": skill_ratio, "published": True}),
        encoding="utf-8",
    )


def _discarded(models_dir: Path, ticker: str, skill_ratio: float = 1.3) -> None:
    """Lay down the metadata-only artifact of a discarded model."""
    (models_dir / f"{ticker}.json").write_text(
        json.dumps({"ticker": ticker, "skill_ratio": skill_ratio, "published": False}),
        encoding="utf-8",
    )


def _service(trained: list[str], *, side_effect=None) -> MagicMock:
    """A DLService double recording each trained ticker."""
    service = MagicMock()

    async def _train(ticker, *args, **kwargs):
        trained.append(ticker)
        if side_effect is not None:
            outcome = side_effect(ticker)
            if isinstance(outcome, Exception):
                raise outcome
        artifacts = MagicMock()
        artifacts.metadata.skill_ratio = 0.75
        return artifacts

    service.train = AsyncMock(side_effect=_train)
    return service


def _run(models_dir: Path, service: MagicMock, argv: list[str]) -> None:
    """Invoke the script's entry point with patched models dir and service."""
    with patch.object(script, "DEFAULT_MODELS_DIR", models_dir):
        with patch.object(script, "DLService", return_value=service):
            with patch.object(script.sys, "argv", ["train_dl_universe", *argv]):
                script.main()


# ---------------------------------------------------------------------------
# Ticker source
# ---------------------------------------------------------------------------


def test_tickers_come_from_the_shared_universe(tmp_path):
    trained: list[str] = []
    _run(tmp_path, _service(trained), [])
    assert trained == ["AAPL", "MSFT", "NVDA", "PEP", "XOM"]


# ---------------------------------------------------------------------------
# Idempotence and resumability
# ---------------------------------------------------------------------------


def test_skips_tickers_with_a_published_model(tmp_path):
    _published(tmp_path, "AAPL")
    trained: list[str] = []
    _run(tmp_path, _service(trained), [])
    assert "AAPL" not in trained


def test_skips_tickers_already_discarded_on_quality(tmp_path):
    _discarded(tmp_path, "MSFT")
    trained: list[str] = []
    _run(tmp_path, _service(trained), [])
    assert "MSFT" not in trained


def test_force_retrains_everything(tmp_path):
    _published(tmp_path, "AAPL")
    _discarded(tmp_path, "MSFT")
    trained: list[str] = []
    _run(tmp_path, _service(trained), ["--force"])
    assert trained == ["AAPL", "MSFT", "NVDA", "PEP", "XOM"]


def test_metadata_without_a_pt_is_retrained(tmp_path):
    """A published flag with no weights beside it is not a finished ticker."""
    (tmp_path / "NVDA.json").write_text(json.dumps({"published": True}), encoding="utf-8")
    trained: list[str] = []
    _run(tmp_path, _service(trained), [])
    assert "NVDA" in trained


def test_limit_caps_the_tickers_processed(tmp_path):
    trained: list[str] = []
    _run(tmp_path, _service(trained), ["--limit", "2"])
    assert trained == ["AAPL", "MSFT"]


def test_limit_resumes_where_the_previous_run_stopped(tmp_path):
    _published(tmp_path, "AAPL")
    _published(tmp_path, "MSFT")
    trained: list[str] = []
    _run(tmp_path, _service(trained), ["--limit", "2"])
    assert trained == ["NVDA", "PEP"]


# ---------------------------------------------------------------------------
# --tickers
# ---------------------------------------------------------------------------


def test_tickers_trains_only_the_given_symbols(tmp_path):
    trained: list[str] = []
    _run(tmp_path, _service(trained), ["--tickers", "NVDA,PEP"])
    assert trained == ["NVDA", "PEP"]


def test_tickers_combines_with_force(tmp_path):
    _published(tmp_path, "NVDA")
    trained: list[str] = []
    _run(tmp_path, _service(trained), ["--tickers", "NVDA,PEP", "--force"])
    assert trained == ["NVDA", "PEP"]


def test_tickers_without_force_still_skips_resolved_symbols(tmp_path):
    _published(tmp_path, "NVDA")
    trained: list[str] = []
    _run(tmp_path, _service(trained), ["--tickers", "NVDA,PEP"])
    assert trained == ["PEP"]


def test_tickers_with_clean_aborts_with_an_error(tmp_path, capsys):
    (tmp_path / ".gitkeep").write_text("", encoding="utf-8")
    _published(tmp_path, "AAPL")
    trained: list[str] = []

    with pytest.raises(SystemExit):
        _run(tmp_path, _service(trained), ["--tickers", "NVDA", "--clean", "--yes"])

    assert trained == []
    assert (tmp_path / "AAPL.pt").exists()
    assert "cannot be combined" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Resilience
# ---------------------------------------------------------------------------


def test_a_failing_ticker_does_not_abort_the_run(tmp_path):
    trained: list[str] = []
    service = _service(
        trained, side_effect=lambda t: RuntimeError("yfinance timeout") if t == "MSFT" else None
    )
    _run(tmp_path, service, [])
    assert trained == ["AAPL", "MSFT", "NVDA", "PEP", "XOM"]


def test_a_discarded_ticker_does_not_abort_the_run(tmp_path):
    trained: list[str] = []
    service = _service(
        trained,
        side_effect=lambda t: (
            ModelQualityInsufficientError("skill_ratio=1.310") if t == "MSFT" else None
        ),
    )
    _run(tmp_path, service, [])
    assert trained == ["AAPL", "MSFT", "NVDA", "PEP", "XOM"]


def test_summary_counts_each_outcome(tmp_path, caplog):
    trained: list[str] = []

    def outcome(ticker):
        if ticker == "MSFT":
            return ModelQualityInsufficientError("skill_ratio=1.310")
        if ticker == "NVDA":
            return RuntimeError("boom")
        return None

    with caplog.at_level("INFO"):
        _run(tmp_path, _service(trained, side_effect=outcome), [])

    summary = [m for m in (r.getMessage() for r in caplog.records) if "published=" in m][-1]
    assert "published=3" in summary
    assert "discarded=1" in summary
    assert "errors=1" in summary


def test_progress_log_reports_index_total_and_skill_ratio(tmp_path, caplog):
    trained: list[str] = []
    with caplog.at_level("INFO"):
        _run(tmp_path, _service(trained), ["--limit", "1"])

    messages = [r.getMessage() for r in caplog.records]
    assert any("[1/1]" in m and "AAPL" in m and "0.75" in m for m in messages)


# ---------------------------------------------------------------------------
# --clean
# ---------------------------------------------------------------------------


def test_clean_removes_artifacts_but_keeps_gitkeep(tmp_path):
    (tmp_path / ".gitkeep").write_text("", encoding="utf-8")
    _published(tmp_path, "AAPL")
    _discarded(tmp_path, "MSFT")

    removed = script.clean_artifacts(tmp_path)

    assert removed == 3
    assert (tmp_path / ".gitkeep").exists()
    assert not list(tmp_path.glob("*.pt"))
    assert not list(tmp_path.glob("*.json"))


def test_clean_asks_for_confirmation_and_aborts_on_no(tmp_path):
    (tmp_path / ".gitkeep").write_text("", encoding="utf-8")
    _published(tmp_path, "AAPL")
    trained: list[str] = []

    with patch.object(script, "input", create=True, return_value="n"):
        _run(tmp_path, _service(trained), ["--clean"])

    assert (tmp_path / "AAPL.pt").exists()
    assert trained == []


def test_clean_proceeds_when_confirmed(tmp_path):
    (tmp_path / ".gitkeep").write_text("", encoding="utf-8")
    _published(tmp_path, "AAPL")
    trained: list[str] = []

    with patch.object(script, "input", create=True, return_value="y"):
        _run(tmp_path, _service(trained), ["--clean"])

    assert not (tmp_path / "AAPL.pt").exists()
    assert trained == ["AAPL", "MSFT", "NVDA", "PEP", "XOM"]


def test_yes_skips_the_confirmation_prompt(tmp_path):
    (tmp_path / ".gitkeep").write_text("", encoding="utf-8")
    _published(tmp_path, "AAPL")
    trained: list[str] = []

    def _refuse(*args, **kwargs):
        raise AssertionError("--yes must not prompt")

    with patch.object(script, "input", create=True, side_effect=_refuse):
        _run(tmp_path, _service(trained), ["--clean", "--yes"])

    assert not (tmp_path / "AAPL.pt").exists()
    assert trained == ["AAPL", "MSFT", "NVDA", "PEP", "XOM"]


def test_clean_reports_the_removed_count_in_the_summary(tmp_path, caplog):
    (tmp_path / ".gitkeep").write_text("", encoding="utf-8")
    _published(tmp_path, "AAPL")
    trained: list[str] = []

    with caplog.at_level("INFO"):
        with patch.object(script, "input", create=True, return_value="y"):
            _run(tmp_path, _service(trained), ["--clean"])

    summary = [m for m in (r.getMessage() for r in caplog.records) if "published=" in m][-1]
    assert "removed=2" in summary
