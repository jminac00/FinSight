"""Quality gate: a GRU is only published when it beats the naive predictor.

The naive baseline is a zero return (driftless random walk) and the decision
variable is ``skill_ratio = rmse_model / rmse_naive`` on the same evaluation
split. Training is stubbed out here so each test can pin an exact ratio; the
real ``train_ticker`` wiring is covered in ``test_training.py``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from app.services.deep_learning.artifacts import ModelArtifacts, ModelMetadata
from app.services.deep_learning.preprocessing import INPUT_SIZE
from app.services.deep_learning.recipe import load_frozen_recipe
from app.services.deep_learning.service import DLService, ModelQualityInsufficientError
from app.services.deep_learning.training.evaluation import evaluate, naive_rmse

_SERVICE = "app.services.deep_learning.service"


def _metadata(ticker: str, skill_ratio: float, *, rmse: float = 5.0) -> ModelMetadata:
    """Build metadata with a pinned skill_ratio, as train_ticker would return it."""
    recipe = load_frozen_recipe()
    return ModelMetadata(
        ticker=ticker,
        trained_at=datetime.now(UTC).isoformat(),
        horizon_days=recipe.horizon,
        lookback=recipe.lookback,
        input_size=INPUT_SIZE,
        recipe={
            "hidden_size": recipe.hidden_size,
            "num_layers": recipe.num_layers,
            "dense_units": recipe.dense_units,
            "dropout": recipe.dropout,
        },
        metrics={
            "rmse": rmse,
            "mae": rmse * 0.8,
            "directional_accuracy": 0.5,
            "rmse_naive": rmse / skill_ratio,
        },
        n_samples=300,
        data_through="2026-07-01",
        skill_ratio=skill_ratio,
        published=False,  # a fresh model is a candidate until the gate promotes it
    )


def _artifacts(ticker: str, skill_ratio: float) -> ModelArtifacts:
    """A ModelArtifacts whose state dict is cheap to write, with a pinned ratio."""
    return ModelArtifacts(
        ticker=ticker,
        state_dict={"weight": torch.zeros(2)},
        metadata=_metadata(ticker, skill_ratio),
    )


def _ohlc(n: int = 400, seed: int = 3) -> pd.DataFrame:
    """Synthetic uppercase-column OHLC frame, as get_price_history returns it."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, n)))
    open_ = close * (1.0 + rng.normal(0.0, 0.005, n))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1e6},
        index=pd.bdate_range("2024-01-01", periods=n),
    )


@pytest.fixture
def gated_service(tmp_path):
    """A DLService whose training returns a caller-controlled skill_ratio.

    Yields ``(service, set_ratio)``; ``set_ratio`` pins the ratio the stubbed
    ``train_ticker`` will report on the next ``train()`` call.
    """
    service = DLService(models_dir=tmp_path, max_models=2)
    state: dict[str, float] = {"skill_ratio": 0.5}

    def _fake_train_ticker(ohlc, ticker, recipe, **kwargs):
        return _artifacts(ticker, state["skill_ratio"])

    with (
        patch(f"{_SERVICE}.get_price_history", return_value=_ohlc()),
        patch(f"{_SERVICE}.train_ticker", side_effect=_fake_train_ticker),
    ):
        yield service, lambda ratio: state.update(skill_ratio=ratio)


# ---------------------------------------------------------------------------
# naive_rmse — the zero-return baseline
# ---------------------------------------------------------------------------


def test_naive_rmse_matches_evaluate_against_a_zero_prediction():
    """The baseline reuses evaluate(); it must not re-derive the RMSE formula."""
    y_true = np.array([2.0, -3.0, 1.5, 0.0, -0.5])
    assert naive_rmse(y_true) == pytest.approx(evaluate(y_true, np.zeros_like(y_true))["rmse"])


def test_naive_rmse_is_the_quadratic_mean_of_the_true_returns():
    y_true = np.array([3.0, -4.0])
    assert naive_rmse(y_true) == pytest.approx(np.sqrt((9.0 + 16.0) / 2))


# ---------------------------------------------------------------------------
# Persistence: only models that beat the baseline get a .pt
# ---------------------------------------------------------------------------


async def test_model_worse_than_naive_persists_only_the_json(gated_service, tmp_path):
    service, set_ratio = gated_service
    set_ratio(1.3)  # 30 % worse than doing nothing

    with pytest.raises(ModelQualityInsufficientError):
        await service.train("BAD")

    assert not (tmp_path / "BAD.pt").exists()
    assert (tmp_path / "BAD.json").exists()
    assert json.loads((tmp_path / "BAD.json").read_text())["published"] is False


async def test_model_better_than_naive_persists_both_files(gated_service, tmp_path):
    service, set_ratio = gated_service
    set_ratio(0.85)

    artifacts = await service.train("GOOD")

    assert (tmp_path / "GOOD.pt").exists()
    assert (tmp_path / "GOOD.json").exists()
    assert json.loads((tmp_path / "GOOD.json").read_text())["published"] is True
    assert artifacts.metadata.published is True


async def test_discard_error_reports_the_obtained_ratio(gated_service):
    service, set_ratio = gated_service
    set_ratio(1.42)

    with pytest.raises(ModelQualityInsufficientError, match="1.42"):
        await service.train("BAD")


async def test_ratio_equal_to_the_threshold_is_not_published(gated_service, tmp_path):
    """The gate is strict: publishing requires beating the threshold, not tying it."""
    service, set_ratio = gated_service
    set_ratio(1.0)

    with pytest.raises(ModelQualityInsufficientError):
        await service.train("TIE")

    assert not (tmp_path / "TIE.pt").exists()


# ---------------------------------------------------------------------------
# The threshold comes from settings
# ---------------------------------------------------------------------------


async def test_threshold_is_read_from_settings(gated_service, tmp_path):
    """The same ratio is published or discarded depending only on the setting."""
    service, set_ratio = gated_service
    set_ratio(0.95)

    with patch(f"{_SERVICE}.get_settings", return_value=MagicMock(dl_max_skill_ratio=0.90)):
        with pytest.raises(ModelQualityInsufficientError):
            await service.train("TIGHT")
    assert not (tmp_path / "TIGHT.pt").exists()

    with patch(f"{_SERVICE}.get_settings", return_value=MagicMock(dl_max_skill_ratio=1.0)):
        await service.train("LOOSE")
    assert (tmp_path / "LOOSE.pt").exists()


def test_settings_expose_the_threshold_with_a_permissive_default():
    from app.core.config import Settings

    assert Settings().dl_max_skill_ratio == 1.0


# ---------------------------------------------------------------------------
# A failed retraining never retires an already published model
# ---------------------------------------------------------------------------


async def test_failed_retraining_keeps_the_published_artifact(gated_service, tmp_path):
    service, set_ratio = gated_service
    set_ratio(0.80)
    await service.train("AAPL")
    published_bytes = (tmp_path / "AAPL.pt").read_bytes()
    published_meta = (tmp_path / "AAPL.json").read_text()

    set_ratio(1.5)
    with pytest.raises(ModelQualityInsufficientError):
        await service.train("AAPL")

    # Neither artifact is touched: the ticker stays available with its old model.
    assert (tmp_path / "AAPL.pt").read_bytes() == published_bytes
    assert (tmp_path / "AAPL.json").read_text() == published_meta
    assert json.loads((tmp_path / "AAPL.json").read_text())["published"] is True


async def test_failed_retraining_of_published_model_is_logged_as_warning(
    gated_service, tmp_path, caplog
):
    service, set_ratio = gated_service
    set_ratio(0.80)
    await service.train("AAPL")

    set_ratio(1.5)
    with caplog.at_level("WARNING"):
        with pytest.raises(ModelQualityInsufficientError):
            await service.train("AAPL")

    assert any("AAPL" in record.message % record.args for record in caplog.records)


# ---------------------------------------------------------------------------
# Backwards compatibility with artifacts written before the gate existed
# ---------------------------------------------------------------------------


def test_legacy_metadata_without_the_new_fields_loads(tmp_path):
    """Artifacts trained before the gate have no published/skill_ratio/rmse_naive."""
    legacy = {
        "ticker": "MSFT",
        "trained_at": "2026-07-01T22:00:00+00:00",
        "horizon_days": 10,
        "lookback": 24,
        "input_size": INPUT_SIZE,
        "recipe": {"hidden_size": 32, "num_layers": 1, "dense_units": 16, "dropout": 0.1},
        "metrics": {"rmse": 5.61, "mae": 4.39, "directional_accuracy": 0.6},
        "n_samples": 300,
        "data_through": "2026-07-01",
    }
    (tmp_path / "MSFT.json").write_text(json.dumps(legacy), encoding="utf-8")
    torch.save({"weight": torch.zeros(2)}, tmp_path / "MSFT.pt")

    loaded = ModelArtifacts.load("MSFT", tmp_path)

    assert loaded.metadata.ticker == "MSFT"
    assert loaded.metadata.skill_ratio is None
    # A model already on disk was published by definition — it keeps being served.
    assert loaded.metadata.published is True


# ---------------------------------------------------------------------------
# save_metadata — the discarded-model persistence path
# ---------------------------------------------------------------------------


def test_save_metadata_writes_only_the_json(tmp_path):
    artifacts = _artifacts("NVDA", 1.2)
    artifacts.metadata.published = False

    json_path = artifacts.save_metadata(tmp_path)

    assert json_path == tmp_path / "NVDA.json"
    assert json_path.exists()
    assert not (tmp_path / "NVDA.pt").exists()


def test_save_still_writes_both_files(tmp_path):
    artifacts = _artifacts("NVDA", 0.7)

    pt_path, json_path = artifacts.save(tmp_path)

    assert pt_path.exists() and json_path.exists()


# ---------------------------------------------------------------------------
# train_ticker wires the baseline into the metadata
# ---------------------------------------------------------------------------


def test_train_ticker_records_rmse_naive_and_skill_ratio(make_ohlc):
    from app.services.deep_learning.training.pipeline import train_ticker

    artifacts = train_ticker(make_ohlc(n=400, seed=11), "PEP", load_frozen_recipe(), max_epochs=3)
    metrics = artifacts.metadata.metrics

    assert "rmse_naive" in metrics
    assert artifacts.metadata.skill_ratio == pytest.approx(metrics["rmse"] / metrics["rmse_naive"])
    # The pipeline does not publish: only the service, which owns the threshold, does.
    assert artifacts.metadata.published is False
