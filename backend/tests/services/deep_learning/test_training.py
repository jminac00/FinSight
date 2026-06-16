import numpy as np
import pytest
import torch

from app.services.deep_learning.model import build_model
from app.services.deep_learning.preprocessing import INPUT_SIZE, InsufficientHistoryError
from app.services.deep_learning.recipe import load_frozen_recipe
from app.services.deep_learning.training.evaluation import evaluate
from app.services.deep_learning.training.pipeline import train_ticker


def test_build_model_forward_is_one_dimensional():
    recipe = load_frozen_recipe()
    model = build_model(recipe, INPUT_SIZE)
    x = torch.zeros(8, recipe.lookback, INPUT_SIZE)
    out = model(x)
    assert out.shape == (8,)


def test_evaluate_returns_contract_metrics():
    y_true = np.array([1.0, -2.0, 3.0, -0.5])
    y_pred = np.array([0.5, -1.0, 2.0, 0.5])
    metrics = evaluate(y_true, y_pred)
    assert set(metrics) == {"rmse", "mae", "directional_accuracy"}
    assert metrics["directional_accuracy"] == 0.75  # 3 of 4 signs agree
    assert all(np.isfinite(v) for v in metrics.values())


def test_train_ticker_is_deterministic_and_finite(make_ohlc):
    recipe = load_frozen_recipe()
    ohlc = make_ohlc(n=400, seed=7)
    a = train_ticker(ohlc, "AAPL", recipe, seed=42, max_epochs=5)
    b = train_ticker(ohlc, "AAPL", recipe, seed=42, max_epochs=5)
    assert a.metadata.metrics == b.metadata.metrics
    assert all(np.isfinite(v) for v in a.metadata.metrics.values())
    assert a.metadata.input_size == INPUT_SIZE
    assert a.metadata.horizon_days == recipe.horizon


def test_train_ticker_raises_on_short_history(make_ohlc):
    recipe = load_frozen_recipe()
    with pytest.raises(InsufficientHistoryError):
        train_ticker(make_ohlc(n=30, seed=1), "AAPL", recipe, seed=42, max_epochs=5)
