"""Tests for benchmark.optimize: search space, determinism, K selection, config IO."""

import numpy as np
import optuna
import pytest
import torch

from benchmark.optimize import (
    build_gru,
    evaluate_config,
    read_frozen_config,
    select_k_by_aliasing,
    suggest_config,
    write_frozen_config,
)


def test_suggest_config_builds_valid_gru() -> None:
    params = {
        "lr": 1e-3,
        "lookback": 60,
        "hidden_size": 64,
        "num_layers": 1,
        "dense_units": 32,
        "dropout": 0.1,
        "batch_size": 32,
    }
    trial = optuna.trial.FixedTrial(params)
    cfg = suggest_config(trial, model="gru")

    for key, value in params.items():
        assert cfg[key] == value

    input_size = 4
    model = build_gru(input_size, cfg)
    out = model(torch.randn(2, cfg["lookback"], input_size))
    assert out.shape == (2,)


def test_evaluate_config_is_deterministic_for_gru() -> None:
    rng = np.random.default_rng(0)
    x_tr = rng.standard_normal((40, 12, 4)).astype(np.float32)
    y_tr = rng.standard_normal(40).astype(np.float32)
    x_val = rng.standard_normal((12, 12, 4)).astype(np.float32)
    y_val = rng.standard_normal(12).astype(np.float32)
    cfg = {
        "lr": 1e-3,
        "lookback": 12,
        "hidden_size": 8,
        "num_layers": 1,
        "dense_units": 8,
        "dropout": 0.0,
        "batch_size": 16,
    }

    first = evaluate_config(x_tr, y_tr, x_val, y_val, cfg, seeds=(42, 43), max_epochs=4, patience=4, device="cpu")
    second = evaluate_config(x_tr, y_tr, x_val, y_val, cfg, seeds=(42, 43), max_epochs=4, patience=4, device="cpu")

    assert first == pytest.approx(second, abs=1e-9)
    assert first > 0.0


def test_select_k_by_aliasing_on_multitone_signal() -> None:
    n = 256
    t = np.arange(n, dtype=np.float64)
    # Trend plus three well-separated tones; K should settle before modes alias.
    signal = (
        100.0
        + 0.02 * t
        + 2.0 * np.sin(2 * np.pi * t / 32)
        + 1.0 * np.sin(2 * np.pi * t / 8)
        + 0.5 * np.sin(2 * np.pi * t / 4)
    )

    k = select_k_by_aliasing(signal, alpha=2000.0, k_min=3, k_max=6)
    assert isinstance(k, int)
    assert 3 <= k <= 6
    # Deterministic for a fixed signal.
    assert select_k_by_aliasing(signal, alpha=2000.0, k_min=3, k_max=6) == k


def test_frozen_config_json_roundtrip(tmp_path) -> None:
    config = {
        "model": "gru",
        "use_vmd": False,
        "horizon": 10,
        "params": {
            "lr": 1e-3,
            "lookback": 48,
            "hidden_size": 64,
            "num_layers": 1,
            "dense_units": 32,
            "dropout": 0.1,
            "batch_size": 32,
        },
        "val_rmse": 5.671,
        "tuned_on": ["AAPL", "NVDA", "PEP"],
    }
    path = tmp_path / "frozen.json"
    write_frozen_config(path, config)
    assert path.exists()
    assert read_frozen_config(path) == config
