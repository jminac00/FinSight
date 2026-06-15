"""Tests for the per-mode multi-branch regressor and its runner wiring."""

import numpy as np
import pytest
import torch

from benchmark.models import VMDMultiBranchRegressor, predict, train_model
from benchmark.run import make_network

LOOKBACK = 12
N_MODES = 4
N_AUX = 3
HIDDEN = 8


def _model(input_size: int = N_MODES + N_AUX, n_modes: int = N_MODES) -> VMDMultiBranchRegressor:
    return VMDMultiBranchRegressor(input_size, n_modes=n_modes, rnn_type="gru", hidden_size=HIDDEN)


def test_output_shape() -> None:
    torch.manual_seed(0)
    model = _model()
    x = torch.randn(5, LOOKBACK, N_MODES + N_AUX)
    out = model(x)
    assert out.shape == (5,)
    assert torch.isfinite(out).all()


def test_branch_structure() -> None:
    model = _model()
    assert len(model.mode_branches) == N_MODES
    assert model.aux_branch is not None
    # The head consumes one hidden vector per branch (modes + the single aux branch).
    first_linear = model.head[0]
    assert first_linear.in_features == HIDDEN * (N_MODES + 1)


def test_no_aux_branch() -> None:
    model = _model(input_size=N_MODES, n_modes=N_MODES)
    assert model.aux_branch is None
    assert model.head[0].in_features == HIDDEN * N_MODES
    out = model(torch.randn(3, LOOKBACK, N_MODES))
    assert out.shape == (3,)


def test_gradient_reaches_every_mode_branch() -> None:
    """Every mode must be wired in: a loss must produce a gradient in each branch."""
    torch.manual_seed(0)
    model = _model()
    model.train()
    x = torch.randn(6, LOOKBACK, N_MODES + N_AUX)
    target = torch.randn(6)
    torch.nn.functional.mse_loss(model(x), target).backward()

    for i, branch in enumerate(model.mode_branches):
        grads = [p.grad for p in branch.parameters() if p.grad is not None]
        assert grads, f"mode branch {i} received no gradient"
        assert any(g.abs().sum().item() > 0 for g in grads), f"mode branch {i} has zero gradient"


@pytest.mark.parametrize("n_modes", [0, N_MODES + N_AUX + 1])
def test_invalid_n_modes(n_modes: int) -> None:
    with pytest.raises(ValueError):
        VMDMultiBranchRegressor(N_MODES + N_AUX, n_modes=n_modes)


def test_make_network_builds_multibranch() -> None:
    model = make_network("gru-mb", n_features=N_MODES + N_AUX, lookback=LOOKBACK, n_modes=N_MODES)
    assert isinstance(model, VMDMultiBranchRegressor)
    assert len(model.mode_branches) == N_MODES


def test_trains_and_predicts() -> None:
    """End-to-end on tiny synthetic data: the model integrates with train/predict."""
    rng = np.random.default_rng(0)
    n = 80
    x = rng.standard_normal((n, LOOKBACK, N_MODES + N_AUX)).astype(np.float32)
    y = x[:, -1, 0].astype(np.float32)  # learnable signal from the first mode
    model = _model()
    train_model(model, x[:50], y[:50], x[50:65], y[50:65], max_epochs=3, device="cpu")
    preds = predict(model, x[65:], device="cpu")
    assert preds.shape == (15,)
    assert np.isfinite(preds).all()
