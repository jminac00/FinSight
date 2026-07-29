"""Orchestrates one ticker's training: data -> windows -> fit -> evaluate ->
artifacts. Depends only on a cleaned OHLC frame and the recipe, not on how the
data was obtained (the CLI/loader supplies the frame).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import torch

from app.services.deep_learning.artifacts import ModelArtifacts, ModelMetadata
from app.services.deep_learning.model import build_model, predict
from app.services.deep_learning.preprocessing import (
    INPUT_SIZE,
    InsufficientHistoryError,
    chrono_split,
    make_dataset,
)
from app.services.deep_learning.recipe import GRURecipe
from app.services.deep_learning.training.evaluation import evaluate, naive_rmse
from app.services.deep_learning.training.trainer import fit


def train_ticker(
    ohlc: pd.DataFrame,
    ticker: str,
    recipe: GRURecipe,
    *,
    seed: int = 42,
    max_epochs: int = 100,
    patience: int = 10,
    device: str = "cpu",
    initial_state_dict: dict | None = None,
) -> ModelArtifacts:
    """Train a GRU for one ticker and return its artifacts (not yet persisted).

    When *initial_state_dict* is provided the model is warm-started from those
    weights instead of random initialisation (useful for daily incremental updates).
    Raises InsufficientHistoryError if the history cannot fill all three splits.

    The returned metadata carries ``skill_ratio`` (the model's RMSE relative to
    the zero-return baseline) but is always ``published=False``: deciding whether
    the ratio is good enough belongs to the service, which owns the threshold.
    """
    dataset = make_dataset(ohlc, recipe)
    train, val, test = chrono_split(dataset)
    if min(len(train.y), len(val.y), len(test.y)) == 0:
        raise InsufficientHistoryError(
            f"{ticker}: {len(dataset.y)} samples do not fill train/val/test splits"
        )

    # Seed before building the model so weight initialization is reproducible
    # regardless of any prior global RNG use; fit re-seeds the training loop.
    torch.manual_seed(seed)
    model = build_model(recipe, INPUT_SIZE)
    if initial_state_dict is not None:
        model.load_state_dict(initial_state_dict, strict=True)
    fit(
        model,
        train.X,
        train.y,
        val.X,
        val.y,
        lr=recipe.lr,
        batch_size=recipe.batch_size,
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
        device=device,
    )
    metrics = evaluate(test.y, predict(model, test.X, device=device))
    # Score the model against the zero-return baseline on the very same split,
    # so the ratio compares like with like.
    metrics["rmse_naive"] = naive_rmse(test.y)
    skill_ratio = (
        metrics["rmse"] / metrics["rmse_naive"] if metrics["rmse_naive"] > 0 else float("inf")
    )

    metadata = ModelMetadata(
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
        metrics=metrics,
        n_samples=len(dataset.y),
        data_through=dataset.t_index[-1].date().isoformat(),
        skill_ratio=skill_ratio,
        published=False,  # a candidate until the service's quality gate promotes it
    )
    return ModelArtifacts(ticker=ticker, state_dict=model.state_dict(), metadata=metadata)
