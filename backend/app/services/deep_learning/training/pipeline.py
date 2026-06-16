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
from app.services.deep_learning.training.evaluation import evaluate
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
) -> ModelArtifacts:
    """Train a GRU for one ticker and return its artifacts (not yet persisted).

    Raises InsufficientHistoryError if the history cannot fill all three splits.
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
    )
    return ModelArtifacts(ticker=ticker, state_dict=model.state_dict(), metadata=metadata)
