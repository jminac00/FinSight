"""The frozen GRU recipe — the production hyperparameter contract.

Loaded from the vendored ``gru_frozen_config.json`` (a copy of the Optuna study
output, ADR-0006). It is an immutable value object shared by preprocessing, the
model and the trainer; production refits the weights per ticker but never the
recipe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parent / "gru_frozen_config.json"


@dataclass(frozen=True)
class GRURecipe:
    """Immutable GRU architecture and training recipe."""

    lookback: int
    hidden_size: int
    num_layers: int
    dense_units: int
    dropout: float
    lr: float
    batch_size: int
    horizon: int


def load_frozen_recipe(path: Path = _CONFIG_PATH) -> GRURecipe:
    """Load the frozen GRU recipe from the vendored JSON config."""
    data = json.loads(path.read_text(encoding="utf-8"))
    params = data["params"]
    return GRURecipe(
        lookback=params["lookback"],
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dense_units=params["dense_units"],
        dropout=params["dropout"],
        lr=params["lr"],
        batch_size=params["batch_size"],
        horizon=data["horizon"],
    )
