"""The GRU regressor and its factory — shared by training (to fit) and inference
(to rebuild the architecture before loading a state dict).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from app.services.deep_learning.recipe import GRURecipe


class GRURegressor(nn.Module):
    """GRU core, last hidden step, small dense head -> scalar return prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dense_units: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def build_model(recipe: GRURecipe, input_size: int) -> GRURegressor:
    """Instantiate the GRU regressor from the frozen recipe."""
    return GRURegressor(
        input_size=input_size,
        hidden_size=recipe.hidden_size,
        num_layers=recipe.num_layers,
        dense_units=recipe.dense_units,
        dropout=recipe.dropout,
    )


def predict(
    model: nn.Module, x: np.ndarray, device: str = "cpu", batch_size: int = 512
) -> np.ndarray:
    """Batched inference returning a 1-D numpy array."""
    model.to(device).eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[i : i + batch_size]).to(device)
            preds.append(np.atleast_1d(model(xb).cpu().numpy()))
    return np.concatenate(preds) if preds else np.empty(0, dtype=np.float32)
