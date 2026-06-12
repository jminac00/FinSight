"""Neural regressors (LSTM, GRU, Transformer) and a shared training loop."""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class RNNRegressor(nn.Module):
    """LSTM/GRU regressor: recurrent core, last hidden step, small dense head."""

    def __init__(
        self,
        input_size: int,
        rnn_type: str = "lstm",
        hidden_size: int = 64,
        num_layers: int = 1,
        dense_units: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn_type]
        self.rnn = rnn_cls(
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
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class TransformerRegressor(nn.Module):
    """Transformer-encoder regressor with learned positional embeddings."""

    def __init__(
        self,
        input_size: int,
        seq_len: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dense_units: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(self.proj(x) + self.pos)
        return self.head(z[:, -1, :]).squeeze(-1)


def count_params(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lr: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 100,
    patience: int = 10,
    seed: int = 42,
    device: str = "cpu",
) -> float:
    """Train with Adam + MSE and early stopping on validation loss.

    The best validation state is restored before returning. Returns the
    wall-clock training time in seconds.
    """
    torch.manual_seed(seed)
    model.to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state: dict | None = None
    bad_epochs = 0
    start = time.perf_counter()

    for _ in range(max_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return time.perf_counter() - start


def predict(model: nn.Module, X: np.ndarray, device: str = "cpu", batch_size: int = 512) -> np.ndarray:
    """Batched inference returning a 1-D numpy array."""
    model.to(device).eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)
