"""Persistence of trained models — the single place that knows the on-disk
artifact format. Training writes it and inference reads it, so the schema is
defined once here and both sides agree by construction.

Per ticker: ``{ticker}.pt`` (the model state dict) and ``{ticker}.json`` (self
describing metadata: the recipe used, quality metrics and provenance).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[3] / "ml_models"


@dataclass
class ModelMetadata:
    """Self-describing metadata stored alongside a trained model."""

    ticker: str
    trained_at: str  # ISO-8601 UTC
    horizon_days: int
    lookback: int
    input_size: int
    recipe: dict  # hidden_size, num_layers, dense_units, dropout
    metrics: dict  # rmse, mae, directional_accuracy
    n_samples: int
    data_through: str  # ISO-8601 date of the last training sample


@dataclass
class ModelArtifacts:
    """A trained model plus its metadata, ready to persist or load."""

    ticker: str
    state_dict: dict
    metadata: ModelMetadata

    def save(self, models_dir: Path = DEFAULT_MODELS_DIR) -> tuple[Path, Path]:
        """Write ``{ticker}.pt`` and ``{ticker}.json``; return both paths."""
        models_dir.mkdir(parents=True, exist_ok=True)
        pt_path = models_dir / f"{self.ticker}.pt"
        json_path = models_dir / f"{self.ticker}.json"
        torch.save(self.state_dict, pt_path)
        json_path.write_text(json.dumps(asdict(self.metadata), indent=2), encoding="utf-8")
        return pt_path, json_path

    @classmethod
    def load(cls, ticker: str, models_dir: Path = DEFAULT_MODELS_DIR) -> ModelArtifacts:
        """Load the artifacts for a ticker from ``models_dir``."""
        state_dict = torch.load(models_dir / f"{ticker}.pt", weights_only=True)
        meta = json.loads((models_dir / f"{ticker}.json").read_text(encoding="utf-8"))
        return cls(ticker=ticker, state_dict=state_dict, metadata=ModelMetadata(**meta))
