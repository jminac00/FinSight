import json

import torch

from app.services.deep_learning.artifacts import ModelArtifacts
from app.services.deep_learning.model import build_model
from app.services.deep_learning.preprocessing import INPUT_SIZE
from app.services.deep_learning.recipe import load_frozen_recipe
from app.services.deep_learning.training.pipeline import train_ticker


def test_save_writes_pt_and_json_named_by_ticker(make_ohlc, tmp_path):
    recipe = load_frozen_recipe()
    artifacts = train_ticker(make_ohlc(n=400, seed=2), "NVDA", recipe, seed=42, max_epochs=3)
    pt_path, json_path = artifacts.save(tmp_path)
    assert pt_path == tmp_path / "NVDA.pt"
    assert json_path == tmp_path / "NVDA.json"
    assert pt_path.exists() and json_path.exists()


def test_json_metadata_schema(make_ohlc, tmp_path):
    recipe = load_frozen_recipe()
    artifacts = train_ticker(make_ohlc(n=400, seed=4), "MSFT", recipe, seed=42, max_epochs=3)
    _, json_path = artifacts.save(tmp_path)
    meta = json.loads(json_path.read_text())
    assert set(meta) == {
        "ticker",
        "trained_at",
        "horizon_days",
        "lookback",
        "input_size",
        "recipe",
        "metrics",
        "n_samples",
        "data_through",
        "skill_ratio",
        "published",
    }
    assert set(meta["metrics"]) == {"rmse", "mae", "directional_accuracy", "rmse_naive"}
    assert set(meta["recipe"]) == {"hidden_size", "num_layers", "dense_units", "dropout"}
    assert meta["ticker"] == "MSFT"


def test_load_roundtrip_rebuilds_a_working_model(make_ohlc, tmp_path):
    recipe = load_frozen_recipe()
    artifacts = train_ticker(make_ohlc(n=400, seed=5), "AAPL", recipe, seed=42, max_epochs=3)
    artifacts.save(tmp_path)

    loaded = ModelArtifacts.load("AAPL", tmp_path)
    assert loaded.metadata.ticker == "AAPL"

    model = build_model(recipe, loaded.metadata.input_size)
    model.load_state_dict(loaded.state_dict)
    out = model(torch.zeros(2, recipe.lookback, INPUT_SIZE))
    assert out.shape == (2,)
