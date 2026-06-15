"""Optuna hyperparameter optimization for the selected GRU architecture.

Tunes the GRU (and optionally the VMD-GRU) under the exact causal pipeline of
the benchmark (`build_windows` + `causal_vmd_modes` + `chrono_split`), so the
tuned configuration transfers unchanged to the per-ticker production training.

Usage (from the notebooks/ directory, with the venv Python to keep CUDA torch):
    .\\.venv\\Scripts\\python.exe -m benchmark.optimize --model gru --trials 40
    .\\.venv\\Scripts\\python.exe -m benchmark.optimize --model vmd-gru --trials 40
    .\\.venv\\Scripts\\python.exe -m benchmark.optimize --model gru --trials 5 --quick
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from vmdpy import VMD

from benchmark import baselines, metrics
from benchmark.data import (
    FEATURES,
    WindowDataset,
    build_windows,
    chrono_split,
    engineered_features,
    load_ohlc,
)
from benchmark.models import RNNRegressor, count_params, predict, train_model
from benchmark.vmd import VMD_DC, VMD_INIT, VMD_TAU, VMD_TOL, causal_vmd_modes

logger = logging.getLogger(__name__)

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = NOTEBOOKS_DIR.parent

# Reproducibility: fixed sampler seed and fixed per-trial training seeds.
SAMPLER_SEED = 42
SEEDS = (42, 43)  # averaged per trial to absorb run-to-run noise

DEFAULT_TICKERS = ["AAPL", "NVDA", "PEP"]
QUICK_ROWS = 1800

# VMD K-selection (center-frequency aliasing criterion).
K_MIN = 3
K_MAX = 10
MIN_FREQ_SEP = 0.01  # normalized-frequency gap below which two modes alias
ALPHA_CHOICES = [1000.0, 2000.0, 4000.0]

# Untuned benchmark defaults, used as the comparison baseline.
DEFAULT_CONFIG = {
    "lr": 1e-3,
    "lookback": 60,
    "hidden_size": 64,
    "num_layers": 1,
    "dense_units": 32,
    "dropout": 0.2,
    "batch_size": 64,
    "alpha": 2000.0,
}
DEFAULT_K = 6


@dataclass
class TickerData:
    """OHLC arrays for one ticker, loaded once and reused across trials."""

    ticker: str
    features: np.ndarray
    close: np.ndarray
    dates: pd.DatetimeIndex


def load_ticker_data(ticker: str, data_dir: Path, quick: bool = False) -> TickerData:
    """Load and clean one ticker into reusable numpy arrays."""
    df = load_ohlc(data_dir / f"{ticker}.csv")
    if quick:
        df = df.tail(QUICK_ROWS)
    features = df[FEATURES].to_numpy(dtype=np.float64)
    close = features[:, FEATURES.index("close")]
    logger.info("%s: %d clean rows [%s -> %s]", ticker, len(df), df.index[0].date(), df.index[-1].date())
    return TickerData(ticker=ticker, features=features, close=close, dates=df.index)


def suggest_config(trial: optuna.Trial, model: str = "gru") -> dict:
    """Sample one hyperparameter configuration from the search space.

    `epochs` is deliberately absent: every trial relies on early stopping.
    """
    cfg = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "lookback": trial.suggest_int("lookback", 20, 90),
        "hidden_size": trial.suggest_int("hidden_size", 16, 128),
        "num_layers": trial.suggest_int("num_layers", 1, 2),
        "dense_units": trial.suggest_int("dense_units", 8, 64),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
    }
    if model == "vmd-gru":
        cfg["alpha"] = trial.suggest_categorical("alpha", ALPHA_CHOICES)
    return cfg


def build_gru(input_size: int, cfg: dict) -> RNNRegressor:
    """Instantiate a GRU regressor from a sampled configuration."""
    return RNNRegressor(
        input_size,
        rnn_type="gru",
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dense_units=cfg["dense_units"],
        dropout=cfg["dropout"],
    )


def select_k_by_aliasing(
    segment: np.ndarray,
    *,
    alpha: float = 2000.0,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
    min_freq_sep: float = MIN_FREQ_SEP,
) -> int:
    """Largest K whose VMD modes keep distinct center frequencies.

    Decomposes `segment` for increasing K and stops at the first K where two
    converged center frequencies (last row of VMD's `omega` output) fall within
    `min_freq_sep` of each other — the onset of mode aliasing. Returns the K
    just below that onset.
    """
    segment = np.asarray(segment, dtype=np.float64)
    best_k = k_min
    for k in range(k_min, k_max + 1):
        _, _, omega = VMD(segment, alpha, VMD_TAU, k, VMD_DC, VMD_INIT, VMD_TOL)
        center_freqs = np.sort(omega[-1])
        if np.min(np.diff(center_freqs)) < min_freq_sep:
            break
        best_k = k
    return best_k


def _pruning_callback(trial: optuna.Trial):
    """Epoch callback that reports validation loss to Optuna and prunes."""

    def callback(epoch: int, val_loss: float) -> None:
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return callback


def evaluate_config(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
    *,
    seeds: tuple[int, ...] = SEEDS,
    max_epochs: int = 60,
    patience: int = 10,
    device: str = "cpu",
    trial: optuna.Trial | None = None,
) -> float:
    """Mean validation RMSE of a GRU over several seeds.

    Each seed trains a fresh GRU with early stopping on the validation split.
    When `trial` is given, the first seed reports its per-epoch validation loss
    to Optuna so bad configurations are pruned early.
    """
    input_size = x_tr.shape[2]
    rmses: list[float] = []
    for i, seed in enumerate(seeds):
        torch.manual_seed(seed)  # seed weight init too, not only the training loop
        model = build_gru(input_size, cfg)
        callback = _pruning_callback(trial) if trial is not None and i == 0 else None
        train_model(
            model, x_tr, y_tr, x_val, y_val,
            lr=cfg["lr"], batch_size=cfg["batch_size"],
            max_epochs=max_epochs, patience=patience, seed=seed,
            device=device, epoch_callback=callback,
        )
        preds = predict(model, x_val, device=device)
        rmses.append(float(np.sqrt(np.mean((y_val - preds) ** 2))))
    return float(np.mean(rmses))


def _build_dataset(
    td: TickerData,
    model: str,
    cfg: dict,
    k: int,
    *,
    horizon: int,
    decomp_len: int,
    feature_set: str,
    cache_dir: Path,
    t_start: int,
) -> WindowDataset:
    """Build one ticker's samples for `model`, mirroring the benchmark runner."""
    lookback = cfg["lookback"]
    raw = build_windows(td.features, td.close, td.dates, lookback, horizon, t_start)
    if model == "gru":
        return raw
    modes = causal_vmd_modes(
        td.close, lookback=lookback, horizon=horizon, decomp_len=decomp_len,
        k=k, alpha=cfg["alpha"], t_start=t_start, cache_dir=cache_dir, ticker=td.ticker,
    )
    aux_matrix = engineered_features(td.features) if feature_set == "rbg" else td.features[:, :3]
    aux = build_windows(aux_matrix, td.close, td.dates, lookback, horizon, t_start)
    return WindowDataset(X=np.concatenate([modes, aux.X], axis=2), y=raw.y, t_index=raw.t_index)


def _natural_t_start(model: str, lookback: int, decomp_len: int) -> int:
    """First last-known day for a model: VMD needs the longer decomposition warm-up."""
    return (max(lookback, decomp_len) if model == "vmd-gru" else lookback) - 1


def objective(
    trial: optuna.Trial,
    tickers_data: list[TickerData],
    model: str,
    k: int,
    *,
    horizon: int,
    decomp_len: int,
    feature_set: str,
    seeds: tuple[int, ...],
    max_epochs: int,
    patience: int,
    device: str,
    cache_dir: Path,
) -> float:
    """Mean (over tickers) of the GRU validation RMSE relative to the naive baseline.

    Normalizing by the per-ticker naive RMSE (predicting a 0% return) puts every
    ticker on the same scale, so high-volatility names do not dominate the mean.
    A value below 1.0 means the model beats the random walk on validation.
    """
    cfg = suggest_config(trial, model)
    ratios: list[float] = []
    for i, td in enumerate(tickers_data):
        t_start = _natural_t_start(model, cfg["lookback"], decomp_len)
        ds = _build_dataset(
            td, model, cfg, k, horizon=horizon, decomp_len=decomp_len,
            feature_set=feature_set, cache_dir=cache_dir, t_start=t_start,
        )
        train, val, _ = chrono_split(ds, gap=horizon)
        naive_rmse = float(np.sqrt(np.mean(val.y**2)))
        rmse = evaluate_config(
            train.X, train.y, val.X, val.y, cfg,
            seeds=seeds, max_epochs=max_epochs, patience=patience, device=device,
            trial=trial if i == 0 else None,
        )
        ratios.append(rmse / naive_rmse)
    return float(np.mean(ratios))


def write_frozen_config(path: Path, config: dict) -> None:
    """Persist the winning configuration as JSON (the production contract)."""
    Path(path).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def read_frozen_config(path: Path) -> dict:
    """Load a frozen configuration JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_frozen_config(
    study: optuna.Study, model: str, k: int, *, horizon: int, decomp_len: int,
    feature_set: str, tickers: list[str],
) -> dict:
    """Assemble the frozen architecture contract from the best trial."""
    p = study.best_trial.params
    config = {
        "architecture": "gru",
        "use_vmd": model == "vmd-gru",
        "horizon": horizon,
        "params": {
            "lr": p["lr"],
            "lookback": p["lookback"],
            "hidden_size": p["hidden_size"],
            "num_layers": p["num_layers"],
            "dense_units": p["dense_units"],
            "dropout": p["dropout"],
            "batch_size": p["batch_size"],
        },
        "objective": "mean over tickers of (val RMSE / naive val RMSE)",
        "best_objective": study.best_value,
        "tuned_on": tickers,
        "sampler_seed": SAMPLER_SEED,
        "training_seeds": list(SEEDS),
        "study_name": study.study_name,
    }
    if model == "vmd-gru":
        config["vmd"] = {"k": k, "alpha": p["alpha"], "decomp_len": decomp_len, "feature_set": feature_set}
    return config


def export_importance_plot(study: optuna.Study, path: Path) -> None:
    """Export the hyperparameter importance bar chart for the project report."""
    import matplotlib

    matplotlib.use("Agg")
    from optuna.visualization.matplotlib import plot_param_importances

    try:
        ax = plot_param_importances(study)
    except (ValueError, RuntimeError) as exc:
        logger.warning("Could not compute hyperparameter importances: %s", exc)
        return
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=150)
    logger.info("Importance plot written: %s", path)


def _train_eval(
    td: TickerData, model: str, cfg: dict, k: int, *, horizon: int, decomp_len: int,
    feature_set: str, cache_dir: Path, t_start: int, max_epochs: int, patience: int,
    device: str, threshold: float, seed: int,
) -> dict:
    """Train one GRU on a shared sample alignment and score it on the test split."""
    ds = _build_dataset(
        td, model, cfg, k, horizon=horizon, decomp_len=decomp_len,
        feature_set=feature_set, cache_dir=cache_dir, t_start=t_start,
    )
    train, val, test = chrono_split(ds, gap=horizon)
    torch.manual_seed(seed)  # seed weight init too, not only the training loop
    net = build_gru(ds.X.shape[2], cfg)
    train_model(
        net, train.X, train.y, val.X, val.y, lr=cfg["lr"], batch_size=cfg["batch_size"],
        max_epochs=max_epochs, patience=patience, seed=seed, device=device,
    )
    preds = predict(net, test.X, device=device)
    return {**metrics.evaluate(test.y, preds, threshold), "params": count_params(net)}


def final_comparison(
    td: TickerData, model: str, frozen_cfg: dict, k: int, *, horizon: int, decomp_len: int,
    feature_set: str, cache_dir: Path, max_epochs: int, patience: int, device: str,
    threshold: float, seed: int = SEEDS[0],
) -> pd.DataFrame:
    """Naive vs untuned-default vs tuned on one ticker's untouched test split.

    All three predictors are scored on the same test samples: every model uses a
    common first last-known day so the prediction days and targets are identical,
    only the input windows differ.
    """
    tuned_cfg = {**DEFAULT_CONFIG, **frozen_cfg["params"]}
    if model == "vmd-gru":
        tuned_cfg["alpha"] = frozen_cfg["vmd"]["alpha"]
    lookbacks = [DEFAULT_CONFIG["lookback"], tuned_cfg["lookback"]]
    warmup = decomp_len if model == "vmd-gru" else 0
    t_start = max(*lookbacks, warmup) - 1

    # Naive baseline on the shared test split.
    raw = build_windows(td.features, td.close, td.dates, max(lookbacks), horizon, t_start)
    test_y = chrono_split(raw, gap=horizon)[2].y
    rows = [{"model": "naive", **metrics.evaluate(test_y, baselines.naive_zero(len(test_y)), threshold), "params": 0}]

    common = dict(
        td=td, model=model, horizon=horizon, decomp_len=decomp_len, feature_set=feature_set,
        cache_dir=cache_dir, t_start=t_start, max_epochs=max_epochs, patience=patience,
        device=device, threshold=threshold, seed=seed,
    )
    rows.append({"model": f"{model}-default", **_train_eval(cfg=DEFAULT_CONFIG, k=DEFAULT_K, **common)})
    rows.append({"model": f"{model}-tuned", **_train_eval(cfg=tuned_cfg, k=k, **common)})
    return pd.DataFrame(rows).round(4)


def run_study(
    tickers_data: list[TickerData], model: str, k: int, *, n_trials: int, storage: str,
    horizon: int, decomp_len: int, feature_set: str, seeds: tuple[int, ...], max_epochs: int,
    patience: int, device: str, cache_dir: Path,
) -> optuna.Study:
    """Create (or resume) the SQLite-backed study and run `n_trials` trials."""
    sampler = optuna.samplers.TPESampler(seed=SAMPLER_SEED)
    pruner = optuna.pruners.HyperbandPruner(min_resource=5, max_resource=max_epochs, reduction_factor=3)
    study = optuna.create_study(
        study_name=f"{model}_h{horizon}", direction="minimize", sampler=sampler,
        pruner=pruner, storage=storage, load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(
            trial, tickers_data, model, k, horizon=horizon, decomp_len=decomp_len,
            feature_set=feature_set, seeds=seeds, max_epochs=max_epochs,
            patience=patience, device=device, cache_dir=cache_dir,
        ),
        n_trials=n_trials,
    )
    return study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization for the GRU architecture")
    parser.add_argument("--model", choices=["gru", "vmd-gru"], default="gru")
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS), help="comma-separated tuning tickers")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--decomp-len", type=int, default=512)
    parser.add_argument("--feature-set", choices=["ohl", "rbg"], default="ohl")
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=1.5)
    parser.add_argument("--quick", action="store_true", help=f"last {QUICK_ROWS} rows, 15 epochs max")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "nasdaq_prices")
    parser.add_argument("--out-dir", type=Path, default=NOTEBOOKS_DIR / "results" / "optuna")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    max_epochs = 15 if args.quick else args.max_epochs
    cache_dir = NOTEBOOKS_DIR / ".cache"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tickers_data = [load_ticker_data(t, args.data_dir, quick=args.quick) for t in tickers]

    k = DEFAULT_K
    if args.model == "vmd-gru":
        per_ticker = {
            td.ticker: select_k_by_aliasing(td.close[-args.decomp_len:], alpha=DEFAULT_CONFIG["alpha"])
            for td in tickers_data
        }
        k = min(per_ticker.values())  # conservative: no ticker aliases
        logger.info("VMD K by aliasing criterion: %s -> using K=%d", per_ticker, k)

    storage = f"sqlite:///{(args.out_dir / f'{args.model}_study.db').as_posix()}"
    study = run_study(
        tickers_data, args.model, k, n_trials=args.trials, storage=storage,
        horizon=args.horizon, decomp_len=args.decomp_len, feature_set=args.feature_set,
        seeds=SEEDS, max_epochs=max_epochs, patience=args.patience, device=args.device,
        cache_dir=cache_dir,
    )
    logger.info("Best objective (val RMSE / naive): %.4f", study.best_value)

    frozen = build_frozen_config(
        study, args.model, k, horizon=args.horizon, decomp_len=args.decomp_len,
        feature_set=args.feature_set, tickers=tickers,
    )
    write_frozen_config(args.out_dir / f"{args.model}_frozen_config.json", frozen)
    export_importance_plot(study, args.out_dir / f"{args.model}_param_importance.png")

    comparison = final_comparison(
        tickers_data[0], args.model, frozen, k, horizon=args.horizon, decomp_len=args.decomp_len,
        feature_set=args.feature_set, cache_dir=cache_dir, max_epochs=max_epochs,
        patience=args.patience, device=args.device, threshold=args.threshold,
    )
    comparison.to_csv(args.out_dir / f"{args.model}_comparison.csv", index=False)

    print()
    print(f"# {args.model} tuning — best val RMSE/naive = {study.best_value:.4f}")
    print(json.dumps(frozen["params"], indent=2))
    print(f"\n# Final comparison on {tickers_data[0].ticker} (untouched test split)")
    print(comparison.to_markdown(index=False))


if __name__ == "__main__":
    main()
