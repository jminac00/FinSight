"""Benchmark runner: one command, one metrics table.

Usage (from the notebooks/ directory):
    uv run python -m benchmark.run --ticker AAPL
    uv run python -m benchmark.run --ticker AAPL --quick
    uv run python -m benchmark.run --ticker AAPL --models naive,arima,lstm,gru
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from benchmark import baselines, metrics
from benchmark.data import (
    FEATURES,
    WindowDataset,
    build_windows,
    chrono_split,
    engineered_features,
    load_ohlc,
    split_bounds,
)
from benchmark.models import RNNRegressor, TransformerRegressor, count_params, predict, train_model
from benchmark.vmd import causal_vmd_modes

logger = logging.getLogger(__name__)

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = NOTEBOOKS_DIR.parent

ALL_MODELS = [
    "naive",
    "momentum",
    "arima",
    "lstm",
    "gru",
    "transformer",
    "vmd-lstm",
    "vmd-gru",
    "vmd-transformer",
]
QUICK_ROWS = 1800
SEED = 42


def make_network(kind: str, n_features: int, lookback: int) -> torch.nn.Module:
    """Instantiate a network with the shared benchmark hyperparameters."""
    if kind == "transformer":
        return TransformerRegressor(n_features, seq_len=lookback)
    return RNNRegressor(n_features, rnn_type=kind)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Causal model benchmark for the DL module")
    parser.add_argument("--ticker", required=True, help="ticker symbol, e.g. AAPL")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "nasdaq_prices")
    parser.add_argument("--models", default="all", help="comma-separated subset of: " + ",".join(ALL_MODELS))
    parser.add_argument("--lookback", type=int, default=60, help="input window length in days")
    parser.add_argument("--horizon", type=int, default=10, help="prediction horizon in days")
    parser.add_argument("--threshold", type=float, default=1.5, help="trend threshold in %% at the horizon")
    parser.add_argument("--k", type=int, default=6, help="number of VMD modes")
    parser.add_argument("--alpha", type=float, default=2000.0, help="VMD bandwidth penalty")
    parser.add_argument("--decomp-len", type=int, default=512, help="causal VMD decomposition window")
    parser.add_argument(
        "--feature-set",
        choices=["ohl", "rbg"],
        default="ohl",
        help="auxiliary features for VMD models: raw open/high/low (ohl) or range/body/gap (rbg)",
    )
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--quick", action="store_true", help=f"last {QUICK_ROWS} rows, 15 epochs max")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=NOTEBOOKS_DIR / "results")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    ticker = args.ticker.upper()
    requested = ALL_MODELS if args.models == "all" else args.models.split(",")
    unknown = set(requested) - set(ALL_MODELS)
    if unknown:
        raise SystemExit(f"unknown models: {sorted(unknown)}")
    max_epochs = 15 if args.quick else args.max_epochs

    df = load_ohlc(args.data_dir / f"{ticker}.csv")
    if args.quick:
        df = df.tail(QUICK_ROWS)
    logger.info("%s: %d clean rows [%s -> %s]", ticker, len(df), df.index[0].date(), df.index[-1].date())

    features = df[FEATURES].to_numpy(dtype=np.float64)
    close = features[:, FEATURES.index("close")]
    t_start = max(args.lookback, args.decomp_len) - 1  # same samples for every model

    raw_ds = build_windows(features, close, df.index, args.lookback, args.horizon, t_start)
    datasets = {"raw": raw_ds}
    if any(name.startswith("vmd-") for name in requested):
        modes = causal_vmd_modes(
            close,
            lookback=args.lookback,
            horizon=args.horizon,
            decomp_len=args.decomp_len,
            k=args.k,
            alpha=args.alpha,
            t_start=t_start,
            cache_dir=NOTEBOOKS_DIR / ".cache",
            ticker=ticker,
        )
        if args.feature_set == "rbg":
            aux_matrix = engineered_features(features)
        else:
            aux_matrix = features[:, :3]  # raw open, high, low
        aux = build_windows(aux_matrix, close, df.index, args.lookback, args.horizon, t_start)
        datasets["vmd"] = WindowDataset(
            X=np.concatenate([modes, aux.X], axis=2), y=raw_ds.y, t_index=raw_ds.t_index
        )

    n = len(raw_ds.y)
    gap = args.horizon
    _, _, te = split_bounds(n, gap=gap)
    logger.info("%d samples (train/val/test split with %d-sample boundary gaps)", n, gap)

    rows: list[dict] = []
    for name in requested:
        logger.info("=== %s ===", name)
        start = time.perf_counter()
        if name == "naive":
            test = chrono_split(raw_ds, gap=gap)[2]
            preds, params = baselines.naive_zero(len(test.y)), 0
        elif name == "momentum":
            test = chrono_split(raw_ds, gap=gap)[2]
            preds, params = baselines.momentum(raw_ds.y, te.start, n, args.horizon), 0
        elif name == "arima":
            test = chrono_split(raw_ds, gap=gap)[2]
            sample_ts = t_start + np.arange(te.start, n)
            preds, params = baselines.arima(np.log(close), sample_ts, args.horizon), 3
        else:
            kind = name.removeprefix("vmd-")
            ds = datasets["vmd" if name.startswith("vmd-") else "raw"]
            train, val, test = chrono_split(ds, gap=gap)
            model = make_network(kind, ds.X.shape[2], args.lookback)
            train_model(
                model, train.X, train.y, val.X, val.y,
                max_epochs=max_epochs, seed=SEED, device=args.device,
            )
            preds, params = predict(model, test.X, device=args.device), count_params(model)

        row = {"model": name, **metrics.evaluate(test.y, preds, args.threshold)}
        row["params"] = params
        row["time_s"] = round(time.perf_counter() - start, 1)
        rows.append(row)
        logger.info("%s: rmse=%.3f dir_acc=%s", name, row["rmse"], f"{row['dir_acc']:.3f}" if row["dir_acc"] == row["dir_acc"] else "n/a")

    table = pd.DataFrame(rows).round(4)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    quick_suffix = "_quick" if args.quick else ""
    out_file = args.out_dir / f"{ticker}_benchmark_{args.feature_set}{quick_suffix}.csv"
    table.to_csv(out_file, index=False)

    print()
    print(
        f"# {ticker} — {args.horizon}-day return benchmark "
        f"(threshold ±{args.threshold}%, feature set: {args.feature_set})"
    )
    print(table.to_markdown(index=False))
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
