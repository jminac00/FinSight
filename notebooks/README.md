# FinSight — Deep Learning Research

Offline research environment for the deep learning module (Module 2). Standalone
uv project, independent from the backend environment.

## Contents

| Path | Purpose |
|---|---|
| `vmd-ssa-lstm.ipynb` | Baseline research notebook (starting point, see #19) |
| `benchmark/` | Causal model benchmark (see #20) |
| `benchmark/optimize.py` | Optuna hyperparameter study for the GRU architecture (see #21) |
| `tests/` | Tests for the benchmark's causality and target alignment guarantees |
| `results/` | Benchmark output tables (CSV) |
| `results/optuna/` | Tuning artifacts: study DB, frozen config, importance plot, comparison |
| `.cache/` | VMD decomposition cache (gitignored) |

## Benchmark

Trains and evaluates every candidate model on identical, leakage-free samples:

- **Causal VMD**: each sample's modes come from a decomposition of the
  `decomp_len` closes ending at its prediction day t — only past data, matching
  inference-time conditions (right-edge decomposition).
- **Direct 10-day target**: cumulative percentage return between t and t+10.
  Dimensionless, so no inverse scaling is needed.
- **Chronological split** (70/15/15) with `horizon`-sample gaps at the
  boundaries so no target window spans two partitions.
- **Models**: zero-return naive (random walk), momentum, ARIMA(1,1,1) with
  drift (walk-forward), LSTM, GRU, Transformer encoder, and the same three
  networks on VMD features (modes of close + auxiliary features).
- **Auxiliary feature set** for the VMD models, selected with `--feature-set`:
  - `ohl` (default): raw open/high/low levels.
  - `rbg`: engineered intraday features — range `(high-low)/close`, body
    `(close-open)/close`, gap `(open-prev_close)/prev_close`. These capture
    the information OHL carry beyond the close, which the raw levels barely do
    (open, high, low and close are ~99% correlated day to day).
- **Multi-branch VMD models** (`vmd-lstm-mb`, `vmd-gru-mb`): instead of feeding
  all modes as one multivariate input, each VMD mode gets its own recurrent
  branch (the auxiliary features share one extra branch); the branches' last
  hidden states are concatenated and a shared head predicts the 10-day return.
  This is the leakage-free, direct-target version of the classic "one model per
  mode" scheme — trained end-to-end toward the return, not by reconstructing and
  summing modes.
- **Metrics**: RMSE/MAE on the 10-day return (%), directional accuracy,
  3-class trend accuracy and macro F1, parameter count, wall time.

### Setup

```bash
cd notebooks
uv sync
```

PyTorch installs as the CPU build by default. For CUDA (recommended for full
runs):

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### Usage

```bash
# Full benchmark for one ticker (CSV expected in ../data/nasdaq_prices/)
uv run python -m benchmark.run --ticker AAPL

# Fast end-to-end verification (last 1800 rows, 15 epochs max)
uv run python -m benchmark.run --ticker AAPL --quick

# Subset of models
uv run python -m benchmark.run --ticker AAPL --models naive,arima,lstm,gru

# Compare the single-input VMD model against its multi-branch variant
uv run python -m benchmark.run --ticker AAPL --models vmd-gru,vmd-gru-mb

# Compare auxiliary feature sets for the VMD models on the same ticker
uv run python -m benchmark.run --ticker AAPL --feature-set ohl
uv run python -m benchmark.run --ticker AAPL --feature-set rbg
```

Output is written to `results/{TICKER}_benchmark_{feature_set}.csv`, so the two
feature sets coexist for comparison. The VMD cache does not depend on the
feature set, so the second run reuses it and only retrains the networks.

First full run per ticker computes one VMD decomposition per sample
(~30 min for a long history); results are cached in `.cache/` and reruns are
immediate.

### Tests

```bash
uv run pytest
```

The tests pin down the two guarantees everything else depends on: features at
day t never change when data after t changes (causality), and targets align
with the intended t -> t+horizon return.

## Hyperparameter optimization (Optuna)

`benchmark/optimize.py` tunes the selected GRU architecture under the same causal
pipeline as the benchmark, then freezes one configuration as the contract the
per-ticker production training consumes.

- **Sampler / pruner**: TPE sampler + Hyperband pruner — each trial reports its
  per-epoch validation loss so weak configurations stop early.
- **Search space**: learning rate (log `[1e-4, 1e-2]`), lookback `[20, 90]`,
  hidden size `[16, 128]`, layers `[1, 2]`, dense units `[8, 64]`, dropout
  `[0, 0.3]`, batch size `{16, 32, 64}`. `epochs` is not searched — every trial
  relies on early stopping.
- **Objective**: mean over the tuning tickers of the validation RMSE divided by
  the naive (0%-return) RMSE. The ratio normalizes the per-ticker scale so a
  high-volatility name (NVDA) does not dominate a stable one (PEP); below 1.0
  means the model beats the random walk on validation. Each trial averages two
  seeds to absorb run-to-run noise.
- **VMD variant** (`--model vmd-gru`): K is chosen by the center-frequency
  aliasing criterion — the largest K whose VMD modes keep distinct center
  frequencies, read from VMD's `omega` output — and `alpha` joins the search
  space. Decompositions are cached in `.cache/`.
- **Reproducible & resumable**: fixed sampler and training seeds plus SQLite
  storage; rerunning the same command resumes the study.

### Usage

Run with the venv Python directly so the manually installed CUDA build of torch
is kept (`uv run` would re-sync to the CPU build):

```bash
# Tune the GRU on the representative tickers (writes to results/optuna/)
.venv/Scripts/python.exe -m benchmark.optimize --model gru --trials 40

# Quick end-to-end check (1800 rows, 15 epochs max, few trials)
.venv/Scripts/python.exe -m benchmark.optimize --model gru --trials 5 --quick
```

Each run writes, under `results/optuna/`: the SQLite study (`{model}_study.db`),
the frozen configuration (`{model}_frozen_config.json`), the hyperparameter
importance plot (`{model}_param_importance.png`) and the comparison table
(`{model}_comparison.csv` — tuned vs benchmark default vs naive on the untouched
test split).

> The `vmd-gru` study is an offline activity: every new (lookback, alpha) pair
> triggers a full-history VMD decomposition (~30 min each), so a full sweep is
> impractical interactively. The tooling is identical (`--model vmd-gru`).

## Architecture decision rule

Agreed before looking at results: if LSTM and GRU tie within noise, **GRU wins**
(lower training cost matters — production retrains daily per ticker on CPU,
Render free tier). LSTM is chosen only if it wins clearly and consistently
across tickers.
