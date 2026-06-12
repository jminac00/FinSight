# FinSight — Deep Learning Research

Offline research environment for the deep learning module (Module 2). Standalone
uv project, independent from the backend environment.

## Contents

| Path | Purpose |
|---|---|
| `vmd-ssa-lstm.ipynb` | Baseline research notebook (starting point, see #19) |
| `benchmark/` | Causal model benchmark (see #20) |
| `tests/` | Tests for the benchmark's causality and target alignment guarantees |
| `results/` | Benchmark output tables (CSV) |
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
  networks on VMD features (modes of close + OHL).
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
```

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

## Architecture decision rule

Agreed before looking at results: if LSTM and GRU tie within noise, **GRU wins**
(lower training cost matters — production retrains daily per ticker on CPU,
Render free tier). LSTM is chosen only if it wins clearly and consistently
across tickers.
