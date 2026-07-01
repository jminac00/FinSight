# ADR-0006 — GRU without VMD as the deep learning production model

- **Status**: Accepted
- **Date**: June 2026
- **Source**: Internal architecture design (research issues #19–#21)

## Context

Module 2 forecasts a stock's short-term price trend. The research phase
(notebooks, issues #19–#21) evaluated candidate architectures under a strict,
leakage-free protocol and had to settle two questions before any production code
was written: **which network**, and **whether to keep Variational Mode
Decomposition (VMD)** as a preprocessing step.

The literature reports large gains from VMD-based pipelines (e.g. VMD-SSA-LSTM),
but most of those results decompose the *entire* series once, leaking future
information into past samples. The benchmark (`notebooks/benchmark/`) instead:

- decomposes causally (each sample's modes come only from closes ending at its
  prediction day — right-edge decomposition);
- predicts a **direct, dimensionless 10-day cumulative return** (no inverse
  scaling);
- splits chronologically (70/15/15) with horizon-sized gaps so no target window
  spans two partitions;
- scores against a zero-return naive (random walk) baseline.

An Optuna study (TPE + Hyperband, fixed seeds, two seeds averaged per trial)
then tuned the GRU under that same pipeline, with the objective being the
validation RMSE divided by the naive RMSE — a scale-free skill score where
**< 1.0 beats the random walk**. The full results are in
`notebooks/results/optuna/_tuning_summary.md`.

A rule was agreed **before** seeing results: if LSTM and GRU tie within noise,
GRU wins, because production retrains daily per ticker on CPU (Render free tier)
and the GRU is cheaper to train. LSTM would be chosen only on a clear, consistent
win.

## Decision

Production uses a **plain GRU, no VMD**.

1. **GRU over LSTM** — the two tie within noise on the causal benchmark, so the
   pre-agreed tie-break (lower training cost) selects the GRU.

2. **No VMD** — on the optimized, cross-comparable validation objective the tuned
   plain GRU (0.9803) edged out the tuned VMD-GRU (0.9830). The gap is small but
   points against VMD, which also carries costs the plain GRU does not: a
   `decomp_len = 512` warm-up that makes short-history tickers nearly unusable, a
   per-sample decomposition at inference time, and more parameters (21.8k vs
   6.7k). This confirms the benchmark's finding that **the VMD advantage reported
   in the look-ahead-biased literature disappears under a causal pipeline.**

3. **Frozen recipe, refit daily** — only the architecture/recipe is frozen
   (`notebooks/results/optuna/gru_frozen_config.json`): lookback 24, hidden size
   37, 1 layer, dense units 50, dropout 0.243, lr 3.44e-4, batch size 16. The
   tuned net is ~2.3× smaller than the benchmark default (6.7k vs 15.5k params).
   Production refits the weights per ticker; the recipe is the contract the
   per-ticker training and `ml_models/{ticker}.json` metadata consume.

4. **Return-based output contract** — the model predicts a 10-day cumulative
   return (%), so `DLResult` exposes `predicted_return_pct` (with
   `predicted_price` derived from the current close) and the metrics are RMSE/MAE
   on the return plus directional accuracy — not the price-level `mape`/`r2` of
   the original stub. The user-facing `trend` is the benchmark's 3-class label
   (alcista/bajista/neutral) using a symmetric **neutral band of 1.5 %** at the
   horizon, matching the threshold the benchmark scored trend accuracy with.

## Consequences

- The production runtime (per-ticker training script, `DLService` inference,
  daily APScheduler retraining) is built against a GRU and the return-based
  contract from the start; no VMD code path is needed in the backend.
- Daily CPU retraining is cheap: a 6.7k-parameter GRU on a single lookback-24
  window trains in seconds per ticker, fitting the Render free tier.
- Short-history tickers (no 512-day VMD warm-up) become usable, widening
  Nasdaq coverage.
- Honesty about predictability: the 10-day return is only marginally more
  predictable than a random walk (skill score ~0.98), so the neutral band and
  the prominent disclaimer (RF-06) matter — the report must not overstate
  confidence.
- The VMD tooling remains in the research notebooks (`--model vmd-gru`) for
  reproducibility and the written record, but is not promoted to production.
- The price data source and Nasdaq coverage scope are documented separately in
  [ADR-0008](0008-dl-data-source-and-nasdaq-scope.md).
