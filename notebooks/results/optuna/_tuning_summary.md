# Optuna tuning summary — GRU vs VMD-GRU

Hyperparameter optimization of the selected GRU architecture (issue #21), tuned on
AAPL, NVDA and PEP under the benchmark's causal pipeline. TPE sampler + Hyperband
pruner, fixed seeds, SQLite storage. Objective: mean over tickers of the
validation RMSE divided by the naive (0%-return) RMSE, averaged over 2 seeds — a
scale-free skill score where **< 1.0 beats the random walk**.

| study | trials | best objective (val RMSE / naive) |
|---|--:|--:|
| `gru` | 40 | **0.9803** |
| `vmd-gru` | 20 | 0.9830 |

The trial budgets are asymmetric because every new (lookback, alpha) pair in the
VMD study forces a full-history decomposition (~2–5 min each), so `vmd-gru` is far
costlier per trial. The VMD study selected **K = 3** by the center-frequency
aliasing criterion (min over tickers; AAPL→3, NVDA→3, PEP→4) and tuned `alpha`.

## Decision — VMD does not help, even tuned

On the optimized validation objective (comparable across studies — same tickers,
same seeds, same normalization), the tuned plain **GRU (0.9803) edges out the
tuned VMD-GRU (0.9830)**. The gap is small, but it points the wrong way for VMD,
and VMD carries real costs the plain GRU does not:

- a `decomp_len = 512` warm-up that makes short-history tickers (e.g. ARM, ~600
  sessions) nearly unusable;
- a per-sample decomposition at inference time;
- more parameters at the tuned optimum (21.8k vs 6.7k).

This confirms the benchmark's deferred finding at fixed hyperparameters: **the
VMD advantage reported in the look-ahead-biased literature disappears under a
causal pipeline.** Production uses the **plain GRU, no VMD.**

## Frozen GRU configuration (the production contract)

`gru_frozen_config.json` — consumed by the per-ticker training (`ml_models/{ticker}.json`):

| param | value |
|---|--:|
| lookback | 24 |
| hidden_size | 37 |
| num_layers | 1 |
| dense_units | 50 |
| dropout | 0.243 |
| lr | 3.44e-4 |
| batch_size | 16 |

Only the architecture/recipe is frozen; production refits the weights daily per
ticker. The tuned net is **~2.3× smaller than the benchmark default** (6.7k vs
15.5k params), which matters for daily CPU retraining on the Render free tier.

## Hyperparameter importance (gru study)

`gru_param_importance.png`: **lookback dominates (0.87)**; hidden_size (0.04),
num_layers (0.03), dense_units (0.02) and dropout (0.02) are minor; lr and
batch_size are negligible (< 0.01). The window length is the decisive knob for the
10-day return — consistent with the benchmark, where the close decomposition
carried the signal and the auxiliary set was immaterial.

## Final comparison on the untouched test split (AAPL)

Tuned model vs benchmark default vs naive, each scored on the same held-out test
samples within its study. RMSE/MAE in percentage points of the 10-day return.

**gru study** (samples aligned at lookback 60):

| model | rmse | mae | dir_acc | params |
|---|--:|--:|--:|--:|
| naive | 5.7339 | 4.5387 | — | 0 |
| gru-default | 5.6245 | 4.4080 | 0.6025 | 15 553 |
| gru-tuned | **5.6108** | 4.3916 | 0.6025 | 6 724 |

**vmd-gru study** (samples aligned at the 512-day VMD warm-up):

| model | rmse | mae | dir_acc | params |
|---|--:|--:|--:|--:|
| naive | 5.7645 | 4.5617 | — | 0 |
| vmd-gru-default | 5.6715 | 4.4418 | 0.5932 | 16 513 |
| vmd-gru-tuned | 5.6651 | 4.4486 | 0.5963 | 21 777 |

Both tuned models beat their defaults and the naive baseline, by margins
consistent with the benchmark (the 10-day price level is only marginally more
predictable than a random walk). The two tables use different sample alignments
(different VMD warm-up), so the cross-study decision rests on the validation
objective above, not on a direct RMSE comparison between these tables.
