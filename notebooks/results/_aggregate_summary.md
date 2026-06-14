# Benchmark aggregate — 8 tickers (AAPL, NVDA, TSLA, MSFT, ABNB, DDOG, PEP, INTC)

10-day return forecasting, causal pipeline, fixed (untuned) hyperparameters,
trend threshold ±1.5%. RMSE/MAE in percentage points of the 10-day return.

## GPU non-determinism noise floor

The raw `lstm`/`gru`/`transformer` models do not depend on the feature set, so
their RMSE difference between the `ohl` and `rbg` runs is pure run-to-run noise:

| model | mean \|ΔRMSE\| | max \|ΔRMSE\| | relative |
|---|--:|--:|--:|
| lstm | 0.0869 | 0.3728 | 1.08% |
| gru | 0.0000 | 0.0000 | 0.00% |
| transformer | 0.0000 | 0.0000 | 0.00% |

cuDNN LSTM is non-deterministic on GPU; GRU and the Transformer are
deterministic. **Any RMSE gap below ~0.09 is noise**, and only LSTM carries it.

## RMSE per ticker (feature set = ohl)

| ticker | naive | lstm | gru | transformer | vmd-lstm | vmd-gru | vmd-transformer |
|---|--:|--:|--:|--:|--:|--:|--:|
| AAPL | 5.764 | 5.644 | 5.675 | 5.682 | 5.653 | 5.672 | 5.666 |
| NVDA | 13.714 | 13.622 | 13.572 | 13.592 | 13.645 | 13.630 | 13.622 |
| TSLA | 11.979 | 11.831 | 12.233 | 11.936 | 11.884 | 11.890 | 11.815 |
| MSFT | 4.829 | 4.828 | 4.834 | 4.841 | 4.848 | 4.842 | 4.871 |
| ABNB | 6.339 | 5.747 | 5.964 | 6.070 | 5.868 | 5.930 | 5.987 |
| DDOG | 10.173 | 10.173 | 10.178 | 10.273 | 10.171 | 10.166 | 10.192 |
| PEP | 3.711 | 3.686 | 3.682 | 3.681 | 3.671 | 3.691 | 3.683 |
| INTC | 8.733 | 8.840 | 8.831 | 8.735 | 8.719 | 8.743 | 8.738 |
| **mean** | **8.155** | **8.046** | **8.121** | **8.101** | **8.057** | **8.071** | **8.072** |

## Decision 1 — VMD vs no-VMD (mean RMSE, ohl)

| pair | no-VMD | VMD | Δ (VMD−base) | VMD better in |
|---|--:|--:|--:|--:|
| lstm / vmd-lstm | 8.046 | 8.057 | +0.011 | 3/8 |
| gru / vmd-gru | 8.121 | 8.071 | −0.050 | 5/8 |
| transformer / vmd-transformer | 8.101 | 8.072 | −0.030 | 4/8 |

All |Δ| are below the LSTM noise floor (0.087). **No significant VMD effect at
fixed hyperparameters.** Once the look-ahead bias of the baseline notebook is
removed, the VMD advantage reported in the literature largely disappears.

## Decision 2 — LSTM vs GRU (mean RMSE, ohl)

| variant | LSTM | GRU | Δ (GRU−LSTM) | LSTM better in |
|---|--:|--:|--:|--:|
| raw | 8.046 | 8.121 | +0.075 | 5/8 |
| vmd | 8.057 | 8.071 | +0.013 | 5/8 |

LSTM's edge (0.075 raw, 0.013 vmd) is **smaller than LSTM's own run-to-run noise
(0.087)** → tie within noise. By the agreed rule, **GRU wins**: equivalent
accuracy, deterministic, ~25% fewer parameters (15.5k vs 20k), faster to train
(matters for daily CPU retraining in production).

## Decision 3 — feature set ohl vs rbg (VMD models, mean RMSE)

| model | ohl | rbg | Δ (rbg−ohl) | rbg better in |
|---|--:|--:|--:|--:|
| vmd-lstm | 8.057 | 8.076 | +0.018 | 4/8 |
| vmd-gru | 8.071 | 8.082 | +0.012 | 5/8 |
| vmd-transformer | 8.072 | 8.037 | −0.035 | 3/8 |

All within noise. **Engineered range/body/gap features neither help nor hurt**
versus raw OHL. The close decomposition carries the signal; the auxiliary set is
immaterial. Keep `ohl` (default, simplest) — `rbg` is more principled but
empirically equivalent.

## Decision 4 — best network vs naive (RMSE, ohl)

| ticker | naive | best NN | model | Δ |
|---|--:|--:|---|--:|
| AAPL | 5.764 | 5.644 | lstm | −0.121 |
| NVDA | 13.714 | 13.572 | gru | −0.143 |
| TSLA | 11.979 | 11.815 | vmd-transformer | −0.165 |
| MSFT | 4.829 | 4.828 | lstm | −0.001 |
| ABNB | 6.339 | 5.747 | lstm | −0.592 |
| DDOG | 10.173 | 10.166 | vmd-gru | −0.006 |
| PEP | 3.711 | 3.671 | vmd-lstm | −0.040 |
| INTC | 8.733 | 8.719 | vmd-lstm | −0.014 |

Every network beats the random-walk naive, but by negligible margins for half
the tickers (MSFT, DDOG, INTC, PEP ≈ tied). Meaningful edge only on high-volatility
names (ABNB, TSLA, NVDA, AAPL). **10-day price level is barely more predictable
than a random walk** for stable tickers.

## Directional accuracy (mean across tickers, ohl)

| model | dir_acc |
|---|--:|
| lstm | 0.5665 |
| vmd-gru | 0.5669 |
| vmd-lstm | 0.5611 |
| gru | 0.5594 |
| vmd-transformer | 0.5550 |
| transformer | 0.5395 |

All modestly above 0.5. **Direction is where the (small) real signal is**, more
than RMSE suggests — relevant because the product reports a trend, not a price.

## Synthesis

- **Architecture: GRU.** Tied with LSTM within noise → cheaper, deterministic, fewer params.
- **VMD: no measurable benefit at fixed hyperparameters.** Decision deferred to the
  tuning phase (tune GRU and VMD-GRU, let the tuned comparison plus the short-history
  cost of `decomp_len` decide). Honest, publishable methodological result.
- **Feature set: ohl ≈ rbg.** Keep the simplest.
- **Predictability:** level barely beats random walk except on volatile tickers;
  direction ~0.56 is the modest real edge → lean the product on the trend framing.
