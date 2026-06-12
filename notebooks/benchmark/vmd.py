"""Causal VMD feature extraction: each sample decomposes only its own past."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
from vmdpy import VMD

logger = logging.getLogger(__name__)

# Fixed VMD parameters (standard values in the literature).
VMD_TAU = 0.0  # exact reconstruction not enforced (noise-tolerant)
VMD_DC = 0  # no mode pinned at zero frequency
VMD_INIT = 1  # uniform initialization of center frequencies
VMD_TOL = 1e-7


def _cache_key(close: np.ndarray, lookback: int, horizon: int, decomp_len: int, k: int, alpha: float, t_start: int) -> str:
    digest = hashlib.sha1(close.tobytes()).hexdigest()[:12]
    return f"vmd_{digest}_lb{lookback}_h{horizon}_L{decomp_len}_K{k}_a{alpha:g}_t{t_start}"


def causal_vmd_modes(
    close: np.ndarray,
    lookback: int,
    horizon: int,
    decomp_len: int = 512,
    k: int = 6,
    alpha: float = 2000.0,
    t_start: int | None = None,
    cache_dir: Path | None = None,
    ticker: str = "",
) -> np.ndarray:
    """VMD mode windows computed causally for every sample.

    For each last-known day t, the `decomp_len` closes ending at t (and only
    those) are decomposed into k modes; the last `lookback` columns form the
    sample's mode window, z-scored per mode. This matches inference-time
    conditions exactly: the model always sees right-edge decompositions of
    past data.

    Returns an array of shape (n_samples, lookback, k) aligned with
    build_windows(..., t_start=t_start).
    """
    if decomp_len % 2:
        raise ValueError(f"decomp_len must be even, got {decomp_len}")
    if t_start is None:
        t_start = decomp_len - 1
    if t_start < decomp_len - 1:
        raise ValueError(f"t_start={t_start} smaller than decomp_len-1={decomp_len - 1}")

    close = np.asarray(close, dtype=np.float64)
    cache_file: Path | None = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        name = _cache_key(close, lookback, horizon, decomp_len, k, alpha, t_start)
        cache_file = cache_dir / f"{ticker}_{name}.npz"
        if cache_file.exists():
            logger.info("VMD cache hit: %s", cache_file.name)
            return np.load(cache_file)["modes"]

    n_samples = len(close) - horizon - t_start
    modes = np.empty((n_samples, lookback, k), dtype=np.float32)
    for i, t in enumerate(range(t_start, len(close) - horizon)):
        segment = close[t - decomp_len + 1 : t + 1]
        u, _, _ = VMD(segment, alpha, VMD_TAU, k, VMD_DC, VMD_INIT, VMD_TOL)
        window = u[:, -lookback:].T.astype(np.float64)  # (lookback, k)
        mu = window.mean(axis=0)
        sd = window.std(axis=0)
        sd[sd < 1e-8] = 1.0
        modes[i] = (window - mu) / sd
        if (i + 1) % 1000 == 0:
            logger.info("VMD decompositions: %d/%d", i + 1, n_samples)

    if cache_file is not None:
        np.savez_compressed(cache_file, modes=modes)
        logger.info("VMD cache written: %s", cache_file.name)
    return modes
