"""Criteo Uplift v2.1 loader with stratified subsampling.

Reference: Diemert et al. (2018), "A Large Scale Benchmark for Uplift Modeling",
published by Criteo AI Labs.  The dataset has ~14M rows, 12 anonymized float
features (f0-f11), a binary treatment indicator, and binary outcomes
(``visit``, ``conversion``).

Since the full dataset is very large (~2.5 GB uncompressed), we load with a
configurable stratified subsample that preserves the treatment ratio (~85%
treated / 15% control).  The default subsample size is 100K.

The returned bundle follows the same schema as ``hillstrom_loader.load_hillstrom``
so it plugs directly into the experiment scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
_FALLBACK_PATHS = [
    _REPO_ROOT / "data" / "criteo-uplift" / "criteo-uplift-v2.1.csv.gz",
    _REPO_ROOT / "data" / "criteo" / "criteo_uplift.csv",
]

_FEATURE_COLS = [f"f{i}" for i in range(12)]


def _resolve_path(path: Optional[Path]) -> Path:
    candidates: List[Path] = []
    if path is not None:
        candidates.append(Path(path))
    candidates.extend(p.resolve() for p in _FALLBACK_PATHS)
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Criteo Uplift file not found in any of:\n  "
        + "\n  ".join(str(p) for p in candidates)
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CriteoConfig:
    subsample_n: int = 100_000
    outcome: str = "conversion"       # "conversion" or "visit"
    standardize: bool = True
    seed: int = 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_criteo(
    cfg: Optional[CriteoConfig] = None,
    path: Optional[Path] = None,
) -> Dict[str, object]:
    """Load Criteo Uplift v2.1 with stratified subsampling.

    Returns
    -------
    dict with keys
        X : (N, 12) float64 covariate matrix (z-scored if standardize=True)
        T : (N,)  int64   binary treatment
        Y : (N,)  float64 binary outcome
        feature_names : list[str]
        propensity   : (N,) float64 (constant, empirical treatment rate)
        meta         : dict with summary stats
    """
    cfg = cfg or CriteoConfig()
    if cfg.outcome not in ("conversion", "visit"):
        raise ValueError(f"outcome must be 'conversion' or 'visit', got {cfg.outcome!r}")

    csv_path = _resolve_path(path)

    # Read all columns we need (skip 'exposure' to save memory)
    usecols = _FEATURE_COLS + ["treatment", cfg.outcome]
    df = pd.read_csv(csv_path, usecols=usecols)

    # Stratified subsample: preserve treatment ratio
    rng = np.random.default_rng(cfg.seed)
    n_total = len(df)
    if cfg.subsample_n < n_total:
        t_mask = df["treatment"].values == 1
        n_t = int(t_mask.sum())
        n_c = n_total - n_t
        ratio = n_t / n_total

        n_t_sample = int(round(cfg.subsample_n * ratio))
        n_c_sample = cfg.subsample_n - n_t_sample

        idx_t = rng.choice(np.where(t_mask)[0], size=min(n_t_sample, n_t), replace=False)
        idx_c = rng.choice(np.where(~t_mask)[0], size=min(n_c_sample, n_c), replace=False)
        idx = np.sort(np.concatenate([idx_t, idx_c]))
        df = df.iloc[idx].reset_index(drop=True)

    T = df["treatment"].to_numpy(dtype=np.int64)
    Y = df[cfg.outcome].to_numpy(dtype=np.float64)
    X = df[_FEATURE_COLS].to_numpy(dtype=np.float64)

    # Handle NaN/Inf
    nan_mask = ~np.isfinite(X)
    if nan_mask.any():
        col_means = np.nanmean(X, axis=0)
        for j in range(X.shape[1]):
            X[nan_mask[:, j], j] = col_means[j]

    # Z-score standardize
    if cfg.standardize:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)
        X = (X - mu) / sd

    pi_val = float(T.mean())
    pi = np.full(X.shape[0], pi_val, dtype=np.float64)

    meta = {
        "path": str(csv_path),
        "n_full_dataset": n_total,
        "n_subsample": int(X.shape[0]),
        "n_treated": int(T.sum()),
        "n_control": int((1 - T).sum()),
        "treatment_rate": pi_val,
        "outcome": cfg.outcome,
        "y_mean_treated": float(Y[T == 1].mean()) if (T == 1).any() else 0.0,
        "y_mean_control": float(Y[T == 0].mean()) if (T == 0).any() else 0.0,
        "naive_ate": (
            float(Y[T == 1].mean() - Y[T == 0].mean())
            if (T == 1).any() and (T == 0).any() else 0.0
        ),
        "d_features": int(X.shape[1]),
    }

    return {
        "X": X,
        "T": T,
        "Y": Y,
        "feature_names": _FEATURE_COLS,
        "propensity": pi,
        "meta": meta,
    }


def split_train_calib_test(
    bundle: Dict[str, object],
    test_frac: float = 0.25,
    calib_frac: float = 0.20,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Three-way split: train / calib / test.

    Same logic as ``hillstrom_loader.split_train_calib_test``.
    """
    N = bundle["X"].shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)

    n_test = int(round(N * test_frac))
    n_rest = N - n_test
    n_calib = int(round(n_rest * calib_frac))

    test_idx = idx[:n_test]
    calib_idx = idx[n_test : n_test + n_calib]
    train_idx = idx[n_test + n_calib :]

    return {
        "train_idx": np.sort(train_idx),
        "calib_idx": np.sort(calib_idx),
        "test_idx": np.sort(test_idx),
    }
