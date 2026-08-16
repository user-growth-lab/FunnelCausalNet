"""Hillstrom MineThatData E-Mail Analytics challenge loader.

Reference: Kevin Hillstrom (2008), https://blog.minethatdata.com/2008/03/.
The dataset has 64,000 customers, randomly split into three groups:
    - "Womens E-Mail"  (treatment, 1/3)
    - "Mens E-Mail"    (treatment, 1/3)
    - "No E-Mail"      (control,   1/3)

Three outcomes are recorded over the following two weeks: ``visit`` (binary),
``conversion`` (binary), ``spend`` (continuous, USD).

This loader returns a feature matrix X, a binary treatment T, and an outcome Y
ready to feed the dual-head network. By default we use the Womens E-Mail vs
No E-Mail comparison (the cleanest 1:1 randomized contrast) with ``visit`` as
outcome, which is the convention used in most uplift papers.

Notes
-----
- Hillstrom has no ground-truth ITE / sleeping-dog label. Evaluation must rely
  on policy-value style estimators (Qini, AUUC, IPW value) and on the
  semi-synthetic counterpart (``sleeping_dog_generator.generate``) for any
  claim about coverage / sleeping-dog precision.
- Categorical encoding follows a fixed convention documented in this repo's
  README: one-hot for ``history_segment``, ``zip_code``, ``channel``; numeric for the rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
_LOCAL_RAW = _REPO_ROOT / "data" / "raw" / "hillstrom.csv"
_FALLBACK_PATHS = [
    _REPO_ROOT
    / "data"
    / "hillstrom"
    / "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv",
    _REPO_ROOT / "data" / "hillstrom.csv",
]


def _resolve_path(path: Optional[Path]) -> Path:
    """Pick the first existing path among (user, local, fallbacks)."""
    candidates: List[Path] = []
    if path is not None:
        candidates.append(Path(path))
    candidates.append(_LOCAL_RAW.resolve())
    candidates.extend(p.resolve() for p in _FALLBACK_PATHS)

    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "hillstrom.csv not found in any of:\n  " +
        "\n  ".join(str(p) for p in candidates)
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VALID_TREATMENT = ("womens", "mens", "any")
VALID_OUTCOME = ("visit", "conversion", "spend")
_NUMERIC_COLS = ("recency", "history", "mens", "womens", "newbie")
_ONEHOT_COLS = ("history_segment", "zip_code", "channel")


@dataclass
class HillstromConfig:
    treatment: str = "womens"          # one of VALID_TREATMENT
    outcome: str = "visit"             # one of VALID_OUTCOME
    standardize: bool = True           # z-score numeric columns
    propensity_known: float = 0.5      # randomized -> 1/2 by construction
    keep_only_used_segments: bool = True   # drop Mens E-Mail when treatment="womens", etc.

    def validate(self) -> None:
        if self.treatment not in VALID_TREATMENT:
            raise ValueError(f"treatment must be in {VALID_TREATMENT}, got {self.treatment!r}")
        if self.outcome not in VALID_OUTCOME:
            raise ValueError(f"outcome must be in {VALID_OUTCOME}, got {self.outcome!r}")


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _build_treatment(seg: pd.Series, mode: str) -> np.ndarray:
    """Map the ``segment`` column to a binary treatment indicator T."""
    if mode == "womens":
        return (seg == "Womens E-Mail").to_numpy().astype(np.int64)
    if mode == "mens":
        return (seg == "Mens E-Mail").to_numpy().astype(np.int64)
    if mode == "any":
        return seg.isin(["Womens E-Mail", "Mens E-Mail"]).to_numpy().astype(np.int64)
    raise ValueError(mode)


def _filter_segments(df: pd.DataFrame, mode: str, drop_unused: bool) -> pd.DataFrame:
    """Keep only the (treatment, control) rows relevant to ``mode``.

    Hillstrom is a 3-arm design. For a clean 1:1 contrast we drop the unused arm.
    """
    if not drop_unused:
        return df
    if mode == "womens":
        keep = df["segment"].isin(["Womens E-Mail", "No E-Mail"])
    elif mode == "mens":
        keep = df["segment"].isin(["Mens E-Mail", "No E-Mail"])
    else:
        keep = df["segment"].isin(["Womens E-Mail", "Mens E-Mail", "No E-Mail"])
    return df.loc[keep].reset_index(drop=True)


def _encode_features(
    df: pd.DataFrame, standardize: bool,
) -> tuple[np.ndarray, List[str]]:
    """Build a (N, d) float matrix and return the column names.

    Numeric columns are z-scored (when ``standardize=True``); categorical columns
    are one-hot encoded with a deterministic column order.
    """
    parts: List[np.ndarray] = []
    names: List[str] = []

    numeric = df.loc[:, list(_NUMERIC_COLS)].to_numpy(dtype=np.float64)
    if standardize:
        mu = numeric.mean(axis=0, keepdims=True)
        sd = numeric.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)
        numeric = (numeric - mu) / sd
    parts.append(numeric)
    names.extend(list(_NUMERIC_COLS))

    for col in _ONEHOT_COLS:
        cats = sorted(df[col].dropna().unique().tolist())
        for c in cats:
            parts.append((df[col] == c).to_numpy(dtype=np.float64).reshape(-1, 1))
            names.append(f"{col}={c}")

    X = np.concatenate(parts, axis=1)
    return X, names


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_hillstrom(
    cfg: Optional[HillstromConfig] = None,
    path: Optional[Path] = None,
) -> Dict[str, object]:
    """Load Hillstrom and return a bundle ready for the uplift pipeline.

    Returns
    -------
    dict with keys
        X : (N, d) float64 covariate matrix
        T : (N,)  int64   binary treatment indicator
        Y : (N,)  float64 outcome (binary for visit/conversion, USD for spend)
        feature_names : list[str] length d
        propensity   : (N,) float64 (constant = cfg.propensity_known)
        meta         : dict with summary stats and the resolved file path
    """
    cfg = cfg or HillstromConfig()
    cfg.validate()

    csv_path = _resolve_path(path)
    df = pd.read_csv(csv_path)

    expected_cols = {"recency", "history_segment", "history", "mens", "womens",
                     "zip_code", "newbie", "channel", "segment",
                     "visit", "conversion", "spend"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Hillstrom file missing expected columns: {sorted(missing)}")

    df = _filter_segments(df, cfg.treatment, cfg.keep_only_used_segments)

    T = _build_treatment(df["segment"], cfg.treatment)
    Y = df[cfg.outcome].to_numpy(dtype=np.float64)
    X, feature_names = _encode_features(df, standardize=cfg.standardize)

    pi = np.full(X.shape[0], cfg.propensity_known, dtype=np.float64)

    meta = {
        "path": str(csv_path),
        "n_total": int(X.shape[0]),
        "n_treated": int(T.sum()),
        "n_control": int((1 - T).sum()),
        "treatment_mode": cfg.treatment,
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
        "feature_names": feature_names,
        "propensity": pi,
        "meta": meta,
    }


def inject_sleeping_dog(
    bundle: Dict[str, object],
    rho_inj: float,
    seed: int = 0,
    flip_prob: float = 0.5,
    rule: str = "recency_history",
) -> Dict[str, object]:
    """Inject a synthetic sleeping-dog signal into the Hillstrom outcome.

    Hillstrom's natural visit/conversion data has essentially no sleeping-dog
    behaviour (treated visit rate >= control everywhere), so risk-sensitive
    methods cannot demonstrate any value on the raw data. To justify our
    framework on real-data marketing covariates we synthetically install a
    sub-population that "responds negatively to e-mails" by stochastically
    suppressing their visit/conversion outcome under T=1.

    Procedure
    ---------
    1. Pick a trigger sub-population by ``rule`` (default: high-activity users
       defined as ``recency <= 3 AND history <= 50``, simulating the empirically
       documented "low-recency low-monetary users find e-mail intrusive"
       segment in marketing literature).
    2. Restrict to a random ``rho_inj`` fraction of the trigger population
       (controls the injected sleeping-dog rate so we can scan the magnitude).
    3. For every selected user with T=1, flip their outcome 1 -> 0 with
       probability ``flip_prob`` (Bernoulli; ``flip_prob = 0.5`` means we
       induce a roughly -0.5 ITE for those users).

    The control arm (T=0) is left untouched so the randomization assumption
    used by IPW value estimators remains valid.

    Returns
    -------
    A NEW bundle dict with the same keys as ``load_hillstrom``, plus
    ``meta["injection"]`` describing what was done.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(bundle["X"])
    T = np.asarray(bundle["T"]).astype(np.int64)
    Y = np.asarray(bundle["Y"]).astype(np.float64).copy()
    pi = np.asarray(bundle["propensity"])
    feature_names = list(bundle["feature_names"])

    # Recency / history live in the first two numeric columns by construction
    # of _encode_features (they are standardized -> we need their pre-scale
    # values, but the loader does NOT keep the raw frame. We fall back to
    # quantile-based proxies on the standardized features.)
    if rule == "recency_history":
        rec_idx = feature_names.index("recency")
        hist_idx = feature_names.index("history")
        rec_z = X[:, rec_idx]
        hist_z = X[:, hist_idx]
        # "recency <= 3" approximates the bottom-third of standardized recency;
        # "history <= 50" approximates the bottom-third of standardized history.
        rec_thr = float(np.quantile(rec_z, 1.0 / 3.0))
        hist_thr = float(np.quantile(hist_z, 1.0 / 3.0))
        trigger_mask = (rec_z <= rec_thr) & (hist_z <= hist_thr)
    elif rule == "all":
        trigger_mask = np.ones(len(T), dtype=bool)
    else:
        raise ValueError(f"unknown injection rule: {rule!r}")

    n_trigger = int(trigger_mask.sum())
    n_target = int(round(rho_inj * len(T)))
    n_target = min(n_target, n_trigger)

    trigger_idx = np.flatnonzero(trigger_mask)
    chosen = rng.choice(trigger_idx, size=n_target, replace=False)

    treated_chosen = chosen[T[chosen] == 1]
    flip_decisions = rng.random(len(treated_chosen)) < flip_prob
    flipped_idx = treated_chosen[flip_decisions]

    n_actual_flipped = 0
    for i in flipped_idx:
        if Y[i] > 0.5:
            Y[i] = 0.0
            n_actual_flipped += 1

    new_bundle = dict(bundle)
    new_bundle["Y"] = Y
    new_meta = dict(bundle["meta"])
    new_meta["injection"] = {
        "rule": rule,
        "rho_inj": float(rho_inj),
        "flip_prob": float(flip_prob),
        "seed": int(seed),
        "n_trigger_pool": int(n_trigger),
        "n_chosen": int(len(chosen)),
        "n_treated_in_chosen": int(len(treated_chosen)),
        "n_flipped": int(n_actual_flipped),
        "y_mean_after": float(Y.mean()),
        "y_mean_treated_after": float(Y[T == 1].mean()) if (T == 1).any() else 0.0,
        "y_mean_control_after": float(Y[T == 0].mean()) if (T == 0).any() else 0.0,
        "naive_ate_after": (
            float(Y[T == 1].mean() - Y[T == 0].mean())
            if (T == 1).any() and (T == 0).any() else 0.0
        ),
    }
    new_bundle["meta"] = new_meta
    return new_bundle


def split_train_calib_test(
    bundle: Dict[str, object],
    test_frac: float = 0.25,
    calib_frac: float = 0.20,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Three-way split for the conformal pipeline.

    Returns a dict with ``train_idx``, ``calib_idx``, ``test_idx`` numpy arrays.
    Test fraction is taken first; calib_frac is then a fraction of the remaining
    pool (so calib_frac=0.20 means 20% of the train+calib pool).
    """
    n = bundle["X"].shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    n_test = int(round(n * test_frac))
    test_idx = perm[:n_test]
    pool = perm[n_test:]
    n_calib = int(round(len(pool) * calib_frac))
    calib_idx = pool[:n_calib]
    train_idx = pool[n_calib:]

    return {"train_idx": train_idx, "calib_idx": calib_idx, "test_idx": test_idx}
