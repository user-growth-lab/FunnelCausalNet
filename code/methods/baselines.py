"""Lightweight meta-learner baselines for uplift / ITE estimation.

Implementations follow Künzel et al. (PNAS 2019) and Kennedy (EJS 2023).
All learners share a common interface:

    learner = TLearner(...)
    learner.fit(X, T, Y, pi=None)      # pi is propensity P(T=1|X) on X
    tau_hat = learner.predict_ite(X)

Estimators are scikit-learn regressors (GradientBoostingRegressor by default)
to keep dependencies minimal. The shared `BaselineConfig` controls capacity.

Why these four:
    - S-Learner : minimal-modeling baseline; underestimates HTE when T is weak
    - T-Learner : matches the structural assumption of our DualHeadNet
                  (independent per-arm estimators) without representation sharing
    - X-Learner : meta-learner robust to imbalanced treatment assignment
    - DR-Learner: doubly-robust meta-learner; oracle property under either
                  outcome or propensity correctness; common 'gold' baseline
                  in modern uplift papers (Hines et al. 2022, Curth & Schaar
                  ICML 2021)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------


@dataclass
class BaselineConfig:
    """Per-learner regressor capacity. Keep modest for speed across seeds."""

    n_estimators: int = 100
    max_depth: int = 3
    subsample: float = 0.8
    learning_rate: float = 0.1
    random_state: int = 0
    propensity_clip: float = 0.05    # clip pi(x) to [clip, 1-clip] for X/DR


def _make_logreg(seed: int = 0) -> LogisticRegression:
    """Default propensity model. lbfgs is fast and stable on standardized X."""
    return LogisticRegression(
        solver="lbfgs", max_iter=1000, C=1.0, random_state=seed,
    )


def _fit_propensity(X: np.ndarray, T: np.ndarray, seed: int = 0) -> np.ndarray:
    """Logistic regression propensity score on X."""
    lr = _make_logreg(seed)
    lr.fit(X, T)
    return lr.predict_proba(X)[:, 1]


def _make_gbr(cfg: BaselineConfig, seed_offset: int = 0) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        subsample=cfg.subsample,
        learning_rate=cfg.learning_rate,
        random_state=cfg.random_state + seed_offset,
    )


# ---------------------------------------------------------------------------
# S-Learner
# ---------------------------------------------------------------------------


class SLearner:
    """S-Learner: single regressor on the augmented feature [X, T].

    tau_hat(x) = mu(X, T=1) - mu(X, T=0)
    """

    name = "S-Learner"

    def __init__(self, cfg: Optional[BaselineConfig] = None) -> None:
        self.cfg = cfg or BaselineConfig()
        self.model: Optional[GradientBoostingRegressor] = None

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray,
            pi: Optional[np.ndarray] = None) -> "SLearner":
        Xa = np.concatenate([X, T.reshape(-1, 1).astype(np.float64)], axis=1)
        self.model = _make_gbr(self.cfg)
        self.model.fit(Xa, Y)
        return self

    def predict_ite(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("SLearner.fit must be called first")
        N = X.shape[0]
        X1 = np.concatenate([X, np.ones((N, 1))], axis=1)
        X0 = np.concatenate([X, np.zeros((N, 1))], axis=1)
        return self.model.predict(X1) - self.model.predict(X0)


# ---------------------------------------------------------------------------
# T-Learner
# ---------------------------------------------------------------------------


class TLearner:
    """T-Learner: independent per-arm regressors mu_0, mu_1.

    tau_hat(x) = mu_1(x) - mu_0(x).
    """

    name = "T-Learner"

    def __init__(self, cfg: Optional[BaselineConfig] = None) -> None:
        self.cfg = cfg or BaselineConfig()
        self.mu_0: Optional[GradientBoostingRegressor] = None
        self.mu_1: Optional[GradientBoostingRegressor] = None

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray,
            pi: Optional[np.ndarray] = None) -> "TLearner":
        idx0 = np.where(T == 0)[0]
        idx1 = np.where(T == 1)[0]
        if len(idx0) < 10 or len(idx1) < 10:
            raise ValueError("Each arm needs at least 10 samples to fit T-Learner")
        self.mu_0 = _make_gbr(self.cfg, seed_offset=0)
        self.mu_1 = _make_gbr(self.cfg, seed_offset=1)
        self.mu_0.fit(X[idx0], Y[idx0])
        self.mu_1.fit(X[idx1], Y[idx1])
        return self

    def predict_potentials(self, X: np.ndarray):
        if self.mu_0 is None or self.mu_1 is None:
            raise RuntimeError("TLearner.fit must be called first")
        return self.mu_0.predict(X), self.mu_1.predict(X)

    def predict_ite(self, X: np.ndarray) -> np.ndarray:
        m0, m1 = self.predict_potentials(X)
        return m1 - m0


# ---------------------------------------------------------------------------
# X-Learner (Künzel et al. 2019)
# ---------------------------------------------------------------------------


class XLearner:
    """X-Learner: T-Learner + cross-update with IPW combining.

    Stage 1: fit mu_0, mu_1 on each arm (T-Learner step)
    Stage 2: build pseudo-treatment-effects:
        D_1 = Y_1 - mu_0(X_1)        (treated)
        D_0 = mu_1(X_0) - Y_0        (controls)
    Stage 3: fit two regressors:
        tau_1(x) ~ D_1 on treated
        tau_0(x) ~ D_0 on controls
    Stage 4: combine with propensity:
        tau_hat(x) = pi(x) * tau_0(x) + (1 - pi(x)) * tau_1(x)
    """

    name = "X-Learner"

    def __init__(self, cfg: Optional[BaselineConfig] = None) -> None:
        self.cfg = cfg or BaselineConfig()
        self.t_learner: Optional[TLearner] = None
        self.tau_0_model: Optional[GradientBoostingRegressor] = None
        self.tau_1_model: Optional[GradientBoostingRegressor] = None
        self._pi_estimator: Optional[LogisticRegression] = None
        self._pi_external: bool = False

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray,
            pi: Optional[np.ndarray] = None) -> "XLearner":
        self.t_learner = TLearner(self.cfg).fit(X, T, Y)
        m0, m1 = self.t_learner.predict_potentials(X)

        idx0 = np.where(T == 0)[0]
        idx1 = np.where(T == 1)[0]
        D_1 = Y[idx1] - m0[idx1]      # observed treated minus counterfactual
        D_0 = m1[idx0] - Y[idx0]      # counterfactual minus observed control

        self.tau_1_model = _make_gbr(self.cfg, seed_offset=2)
        self.tau_0_model = _make_gbr(self.cfg, seed_offset=3)
        self.tau_1_model.fit(X[idx1], D_1)
        self.tau_0_model.fit(X[idx0], D_0)

        if pi is None:
            self._pi_estimator = _make_logreg(self.cfg.random_state)
            self._pi_estimator.fit(X, T)
            self._pi_external = False
        else:
            self._pi_external = True
            self._pi_train = np.asarray(pi, dtype=np.float64)
        return self

    def _pi_at(self, X: np.ndarray) -> np.ndarray:
        if self._pi_estimator is not None:
            pi = self._pi_estimator.predict_proba(X)[:, 1]
        else:
            # When pi was provided on the fit set we can't extrapolate to X;
            # for a fair X-Learner at test time we still need a model. Refit
            # logistic regression on the closest available data: error out
            # to make the contract explicit.
            raise RuntimeError(
                "XLearner with externally provided pi can't extrapolate; "
                "either omit pi at fit time so an internal estimator is "
                "built, or provide pi for prediction explicitly."
            )
        clip = self.cfg.propensity_clip
        return np.clip(pi, clip, 1.0 - clip)

    def predict_ite(self, X: np.ndarray) -> np.ndarray:
        if self.tau_0_model is None or self.tau_1_model is None:
            raise RuntimeError("XLearner.fit must be called first")
        tau0 = self.tau_0_model.predict(X)
        tau1 = self.tau_1_model.predict(X)
        pi = self._pi_at(X)
        return pi * tau0 + (1.0 - pi) * tau1


# ---------------------------------------------------------------------------
# DR-Learner (Kennedy EJS 2023)
# ---------------------------------------------------------------------------


class DRLearner:
    """DR-Learner: regress the DR pseudo-outcome on X.

    Stage 1: fit nuisances mu_0, mu_1, pi on the *fit* split
    Stage 2: build DR pseudo-outcome on the *cross-fit* split:
        Y_pseudo = mu_1(X) - mu_0(X)
                   + (T - pi)/(pi(1-pi)) * (Y - mu_T(X))
    Stage 3: fit a regressor tau_hat(X) ~ Y_pseudo

    For simplicity and matching paper's compute budget we drop K-fold
    cross-fitting and use the same data for nuisance fitting and
    pseudo-outcome construction. Coverage / consistency arguments still
    hold under sample-splitting but bias may be larger; this is the
    standard 'plug-in' DR-Learner used in many empirical studies.
    """

    name = "DR-Learner"

    def __init__(self, cfg: Optional[BaselineConfig] = None) -> None:
        self.cfg = cfg or BaselineConfig()
        self.t_learner: Optional[TLearner] = None
        self.tau_model: Optional[GradientBoostingRegressor] = None
        self._pi_estimator: Optional[LogisticRegression] = None
        self._pi_external: bool = False

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray,
            pi: Optional[np.ndarray] = None) -> "DRLearner":
        self.t_learner = TLearner(self.cfg).fit(X, T, Y)
        m0, m1 = self.t_learner.predict_potentials(X)
        m_T = np.where(T == 1, m1, m0)

        if pi is None:
            self._pi_estimator = _make_logreg(self.cfg.random_state)
            self._pi_estimator.fit(X, T)
            pi_hat = self._pi_estimator.predict_proba(X)[:, 1]
            self._pi_external = False
        else:
            pi_hat = np.asarray(pi, dtype=np.float64)
            self._pi_external = True

        clip = self.cfg.propensity_clip
        pi_hat = np.clip(pi_hat, clip, 1.0 - clip)

        weight = (T - pi_hat) / (pi_hat * (1.0 - pi_hat))
        Y_pseudo = (m1 - m0) + weight * (Y - m_T)

        self.tau_model = _make_gbr(self.cfg, seed_offset=4)
        self.tau_model.fit(X, Y_pseudo)
        return self

    def predict_ite(self, X: np.ndarray) -> np.ndarray:
        if self.tau_model is None:
            raise RuntimeError("DRLearner.fit must be called first")
        return self.tau_model.predict(X)


# ---------------------------------------------------------------------------
# Convenience registry
# ---------------------------------------------------------------------------


BASELINE_REGISTRY = {
    "S-Learner":  SLearner,
    "T-Learner":  TLearner,
    "X-Learner":  XLearner,
    "DR-Learner": DRLearner,
}


def fit_baseline(name: str, X: np.ndarray, T: np.ndarray, Y: np.ndarray,
                 pi: Optional[np.ndarray] = None,
                 cfg: Optional[BaselineConfig] = None):
    if name not in BASELINE_REGISTRY:
        raise ValueError(
            f"unknown baseline '{name}'; choose one of {list(BASELINE_REGISTRY)}"
        )
    return BASELINE_REGISTRY[name](cfg).fit(X, T, Y, pi=pi)
