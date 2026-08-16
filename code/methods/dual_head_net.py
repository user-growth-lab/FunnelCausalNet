"""Dual-Head Decoupled Network for risk-aware uplift modeling.

Mathematical alignment: shared encoder with dual outcome heads (control vs
treatment), consistent with standard dual-head uplift formulations:

    phi(x)         = shared encoder backbone
    mu_0(x)        = control-head MLP on phi(x)        (positive-effect head*)
    mu_1(x)        = treatment-head MLP on phi(x)      (adverse-effect head*)
    tau_hat(x)     = mu_1(x) - mu_0(x)                 (point ITE estimate)
    residuals      = (Y - mu_T(x)) on a held-out fold  (used by Conformal layer)

(*) The "positive-effect head" / "adverse-effect head" framing in the paper
is conceptual: in implementation, we estimate mu_0 and mu_1 separately, and
the dual-head structure ensures the adverse signal in mu_1 is decoupled from
the dominant positive trend captured by mu_0. The Conformal Adverse-Effect
Detection layer (W1-5) then converts (mu_0, mu_1) plus residual quantiles
into a confident sleeping-dog flag tau_high(x) < 0.

Training strategy (T-Learner with shared encoder):
    - Each sample contributes a loss only through its observed-treatment head.
    - Gradient flows through the shared encoder, balancing both heads.
    - Optional treatment-balanced sampling to mitigate propensity skew.
    - Y is internally standardized; predictions are de-standardized at the end.

Reference design lineage:
    - Shalit et al. (TARNet, ICML 2017): shared encoder + per-treatment heads
    - Künzel et al. (T-Learner, PNAS 2019): per-treatment training objective
    - Built on top of prior uplift-baseline nn_base patterns (shared trunk + task heads).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.device import pick_device


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DualHeadConfig:
    """Architecture hyper-parameters for the dual-head network."""

    d_in: int = 25
    rep_hidden: List[int] = field(default_factory=lambda: [128, 64])
    rep_dim: int = 32
    head_hidden: List[int] = field(default_factory=lambda: [32])
    dropout: float = 0.1
    activation: str = "elu"  # one of: elu, relu, gelu

    # ----- Quantile heads (Conformal Quantile Regression for ITE) -----
    # When enabled, two extra heads (lo, hi) per arm produce X-dependent
    # quantile predictions trained with the pinball loss. Combined with the
    # CQR calibration in `conformal_calibration.mode='cqr'`, this yields
    # tau_low(X) / tau_high(X) that directly target the *tail* of the ITE
    # distribution rather than relying on residual dispersion as a proxy.
    quantile_heads: bool = False
    quantile_lo: float = 0.05
    quantile_hi: float = 0.95
    pinball_weight: float = 1.0    # weight of the pinball loss vs MSE


@dataclass
class TrainConfig:
    """Training hyper-parameters."""

    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 15
    val_frac: float = 0.2
    seed: int = 0
    verbose: bool = False
    grad_clip: float = 5.0
    treatment_balance: bool = True   # batch-level resampling to balance T=0/T=1


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


_ACT_MAP = {"elu": nn.ELU, "relu": nn.ReLU, "gelu": nn.GELU}


def _mlp(in_dim: int, hidden_dims: List[int], out_dim: int,
         activation: str = "elu", dropout: float = 0.0) -> nn.Sequential:
    act_cls = _ACT_MAP[activation]
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(act_cls())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# DualHeadNet module
# ---------------------------------------------------------------------------


class DualHeadNet(nn.Module):
    """Shared encoder phi(x) + two per-treatment regression heads.

    Forward modes:
        forward(x)               -> (mu_0_pred, mu_1_pred)        each (N,)
        forward(x, t_idx=t)      -> mu_T(x) for the observed t    (N,)
        forward(x, return_phi=True) -> (mu_0, mu_1, phi)
    """

    def __init__(self, cfg: DualHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = _mlp(cfg.d_in, cfg.rep_hidden, cfg.rep_dim,
                            activation=cfg.activation, dropout=cfg.dropout)
        self.head_0 = _mlp(cfg.rep_dim, cfg.head_hidden, 1,
                           activation=cfg.activation, dropout=cfg.dropout)
        self.head_1 = _mlp(cfg.rep_dim, cfg.head_hidden, 1,
                           activation=cfg.activation, dropout=cfg.dropout)
        if cfg.quantile_heads:
            # Lo / hi quantile heads per arm; same architecture as the mean
            # heads. Trained with pinball loss in parallel with the MSE loss.
            self.head_0_lo = _mlp(cfg.rep_dim, cfg.head_hidden, 1,
                                  activation=cfg.activation, dropout=cfg.dropout)
            self.head_0_hi = _mlp(cfg.rep_dim, cfg.head_hidden, 1,
                                  activation=cfg.activation, dropout=cfg.dropout)
            self.head_1_lo = _mlp(cfg.rep_dim, cfg.head_hidden, 1,
                                  activation=cfg.activation, dropout=cfg.dropout)
            self.head_1_hi = _mlp(cfg.rep_dim, cfg.head_hidden, 1,
                                  activation=cfg.activation, dropout=cfg.dropout)
        else:
            self.head_0_lo = None
            self.head_0_hi = None
            self.head_1_lo = None
            self.head_1_hi = None

    def representation(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(
        self, x: torch.Tensor,
        t_idx: Optional[torch.Tensor] = None,
        return_phi: bool = False,
    ):
        phi = self.encoder(x)
        mu_0 = self.head_0(phi).squeeze(-1)
        mu_1 = self.head_1(phi).squeeze(-1)

        if t_idx is None:
            if return_phi:
                return mu_0, mu_1, phi
            return mu_0, mu_1

        if t_idx.dtype != torch.long:
            t_idx = t_idx.long()
        observed = torch.where(t_idx == 0, mu_0, mu_1)
        if return_phi:
            return observed, phi
        return observed

    def forward_quantiles(self, x: torch.Tensor):
        """Return (mu_0_lo, mu_0_hi, mu_1_lo, mu_1_hi) on the *normalized* scale.

        Raises if the network was built without quantile heads.
        """
        if self.head_0_lo is None:
            raise RuntimeError("DualHeadNet was built without quantile heads "
                               "(set DualHeadConfig.quantile_heads=True)")
        phi = self.encoder(x)
        return (self.head_0_lo(phi).squeeze(-1),
                self.head_0_hi(phi).squeeze(-1),
                self.head_1_lo(phi).squeeze(-1),
                self.head_1_hi(phi).squeeze(-1))


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------


def factual_mse_loss(
    model: DualHeadNet, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor,
) -> torch.Tensor:
    """T-Learner-style factual MSE: each sample contributes via its observed head."""
    mu_0, mu_1 = model(x)
    pred = torch.where(t == 0, mu_0, mu_1)
    return torch.mean((pred - y) ** 2)


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, q: float) -> torch.Tensor:
    """Pinball (quantile) loss at level q in (0, 1). Mean over batch."""
    diff = target - pred
    return torch.mean(torch.maximum(q * diff, (q - 1.0) * diff))


def factual_pinball_loss(
    model: DualHeadNet, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor,
    q_lo: float, q_hi: float,
) -> torch.Tensor:
    """Sum of lo/hi pinball losses on the observed-treatment quantile heads."""
    mu_0_lo, mu_0_hi, mu_1_lo, mu_1_hi = model.forward_quantiles(x)
    pred_lo = torch.where(t == 0, mu_0_lo, mu_1_lo)
    pred_hi = torch.where(t == 0, mu_0_hi, mu_1_hi)
    return pinball_loss(pred_lo, y, q_lo) + pinball_loss(pred_hi, y, q_hi)


def combined_loss(
    model: DualHeadNet, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor,
    pinball_weight: float, q_lo: float, q_hi: float,
) -> torch.Tensor:
    """MSE on mean heads + (optional) pinball loss on quantile heads."""
    loss = factual_mse_loss(model, x, t, y)
    if model.head_0_lo is not None and pinball_weight > 0:
        loss = loss + pinball_weight * factual_pinball_loss(
            model, x, t, y, q_lo, q_hi,
        )
    return loss


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def _to_tensor(arr: np.ndarray, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device=device, dtype=dtype)


def _train_val_split(N: int, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_val = max(1, int(N * val_frac))
    return idx[n_val:], idx[:n_val]


def _balanced_batch_sampler(
    t: np.ndarray, batch_size: int, rng: np.random.Generator,
) -> np.ndarray:
    """Return a single mini-batch index array with ~equal T=0 / T=1 share."""
    idx_0 = np.where(t == 0)[0]
    idx_1 = np.where(t == 1)[0]
    n_each = batch_size // 2
    sel_0 = rng.choice(idx_0, size=min(n_each, len(idx_0)), replace=len(idx_0) < n_each)
    sel_1 = rng.choice(idx_1, size=min(n_each, len(idx_1)), replace=len(idx_1) < n_each)
    sel = np.concatenate([sel_0, sel_1])
    rng.shuffle(sel)
    return sel


def train_dual_head(
    X: np.ndarray, T: np.ndarray, Y: np.ndarray,
    arch_cfg: DualHeadConfig, train_cfg: TrainConfig,
    device: Optional[torch.device] = None,
) -> Tuple[DualHeadNet, dict]:
    """Train DualHeadNet with factual MSE + early stopping.

    Returns
    -------
    model : trained DualHeadNet (best-val state restored)
    info  : dict with `history`, `best_val`, `y_mean`, `y_std`, `epochs_used`,
            `device`, `train_idx`, `val_idx`
    """
    device = device or pick_device()
    if T.dtype != np.int64:
        T = T.astype(np.int64)

    # Determinism control: seed torch BEFORE model construction so weights init
    # is reproducible. MPS may still introduce minor non-determinism in some ops;
    # callers needing strict bitwise reproducibility should run on CPU.
    torch.manual_seed(train_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_cfg.seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(train_cfg.seed)

    # Standardize Y for stable training
    y_mean = float(Y.mean())
    y_std = float(Y.std()) + 1e-6
    Y_norm = (Y - y_mean) / y_std

    tr_idx, va_idx = _train_val_split(X.shape[0], train_cfg.val_frac, train_cfg.seed)

    X_tr = _to_tensor(X[tr_idx], device)
    T_tr = _to_tensor(T[tr_idx], device, dtype=torch.long)
    Y_tr = _to_tensor(Y_norm[tr_idx], device)
    X_va = _to_tensor(X[va_idx], device)
    T_va = _to_tensor(T[va_idx], device, dtype=torch.long)
    Y_va = _to_tensor(Y_norm[va_idx], device)

    model = DualHeadNet(arch_cfg).to(device)
    optim = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay,
    )
    rng = np.random.default_rng(train_cfg.seed)

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad = 0
    history: List[dict] = []

    n_tr = X_tr.shape[0]
    T_tr_np = T[tr_idx]

    pin_w = arch_cfg.pinball_weight if arch_cfg.quantile_heads else 0.0
    q_lo, q_hi = arch_cfg.quantile_lo, arch_cfg.quantile_hi

    def _loss(xb, tb, yb):
        return combined_loss(model, xb, tb, yb, pin_w, q_lo, q_hi)

    for epoch in range(train_cfg.max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        if train_cfg.treatment_balance:
            n_steps = max(1, n_tr // train_cfg.batch_size)
            for _ in range(n_steps):
                sel = _balanced_batch_sampler(T_tr_np, train_cfg.batch_size, rng)
                xb, tb, yb = X_tr[sel], T_tr[sel], Y_tr[sel]
                optim.zero_grad()
                loss = _loss(xb, tb, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                optim.step()
                epoch_loss += float(loss.item())
                n_batches += 1
        else:
            perm = torch.from_numpy(rng.permutation(n_tr)).to(device)
            for s in range(0, n_tr, train_cfg.batch_size):
                sel = perm[s : s + train_cfg.batch_size]
                xb, tb, yb = X_tr[sel], T_tr[sel], Y_tr[sel]
                optim.zero_grad()
                loss = _loss(xb, tb, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                optim.step()
                epoch_loss += float(loss.item())
                n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            val_loss = float(_loss(X_va, T_va, Y_va).item())
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss})

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= train_cfg.patience:
                break

        if train_cfg.verbose and (epoch % 10 == 0 or bad == 0):
            print(f"  epoch {epoch:3d} | train={train_loss:.4f}  val={val_loss:.4f}"
                  f"{'  *' if bad == 0 else ''}")

    model.load_state_dict(best_state)
    return model, {
        "history": history,
        "best_val": best_val,
        "y_mean": y_mean,
        "y_std": y_std,
        "epochs_used": len(history),
        "device": str(device),
        "train_idx": tr_idx,
        "val_idx": va_idx,
    }


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def predict_potentials(
    model: DualHeadNet, X: np.ndarray, info: dict,
    device: Optional[torch.device] = None, batch_size: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict (mu_0, mu_1) on the original Y scale."""
    device = device or pick_device()
    model = model.to(device).eval()
    y_mean, y_std = info["y_mean"], info["y_std"]

    out0 = np.empty(X.shape[0], dtype=np.float64)
    out1 = np.empty(X.shape[0], dtype=np.float64)
    with torch.no_grad():
        for s in range(0, X.shape[0], batch_size):
            xb = _to_tensor(X[s : s + batch_size], device)
            mu_0, mu_1 = model(xb)
            out0[s : s + batch_size] = mu_0.cpu().numpy().astype(np.float64) * y_std + y_mean
            out1[s : s + batch_size] = mu_1.cpu().numpy().astype(np.float64) * y_std + y_mean
    return out0, out1


def predict_ite(
    model: DualHeadNet, X: np.ndarray, info: dict,
    device: Optional[torch.device] = None, batch_size: int = 1024,
) -> np.ndarray:
    """Predict point-estimate ITE = mu_1(x) - mu_0(x)."""
    mu_0, mu_1 = predict_potentials(model, X, info, device, batch_size)
    return mu_1 - mu_0


def predict_quantiles(
    model: DualHeadNet, X: np.ndarray, info: dict,
    device: Optional[torch.device] = None, batch_size: int = 1024,
) -> Dict[str, np.ndarray]:
    """Predict per-arm quantile heads on the original Y scale.

    Returns a dict with mu_0_lo, mu_0_hi, mu_1_lo, mu_1_hi, plus the derived
    *conservative* ITE band:
        tau_low_raw  = mu_1_lo - mu_0_hi   (worst-case ITE: low treated, high control)
        tau_high_raw = mu_1_hi - mu_0_lo   (best-case ITE)
    These are the "raw" quantile bounds; the CQR layer then conformalizes them
    by adding/subtracting q to guarantee marginal coverage.
    """
    if model.head_0_lo is None:
        raise RuntimeError(
            "predict_quantiles requires the model to be built with quantile_heads=True"
        )
    device = device or pick_device()
    model = model.to(device).eval()
    y_mean, y_std = info["y_mean"], info["y_std"]

    out = {k: np.empty(X.shape[0], dtype=np.float64)
           for k in ("mu_0_lo", "mu_0_hi", "mu_1_lo", "mu_1_hi")}
    with torch.no_grad():
        for s in range(0, X.shape[0], batch_size):
            xb = _to_tensor(X[s : s + batch_size], device)
            m0lo, m0hi, m1lo, m1hi = model.forward_quantiles(xb)
            out["mu_0_lo"][s:s + batch_size] = (
                m0lo.cpu().numpy().astype(np.float64) * y_std + y_mean)
            out["mu_0_hi"][s:s + batch_size] = (
                m0hi.cpu().numpy().astype(np.float64) * y_std + y_mean)
            out["mu_1_lo"][s:s + batch_size] = (
                m1lo.cpu().numpy().astype(np.float64) * y_std + y_mean)
            out["mu_1_hi"][s:s + batch_size] = (
                m1hi.cpu().numpy().astype(np.float64) * y_std + y_mean)
    out["tau_low_raw"] = out["mu_1_lo"] - out["mu_0_hi"]
    out["tau_high_raw"] = out["mu_1_hi"] - out["mu_0_lo"]
    return out


def compute_residuals(
    model: DualHeadNet, X: np.ndarray, T: np.ndarray, Y: np.ndarray,
    info: dict, device: Optional[torch.device] = None, batch_size: int = 1024,
) -> Dict[str, np.ndarray]:
    """Compute factual residuals on a held-out set, used by the Conformal layer.

    Returns separate residuals for T=0 and T=1, plus the absolute residuals,
    which the W1-5 Conformal Adverse-Effect Detection module will consume.
    """
    mu_0, mu_1 = predict_potentials(model, X, info, device, batch_size)
    pred = np.where(T == 0, mu_0, mu_1)
    res = Y - pred
    return {
        "residuals": res,
        "abs_residuals": np.abs(res),
        "residuals_t0": res[T == 0],
        "residuals_t1": res[T == 1],
        "mu_0": mu_0,
        "mu_1": mu_1,
        "tau_hat": mu_1 - mu_0,
    }
