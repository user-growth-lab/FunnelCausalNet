"""FunnelCausalNet: funnel-aware multi-task causal network for multi-tier uplift.

Aligned with the manuscript §3 (method outline) and appendix theory notes:

    phi(x)             = shared MLP encoder
    mu_c^(t)(x)        = per-arm CVR head (sigmoid over per-arm logits)
    mu_v^(t)(x)        = per-arm OV head, predicts log E[OV | x, t, conv=1]
                         (output is in *normalized log1p(GMV)* space; we
                         denormalize and use the LogNormal mean correction
                         exp(z + 0.5 sigma^2) - 1 to get the OV expectation)
    mu_g^(t)(x)        = mu_c^(t)(x) * mu_v^(t)(x)            (funnel hard constraint)
    mu_g_anchor^(t)(x) = direct head fit to log1p(GMV) for L_consistency
                         (training-only, prevents mu_c / mu_v drift)

Loss components (all minimised jointly per §I 训练策略):
    L_total = L_c + alpha * L_v + beta * L_consistency + gamma * L_mono

    L_c     = sum_t E_{i:T_i=t} BCE(sigmoid(logit_c[i,t]), y_cvr[i])
    L_v     = sum_t E_{i:T_i=t, y_cvr[i]=1} (mu_v_norm[i,t] - log1p(y_gmv[i])_norm)^2
    L_cons  = E_x ((mu_g_anchor_norm[i,T_i] - log1p(mu_c[i,T_i]*mu_v[i,T_i])_norm)^2)
    L_mono  = sum_t E_x relu(mu_g[i, t-1] - mu_g[i, t])^2

Architectural choice (Q1=C, Q2 implied, Q3=A from 用户 2026-05-06 confirm):
- Q1=C: shared MLP encoder + per-arm last linear layer (`per-arm tower`).
        Implemented compactly as a single _mlp with output dim = num_arms.
- Q3=A: end-to-end joint optimization of all loss terms.

Training data convention:
    Inputs:  X (N, d) float32, T (N,) int [0..K], Y_cvr (N,) int {0,1},
             Y_gmv (N,) float32 (raw RMB scale, ≥ 0).
    Internally we standardize log1p(Y_gmv) for stable training.

Output bundle returned by predict_potentials():
    mu_c_full     : (N, K+1) sigmoid CVR predictions
    mu_v_full     : (N, K+1) E[OV | conv=1, x, t] in raw RMB scale (LogNormal mean)
    mu_g_full     : (N, K+1) GMV expectations from funnel composition
    tau_c         : (N, K+1) τ_CVR vs control arm
    tau_g         : (N, K+1) τ_GMV vs control arm
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
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FunnelArchConfig:
    """Architecture hyper-parameters for FunnelCausalNet."""

    d_in: int = 12                        # 12-d benchmark features
    num_arms: int = 8                     # K+1 (with control)
    rep_hidden: List[int] = field(default_factory=lambda: [128, 64])
    rep_dim: int = 32
    head_hidden: List[int] = field(default_factory=lambda: [32])
    dropout: float = 0.1
    activation: str = "elu"
    use_anchor: bool = True               # whether to enable L_consistency anchor head
    learn_log_sigma: bool = True          # learnable LogNormal noise std for OV
    log_sigma_init: float = 0.8           # initial value (matches Criteo-MT7)

    # ----- Quantile heads for CQR (Innovation II joint Conformal CATE) -----
    # When enabled, four extra per-arm heads predict X-conditional quantile
    # bounds for both CVR (logit-space) and OV (log1p-space). The CQR layer
    # in joint_conformal_cate.py then conformalizes them under split-CP +
    # Bonferroni union bound to give the joint coverage in §II.5 定理 1.
    quantile_heads: bool = False
    quantile_lo: float = 0.05             # default α=0.1 → q_lo=0.05, q_hi=0.95
    quantile_hi: float = 0.95
    pinball_weight: float = 0.5           # weight of pinball loss vs other terms


@dataclass
class FunnelLossConfig:
    """Loss weighting and training-only auxiliary settings."""

    alpha: float = 1.0                    # weight on L_v
    beta: float = 0.5                     # weight on L_consistency
    gamma: float = 0.1                    # weight on L_mono
    enable_consistency: bool = True
    enable_monotonic: bool = True
    # E2 ablation 模式：
    #   'hard'  → mu_g = mu_c * mu_v (funnel composition)；anchor 学 decomp
    #            (= log1p(mu_c*mu_v))，L_consist 当作正则；BCE + MSE
    #   'soft'  → anchor 直接学 log1p(y_g)，L_consist = (anchor - decomp)^2 作软正则
    #   'direct'→ 仅 anchor 学 log1p(y_g)，conv/val 头不参与 GMV 预测路径
    #   'ziln'  → VALOR-style：L_c 用 Focal-BCE，L_v 用 LogNormal NLL（带 sigma 项），
    #            预测路径同 'hard' (mu_g = mu_c × mu_v)。对应 §I.5 定理 2 likelihood
    #            等价性的实证锚点。
    funnel_mode: str = "hard"             # 'hard' | 'soft' | 'direct' | 'ziln'
    # Focal loss params (only used when funnel_mode='ziln')
    focal_alpha: float = 0.25             # class balance weight in BCE
    focal_gamma: float = 2.0              # focusing parameter
    # LogNormal NLL params (only used when funnel_mode='ziln')
    ziln_min_sigma: float = 0.1           # numerical floor for sigma


@dataclass
class FunnelTrainConfig:
    """Training hyper-parameters."""

    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    max_epochs: int = 200
    patience: int = 15
    val_frac: float = 0.2
    seed: int = 0
    verbose: bool = False
    grad_clip: float = 5.0
    arm_balance: bool = True              # per-batch resample to give all arms ≈ same count


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
# Network module
# ---------------------------------------------------------------------------


class FunnelCausalNet(nn.Module):
    """Encoder + per-arm conv head + per-arm OV head + (optional) anchor head.

    Each head ends in a Linear(.., num_arms) so the per-arm "tower" is the
    last linear layer, with the rest of the head shared across arms.
    """

    def __init__(self, cfg: FunnelArchConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = _mlp(cfg.d_in, cfg.rep_hidden, cfg.rep_dim,
                            activation=cfg.activation, dropout=cfg.dropout)
        self.conv_head = _mlp(cfg.rep_dim, cfg.head_hidden, cfg.num_arms,
                              activation=cfg.activation, dropout=cfg.dropout)
        self.val_head = _mlp(cfg.rep_dim, cfg.head_hidden, cfg.num_arms,
                             activation=cfg.activation, dropout=cfg.dropout)
        if cfg.use_anchor:
            self.anchor_head = _mlp(cfg.rep_dim, cfg.head_hidden, cfg.num_arms,
                                    activation=cfg.activation, dropout=cfg.dropout)
        else:
            self.anchor_head = None

        if cfg.quantile_heads:
            mk = lambda: _mlp(cfg.rep_dim, cfg.head_hidden, cfg.num_arms,
                              activation=cfg.activation, dropout=cfg.dropout)
            self.conv_head_lo = mk()
            self.conv_head_hi = mk()
            self.val_head_lo = mk()
            self.val_head_hi = mk()
        else:
            self.conv_head_lo = None
            self.conv_head_hi = None
            self.val_head_lo = None
            self.val_head_hi = None

        if cfg.learn_log_sigma:
            self.log_sigma = nn.Parameter(torch.tensor(cfg.log_sigma_init,
                                                       dtype=torch.float32))
        else:
            self.register_buffer("log_sigma",
                                 torch.tensor(cfg.log_sigma_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        phi = self.encoder(x)
        conv_logits = self.conv_head(phi)              # (N, K+1)
        val_log_norm = self.val_head(phi)              # (N, K+1)
        out = {
            "conv_logits": conv_logits,
            "val_log_norm": val_log_norm,
            "phi": phi,
        }
        if self.anchor_head is not None:
            out["g_anchor_norm"] = self.anchor_head(phi)   # (N, K+1)
        if self.conv_head_lo is not None:
            out["conv_logits_lo"] = self.conv_head_lo(phi)     # (N, K+1)
            out["conv_logits_hi"] = self.conv_head_hi(phi)     # (N, K+1)
            out["val_log_norm_lo"] = self.val_head_lo(phi)     # (N, K+1)
            out["val_log_norm_hi"] = self.val_head_hi(phi)     # (N, K+1)
        return out


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------


def _denorm(z_norm: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return z_norm * std + mean


def funnel_compose_log1p_norm(
    conv_logits: torch.Tensor, val_log_norm: torch.Tensor,
    log_sigma: torch.Tensor, log_g_mean: float, log_g_std: float,
) -> torch.Tensor:
    """Compute log1p(mu_g) in *normalized* scale, where mu_g = mu_c * E[OV].

    val_log_norm is the standardized log1p(GMV|conv=1) prediction. To recover
    E[OV] we denormalize, treat the LogNormal mean correction (+0.5 sigma^2),
    and convert log1p back to original scale via expm1.
    """
    mu_c = torch.sigmoid(conv_logits)                               # (N, K+1)
    val_log = _denorm(val_log_norm, log_g_mean, log_g_std)          # (N, K+1)
    expected_ov = torch.expm1(val_log + 0.5 * log_sigma.pow(2))     # E[OV]
    expected_ov = torch.clamp(expected_ov, min=0.0)
    mu_g = mu_c * expected_ov                                        # (N, K+1)
    log1p_mu_g = torch.log1p(mu_g)
    return (log1p_mu_g - log_g_mean) / max(log_g_std, 1e-6)


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, q: float) -> torch.Tensor:
    """Pinball (quantile) loss at level q ∈ (0, 1). Mean reduction."""
    diff = target - pred
    return torch.mean(torch.maximum(q * diff, (q - 1.0) * diff))


def focal_bce_loss(logits: torch.Tensor, y: torch.Tensor,
                   alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Lin et al. ICCV 2017 Focal Loss for binary classification.

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
        其中 p_t = sigmoid(logits) if y=1 else 1 - sigmoid(logits),
            α_t = α          if y=1 else 1 - α
    在 zero-inflated CTR 下（y=1 极少），focal-BCE 比标准 BCE 更关注 hard positives。
    """
    p = torch.sigmoid(logits)
    p = p.clamp(min=1e-7, max=1.0 - 1e-7)
    y_pos = (y > 0.5).float()
    p_t = y_pos * p + (1.0 - y_pos) * (1.0 - p)
    alpha_t = y_pos * alpha + (1.0 - y_pos) * (1.0 - alpha)
    return torch.mean(-alpha_t * (1.0 - p_t).pow(gamma) * torch.log(p_t))


def lognormal_nll_log1p_norm(val_log_norm_pred: torch.Tensor,
                              y_g_log1p_norm: torch.Tensor,
                              log_sigma: torch.Tensor,
                              log_g_std: float,
                              min_sigma: float = 0.1) -> torch.Tensor:
    """LogNormal NLL on log1p-normalized GMV space. Used in funnel_mode='ziln'.

    在 normalized log1p 空间，sigma_norm = log_sigma.exp() / log_g_std。
    NLL = 0.5 · log(2π · σ_norm²) + (μ̂ - y)² / (2 · σ_norm²)
    （省略与参数无关的常数；最终保留 sigma 调整项以让 sigma 参与梯度。）
    """
    sigma_eff = torch.clamp(torch.exp(log_sigma), min=min_sigma)
    sigma_norm = sigma_eff / max(log_g_std, 1e-6)
    sigma_norm_sq = sigma_norm.pow(2)
    diff_sq = (val_log_norm_pred - y_g_log1p_norm).pow(2)
    return torch.mean(0.5 * torch.log(2.0 * float(np.pi) * sigma_norm_sq)
                      + diff_sq / (2.0 * sigma_norm_sq))


def compute_losses(
    out: Dict[str, torch.Tensor],
    t: torch.Tensor,                # (N,) long
    y_c: torch.Tensor,              # (N,) float
    y_g_log1p_norm: torch.Tensor,   # (N,) float, normalized log1p(GMV)
    loss_cfg: FunnelLossConfig,
    log_sigma: torch.Tensor,
    log_g_mean: float, log_g_std: float,
    arch_cfg: Optional["FunnelArchConfig"] = None,
) -> Dict[str, torch.Tensor]:
    """Return dict of individual loss terms + 'total' (already weighted)."""
    N = t.shape[0]
    arange_N = torch.arange(N, device=t.device)

    conv_logits = out["conv_logits"]              # (N, K+1)
    val_log_norm = out["val_log_norm"]            # (N, K+1)

    # ---- L_c: per-sample factual BCE on observed arm
    #         hard / soft / direct → 标准 BCE
    #         ziln                  → Focal-BCE (zero-inflated 友好)
    obs_logits = conv_logits[arange_N, t]
    if loss_cfg.funnel_mode == "ziln":
        focal_alpha = (loss_cfg.focal_alpha if hasattr(loss_cfg, "focal_alpha")
                       else 0.25)
        focal_gamma = (loss_cfg.focal_gamma if hasattr(loss_cfg, "focal_gamma")
                       else 2.0)
        L_c = focal_bce_loss(obs_logits, y_c, alpha=focal_alpha, gamma=focal_gamma)
    else:
        L_c = nn.functional.binary_cross_entropy_with_logits(obs_logits, y_c)

    # ---- L_v: per-sample factual GMV value loss on conv=1 subset
    #         hard / soft / direct → MSE on log1p-normalized
    #         ziln                  → LogNormal NLL on log1p-normalized
    obs_val_norm = val_log_norm[arange_N, t]
    mask = (y_c > 0.5)
    if mask.any():
        if loss_cfg.funnel_mode == "ziln":
            min_sigma = (loss_cfg.ziln_min_sigma
                         if hasattr(loss_cfg, "ziln_min_sigma") else 0.1)
            L_v = lognormal_nll_log1p_norm(
                obs_val_norm[mask], y_g_log1p_norm[mask],
                log_sigma, log_g_std, min_sigma=min_sigma,
            )
        else:
            diff_v = obs_val_norm[mask] - y_g_log1p_norm[mask]
            L_v = torch.mean(diff_v.pow(2))
    else:
        L_v = torch.zeros((), device=t.device)

    # ---- L_consistency: 三种 funnel_mode 语义不同
    #   hard:  anchor 学 decomp，作为正则（双向引导）
    #   soft:  anchor 直接学 y_g (factual arm)，再加正则 (anchor - decomp)^2
    #   direct:anchor 仅学 y_g，不加 decomp 项（=纯直接回归）
    if loss_cfg.enable_consistency and "g_anchor_norm" in out:
        g_anchor_norm = out["g_anchor_norm"]
        if loss_cfg.funnel_mode == "hard":
            decomp_norm = funnel_compose_log1p_norm(
                conv_logits, val_log_norm, log_sigma, log_g_mean, log_g_std,
            )
            L_consist = torch.mean((g_anchor_norm - decomp_norm).pow(2))
        elif loss_cfg.funnel_mode == "soft":
            obs_anchor = g_anchor_norm[arange_N, t]
            L_anchor_data = torch.mean((obs_anchor - y_g_log1p_norm).pow(2))
            decomp_norm = funnel_compose_log1p_norm(
                conv_logits, val_log_norm, log_sigma, log_g_mean, log_g_std,
            )
            L_anchor_decomp = torch.mean((g_anchor_norm - decomp_norm).pow(2))
            L_consist = L_anchor_data + 0.5 * L_anchor_decomp
        elif loss_cfg.funnel_mode == "direct":
            obs_anchor = g_anchor_norm[arange_N, t]
            L_consist = torch.mean((obs_anchor - y_g_log1p_norm).pow(2))
        elif loss_cfg.funnel_mode == "ziln":
            L_consist = torch.zeros((), device=t.device)
        else:
            raise ValueError(f"unknown funnel_mode={loss_cfg.funnel_mode!r}")
    else:
        L_consist = torch.zeros((), device=t.device)

    # ---- L_mono: per-user, mu_g monotonic in arm index (in normalized log1p space) ----
    if loss_cfg.enable_monotonic:
        decomp_norm_full = funnel_compose_log1p_norm(
            conv_logits, val_log_norm, log_sigma, log_g_mean, log_g_std,
        )
        diffs = decomp_norm_full[:, :-1] - decomp_norm_full[:, 1:]   # ≤ 0 expected
        L_mono = torch.mean(torch.relu(diffs).pow(2))
    else:
        L_mono = torch.zeros((), device=t.device)

    # ---- L_pinball: per-arm pinball loss on conv (logit-space) and val
    # (log1p-normalized space). Trained on observed arm only (T-Learner style).
    L_pinball = torch.zeros((), device=t.device)
    if arch_cfg is not None and arch_cfg.quantile_heads and "conv_logits_lo" in out:
        # CVR head: target = y_c ∈ {0, 1}; pinball on raw probability via sigmoid.
        # We use the *probability* space (sigmoid of logits) as the prediction
        # so the resulting [q_lo, q_hi] are valid CVR probability bounds.
        p_lo_c = torch.sigmoid(out["conv_logits_lo"][arange_N, t])
        p_hi_c = torch.sigmoid(out["conv_logits_hi"][arange_N, t])
        L_pinball = (L_pinball
                     + pinball_loss(p_lo_c, y_c, arch_cfg.quantile_lo)
                     + pinball_loss(p_hi_c, y_c, arch_cfg.quantile_hi))
        # OV head: target = log1p(GMV) normalized, mask conv=1 only.
        v_lo = out["val_log_norm_lo"][arange_N, t]
        v_hi = out["val_log_norm_hi"][arange_N, t]
        if mask.any():
            L_pinball = (L_pinball
                         + pinball_loss(v_lo[mask], y_g_log1p_norm[mask], arch_cfg.quantile_lo)
                         + pinball_loss(v_hi[mask], y_g_log1p_norm[mask], arch_cfg.quantile_hi))
        L_pinball = L_pinball * arch_cfg.pinball_weight

    L_total = (
        L_c
        + loss_cfg.alpha * L_v
        + loss_cfg.beta * L_consist
        + loss_cfg.gamma * L_mono
        + L_pinball
    )
    return {
        "total": L_total,
        "L_c": L_c.detach(),
        "L_v": L_v.detach(),
        "L_consist": L_consist.detach(),
        "L_mono": L_mono.detach(),
        "L_pinball": L_pinball.detach() if isinstance(L_pinball, torch.Tensor) else torch.zeros(()),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _to_tensor(arr: np.ndarray, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device=device, dtype=dtype)


def _train_val_split(N: int, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_val = max(1, int(N * val_frac))
    return idx[n_val:], idx[:n_val]


def _arm_balanced_batch(
    t_np: np.ndarray, batch_size: int, num_arms: int, rng: np.random.Generator,
) -> np.ndarray:
    per_arm = max(1, batch_size // num_arms)
    parts = []
    for arm in range(num_arms):
        idx = np.where(t_np == arm)[0]
        if len(idx) == 0:
            continue
        sel = rng.choice(idx, size=per_arm, replace=len(idx) < per_arm)
        parts.append(sel)
    out = np.concatenate(parts)
    rng.shuffle(out)
    return out


def train_funnel_net(
    X: np.ndarray, T: np.ndarray, Y_cvr: np.ndarray, Y_gmv: np.ndarray,
    arch_cfg: FunnelArchConfig, loss_cfg: FunnelLossConfig,
    train_cfg: FunnelTrainConfig,
    device: Optional[torch.device] = None,
) -> Tuple[FunnelCausalNet, dict]:
    """Train FunnelCausalNet end-to-end with early stopping."""
    device = device or pick_device()
    if T.dtype != np.int64:
        T = T.astype(np.int64)

    torch.manual_seed(train_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_cfg.seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(train_cfg.seed)

    # Standardize log1p(GMV) for stable training
    Y_g_log1p = np.log1p(np.maximum(Y_gmv, 0.0)).astype(np.float64)
    log_g_mean = float(Y_g_log1p.mean())
    log_g_std = float(Y_g_log1p.std()) + 1e-6
    Y_g_log1p_norm = (Y_g_log1p - log_g_mean) / log_g_std

    tr_idx, va_idx = _train_val_split(X.shape[0], train_cfg.val_frac, train_cfg.seed)

    X_tr = _to_tensor(X[tr_idx], device)
    T_tr = _to_tensor(T[tr_idx], device, dtype=torch.long)
    Yc_tr = _to_tensor(Y_cvr[tr_idx].astype(np.float32), device)
    Yg_tr_norm = _to_tensor(Y_g_log1p_norm[tr_idx], device)

    X_va = _to_tensor(X[va_idx], device)
    T_va = _to_tensor(T[va_idx], device, dtype=torch.long)
    Yc_va = _to_tensor(Y_cvr[va_idx].astype(np.float32), device)
    Yg_va_norm = _to_tensor(Y_g_log1p_norm[va_idx], device)

    model = FunnelCausalNet(arch_cfg).to(device)

    # ZILN mode: 直接用 conv=1 子集的 log1p(GMV) 残差 std 初始化 log_sigma 并 freeze。
    # 否则 ZILN NLL 中 sigma 项与 (μ̂-y)²/(2σ²) 项的梯度耦合会让 sigma 偏离真值，
    # 进而通过 expm1(val_log + 0.5σ²) 校正项放大 mu_v 偏差（已在 paper grid 中
    # 实测 mu_g 高估 2-5 倍）。
    if loss_cfg.funnel_mode == "ziln":
        mask_pos_tr = (Y_cvr[tr_idx] > 0.5)
        if mask_pos_tr.any():
            sigma_raw = float(Y_g_log1p[tr_idx][mask_pos_tr].std()) + 1e-6
        else:
            sigma_raw = arch_cfg.log_sigma_init
        sigma_raw = max(sigma_raw, 0.1)
        with torch.no_grad():
            if isinstance(model.log_sigma, torch.nn.Parameter):
                model.log_sigma.copy_(torch.tensor(float(np.log(sigma_raw)),
                                                   device=device))
                model.log_sigma.requires_grad_(False)
            else:
                model.log_sigma = model.log_sigma.detach()
                model.log_sigma.fill_(float(np.log(sigma_raw)))

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg.lr, weight_decay=train_cfg.weight_decay,
    )
    rng = np.random.default_rng(train_cfg.seed)

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad = 0
    history: List[dict] = []
    n_tr = X_tr.shape[0]
    T_tr_np = T[tr_idx]

    def _step(xb, tb, ycb, ygb):
        out = model(xb)
        return compute_losses(
            out, tb, ycb, ygb, loss_cfg, model.log_sigma, log_g_mean, log_g_std,
            arch_cfg=arch_cfg,
        )

    for epoch in range(train_cfg.max_epochs):
        model.train()
        epoch_acc = {"total": 0.0, "L_c": 0.0, "L_v": 0.0, "L_consist": 0.0, "L_mono": 0.0,
                     "L_pinball": 0.0}
        n_batches = 0

        if train_cfg.arm_balance:
            n_steps = max(1, n_tr // train_cfg.batch_size)
            for _ in range(n_steps):
                sel = _arm_balanced_batch(T_tr_np, train_cfg.batch_size,
                                          arch_cfg.num_arms, rng)
                xb = X_tr[sel]; tb = T_tr[sel]; ycb = Yc_tr[sel]; ygb = Yg_tr_norm[sel]
                optim.zero_grad()
                losses = _step(xb, tb, ycb, ygb)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                optim.step()
                for k in epoch_acc:
                    epoch_acc[k] += float(losses[k].item())
                n_batches += 1
        else:
            perm = rng.permutation(n_tr)
            for s in range(0, n_tr, train_cfg.batch_size):
                sel = perm[s:s + train_cfg.batch_size]
                xb = X_tr[sel]; tb = T_tr[sel]; ycb = Yc_tr[sel]; ygb = Yg_tr_norm[sel]
                optim.zero_grad()
                losses = _step(xb, tb, ycb, ygb)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                optim.step()
                for k in epoch_acc:
                    epoch_acc[k] += float(losses[k].item())
                n_batches += 1

        for k in epoch_acc:
            epoch_acc[k] /= max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            val_losses = _step(X_va, T_va, Yc_va, Yg_va_norm)
            val_total = float(val_losses["total"].item())
        history.append({"epoch": epoch, **epoch_acc, "val_total": val_total})

        if val_total < best_val - 1e-5:
            best_val = val_total
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= train_cfg.patience:
                break

        if train_cfg.verbose and (epoch % 5 == 0 or bad == 0):
            print(f"  epoch {epoch:3d} | "
                  f"train_total={epoch_acc['total']:.4f} "
                  f"L_c={epoch_acc['L_c']:.4f} L_v={epoch_acc['L_v']:.4f} "
                  f"L_cons={epoch_acc['L_consist']:.4f} L_mono={epoch_acc['L_mono']:.4f} | "
                  f"val={val_total:.4f}{'  *' if bad == 0 else ''}")

    model.load_state_dict(best_state)
    return model, {
        "history": history,
        "best_val": best_val,
        "log_g_mean": log_g_mean,
        "log_g_std": log_g_std,
        "epochs_used": len(history),
        "device": str(device),
        "train_idx": tr_idx,
        "val_idx": va_idx,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_potentials(
    model: FunnelCausalNet, X: np.ndarray, info: dict,
    device: Optional[torch.device] = None, batch_size: int = 4096,
    funnel_mode: str = "hard",
) -> Dict[str, np.ndarray]:
    """Predict per-arm (mu_c, mu_v, mu_g) and (tau_c, tau_g) on raw scales.

    funnel_mode='hard': mu_g = mu_c * mu_v   (funnel composition)
    funnel_mode='soft' or 'direct': mu_g 直接读 anchor_head 反归一化后的 expm1 值
        （在 'soft' 模式下 anchor 已经被训练成拟合 log1p(y_g)）
    """
    device = device or pick_device()
    model = model.to(device).eval()
    log_g_mean, log_g_std = info["log_g_mean"], info["log_g_std"]
    K = model.cfg.num_arms

    N = X.shape[0]
    mu_c_full = np.empty((N, K), dtype=np.float64)
    mu_v_full = np.empty((N, K), dtype=np.float64)
    mu_g_full = np.empty((N, K), dtype=np.float64)

    with torch.no_grad():
        log_sigma_val = float(model.log_sigma.detach().cpu().numpy())
        sigma_corr = 0.5 * (log_sigma_val ** 2)
        for s in range(0, N, batch_size):
            xb = _to_tensor(X[s:s + batch_size], device)
            out = model(xb)
            mu_c = torch.sigmoid(out["conv_logits"]).cpu().numpy().astype(np.float64)
            val_log_norm = out["val_log_norm"].cpu().numpy().astype(np.float64)
            val_log = val_log_norm * log_g_std + log_g_mean
            expected_ov = np.expm1(val_log + sigma_corr)
            expected_ov = np.clip(expected_ov, 0.0, None)
            mu_c_full[s:s + batch_size] = mu_c
            mu_v_full[s:s + batch_size] = expected_ov

            if funnel_mode in ("hard", "ziln"):
                mu_g_full[s:s + batch_size] = mu_c * expected_ov
            elif funnel_mode in ("soft", "direct"):
                if "g_anchor_norm" not in out:
                    raise RuntimeError(
                        f"funnel_mode={funnel_mode!r} requires use_anchor=True at training")
                g_log = out["g_anchor_norm"].cpu().numpy().astype(np.float64) * log_g_std + log_g_mean
                mu_g_full[s:s + batch_size] = np.clip(np.expm1(g_log), 0.0, None)
            else:
                raise ValueError(f"unknown funnel_mode={funnel_mode!r}")

    tau_c = mu_c_full - mu_c_full[:, [0]]
    tau_g = mu_g_full - mu_g_full[:, [0]]
    return {
        "mu_c_full": mu_c_full,
        "mu_v_full": mu_v_full,
        "mu_g_full": mu_g_full,
        "tau_c": tau_c,
        "tau_g": tau_g,
    }


def predict_quantile_bounds(
    model: FunnelCausalNet, X: np.ndarray, info: dict,
    device: Optional[torch.device] = None, batch_size: int = 4096,
) -> Dict[str, np.ndarray]:
    """Predict raw (pre-conformalized) quantile bounds for conv (probability)
    and val (log1p-GMV space, normalized).

    Returns a dict with:
        conv_lo, conv_hi      : (N, K+1) probabilities ∈ [0, 1]
        val_log1p_lo,
        val_log1p_hi          : (N, K+1) on raw log1p(GMV) scale (denormalized)

    The CQR layer in joint_conformal_cate.py adds a calibration offset to
    each of these bounds to guarantee marginal coverage.
    """
    if model.conv_head_lo is None:
        raise RuntimeError("model was built without quantile_heads=True; "
                           "predict_quantile_bounds requires CQR mode.")
    device = device or pick_device()
    model = model.to(device).eval()
    log_g_mean, log_g_std = info["log_g_mean"], info["log_g_std"]
    K = model.cfg.num_arms
    N = X.shape[0]

    out_dict: Dict[str, np.ndarray] = {
        "conv_lo": np.empty((N, K), dtype=np.float64),
        "conv_hi": np.empty((N, K), dtype=np.float64),
        "val_log1p_lo": np.empty((N, K), dtype=np.float64),
        "val_log1p_hi": np.empty((N, K), dtype=np.float64),
    }

    with torch.no_grad():
        for s in range(0, N, batch_size):
            xb = _to_tensor(X[s:s + batch_size], device)
            out = model(xb)
            out_dict["conv_lo"][s:s + batch_size] = (
                torch.sigmoid(out["conv_logits_lo"]).cpu().numpy().astype(np.float64))
            out_dict["conv_hi"][s:s + batch_size] = (
                torch.sigmoid(out["conv_logits_hi"]).cpu().numpy().astype(np.float64))
            v_lo_norm = out["val_log_norm_lo"].cpu().numpy().astype(np.float64)
            v_hi_norm = out["val_log_norm_hi"].cpu().numpy().astype(np.float64)
            out_dict["val_log1p_lo"][s:s + batch_size] = v_lo_norm * log_g_std + log_g_mean
            out_dict["val_log1p_hi"][s:s + batch_size] = v_hi_norm * log_g_std + log_g_mean

    return out_dict
