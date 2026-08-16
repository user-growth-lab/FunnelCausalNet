"""DragonNet baseline (Shi, Blei, Veitch — NeurIPS 2019) — 多档简化复现.

DragonNet 原文针对二值 treatment, 核心是把因果估计、outcome 预测、propensity
预测放进同一个网络, 并加 *targeted regularization* 做 doubly-robust correction:

    Phi(x)         shared encoder
    Q_t(Phi(x))    per-arm outcome head    (twin heads in 二值 NeurIPS'19 版本)
    g(Phi(x))      propensity head         (sigmoid in二值; multi-class softmax 在多档)
    epsilon        learnable scalar (targeted-reg 扰动)

Loss:
    L_outcome = sum_i w_i * (Y_i - Q_{T_i}(Phi(x_i)))^2
    L_prop    = -sum_i log g_{T_i}(Phi(x_i))                     (multi-class CE)
    L_target  = sum_i (Y_i - Q_{T_i}(Phi(x_i)) - epsilon * H_{T_i}(g, x_i))^2
                where H_{T_i}(g, x) = (T_i - g(x)) / (g(x)*(1-g(x)))   in 二值
                          一般 multi-arm: H_t(g, x) = (1[t=T_i] - g_t(x)) / g_t(x)
                                                      (clever covariate, AIPW-style)

    L_total = L_outcome + lambda_p * L_prop + lambda_targ * L_target

多档扩展 (与 Schwab et al. Perfect-Match 2020 的 multi-arm DragonNet 一致):
- propensity 改为 K+1 路 softmax cross-entropy;
- targeted reg 的 clever covariate 改为 multi-arm AIPW inverse-propensity
  weighted residual; epsilon 仍是 scalar (亦可改 per-arm vector, 但 NeurIPS'19
  附录 ablation 显示 scalar 版本足够).

API 与 funnel_causal_net / ecup_baseline / rerum_baseline 对齐。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .funnel_causal_net import (
    _arm_balanced_batch,
    _mlp,
    _to_tensor,
    _train_val_split,
    pick_device,
)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass
class DragonNetConfig:
    d_in: int = 12
    num_arms: int = 8
    rep_hidden: List[int] = field(default_factory=lambda: [128, 64])
    rep_dim: int = 32
    head_hidden: List[int] = field(default_factory=lambda: [32])
    dropout: float = 0.1
    activation: str = "elu"
    lambda_propensity: float = 1.0
    lambda_targeted: float = 1.0
    targeted_eps_clip: float = 5.0    # 防 epsilon 爆炸
    propensity_floor: float = 1e-2     # 防 1/g 爆炸
    use_conv_head: bool = True


@dataclass
class DragonNetTrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    max_epochs: int = 50
    patience: int = 10
    val_frac: float = 0.2
    seed: int = 0
    verbose: bool = False
    grad_clip: float = 5.0
    arm_balance: bool = True
    alpha_v: float = 0.3              # weight on L_v 在 conv=1 子集


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class DragonNetModel(nn.Module):
    """Shared-encoder + per-arm outcome heads + multi-class propensity head + epsilon."""

    def __init__(self, cfg: DragonNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = _mlp(cfg.d_in, cfg.rep_hidden, cfg.rep_dim,
                            activation=cfg.activation, dropout=cfg.dropout)
        self.gmv_head = _mlp(cfg.rep_dim, cfg.head_hidden, cfg.num_arms,
                             activation=cfg.activation, dropout=cfg.dropout)
        if cfg.use_conv_head:
            self.conv_head = _mlp(cfg.rep_dim, cfg.head_hidden, cfg.num_arms,
                                  activation=cfg.activation, dropout=cfg.dropout)
        else:
            self.conv_head = None
        # multi-arm propensity head (K+1 类 softmax)
        self.propensity_head = _mlp(cfg.rep_dim, cfg.head_hidden, cfg.num_arms,
                                    activation=cfg.activation, dropout=cfg.dropout)
        # learnable scalar epsilon for targeted-reg perturbation (NeurIPS'19)
        self.epsilon = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        phi = self.encoder(x)
        out: Dict[str, torch.Tensor] = {
            "phi": phi,
            "val_log_norm": self.gmv_head(phi),
            "propensity_logits": self.propensity_head(phi),
            "epsilon": self.epsilon,
        }
        if self.conv_head is not None:
            out["conv_logits"] = self.conv_head(phi)
        return out


# ---------------------------------------------------------------------------
# Train + predict
# ---------------------------------------------------------------------------


def train_dragonnet_baseline(
    X: np.ndarray, T: np.ndarray, Y_cvr: np.ndarray, Y_gmv: np.ndarray,
    arch_cfg: DragonNetConfig, train_cfg: DragonNetTrainConfig,
    device: Optional[torch.device] = None,
) -> Tuple[DragonNetModel, dict]:
    device = device or pick_device()
    if T.dtype != np.int64:
        T = T.astype(np.int64)
    torch.manual_seed(train_cfg.seed)

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

    model = DragonNetModel(arch_cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr,
                              weight_decay=train_cfg.weight_decay)
    rng = np.random.default_rng(train_cfg.seed)

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad = 0; history: List[dict] = []
    n_tr = X_tr.shape[0]; T_tr_np = T[tr_idx]

    def _loss(out, t, y_c, y_g_norm):
        N = t.shape[0]; arange = torch.arange(N, device=t.device)
        # L_outcome: GMV regression on conv=1 子集
        g_pred = out["val_log_norm"][arange, t]
        mask = (y_c > 0.5)
        if mask.any():
            L_v = torch.mean((g_pred[mask] - y_g_norm[mask]).pow(2))
        else:
            L_v = torch.zeros((), device=t.device)

        # L_prop: multi-class CE over K+1 arms
        L_prop = nn.functional.cross_entropy(out["propensity_logits"], t)

        # L_target: targeted-reg with multi-arm AIPW clever covariate
        # H_t(x) = 1 / propensity_t(x) (clipped); residual = y - Q_t - epsilon * H_t
        prop_softmax = torch.softmax(out["propensity_logits"], dim=1)
        prop_t = prop_softmax[arange, t].clamp(min=arch_cfg.propensity_floor)
        clever = 1.0 / prop_t                              # (N,)
        eps = out["epsilon"].clamp(-arch_cfg.targeted_eps_clip,
                                   arch_cfg.targeted_eps_clip)
        if mask.any():
            residual = y_g_norm[mask] - g_pred[mask] - eps * clever[mask]
            L_target = torch.mean(residual.pow(2))
        else:
            L_target = torch.zeros((), device=t.device)

        L_total = (train_cfg.alpha_v * L_v
                   + arch_cfg.lambda_propensity * L_prop
                   + arch_cfg.lambda_targeted * L_target)
        if out.get("conv_logits") is not None:
            c_logit = out["conv_logits"][arange, t]
            L_c = nn.functional.binary_cross_entropy_with_logits(c_logit, y_c)
            L_total = L_total + L_c
        else:
            L_c = torch.zeros((), device=t.device)
        return L_total, L_c, L_v, L_prop, L_target

    for epoch in range(train_cfg.max_epochs):
        model.train()
        epoch_acc = {"L": 0.0, "L_c": 0.0, "L_v": 0.0, "L_prop": 0.0, "L_target": 0.0}
        n_batches = 0
        if train_cfg.arm_balance:
            n_steps = max(1, n_tr // train_cfg.batch_size)
            batches = [_arm_balanced_batch(T_tr_np, train_cfg.batch_size,
                                           arch_cfg.num_arms, rng)
                       for _ in range(n_steps)]
        else:
            perm = rng.permutation(n_tr)
            batches = [perm[i:i + train_cfg.batch_size]
                       for i in range(0, n_tr, train_cfg.batch_size)]
        for b_idx in batches:
            xb = X_tr[b_idx]; tb = T_tr[b_idx]
            ycb = Yc_tr[b_idx]; ygb = Yg_tr_norm[b_idx]
            out = model(xb)
            L, L_c, L_v, L_prop, L_target = _loss(out, tb, ycb, ygb)
            optim.zero_grad()
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optim.step()
            epoch_acc["L"] += L.item(); epoch_acc["L_c"] += L_c.item()
            epoch_acc["L_v"] += L_v.item(); epoch_acc["L_prop"] += L_prop.item()
            epoch_acc["L_target"] += L_target.item(); n_batches += 1
        for k in epoch_acc: epoch_acc[k] /= max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            out_va = model(X_va)
            val_L, _, _, _, _ = _loss(out_va, T_va, Yc_va, Yg_va_norm)
            val_total = float(val_L.item())
        history.append({"epoch": epoch + 1, "val_total": val_total, **epoch_acc})

        if val_total < best_val - 1e-5:
            best_val = val_total
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= train_cfg.patience:
                break

    model.load_state_dict(best_state)
    return model, {
        "history": history, "best_val": best_val,
        "log_g_mean": log_g_mean, "log_g_std": log_g_std,
        "epochs_used": len(history), "device": str(device),
    }


def predict_dragonnet_baseline(
    model: DragonNetModel, X: np.ndarray, info: dict,
    device: Optional[torch.device] = None, batch_size: int = 4096,
) -> Dict[str, np.ndarray]:
    """与 funnel/ECUP/RERUM/CFRNet predict 接口对齐。

    DragonNet 推断时:
    - mu_c = sigmoid(conv_logits)       (per-arm CVR)
    - mu_v = expm1(denorm(val_log_norm + epsilon * 1/propensity))
                                         (targeted-reg 校正后的 conv=1 期望 GMV)
    - mu_g = mu_c * mu_v                (funnel composition; 与 ECUP 对齐)
    - tau_c, tau_g vs control arm 0
    """
    device = device or pick_device()
    model = model.to(device).eval()
    log_g_mean, log_g_std = info["log_g_mean"], info["log_g_std"]
    K = model.cfg.num_arms; N = X.shape[0]

    mu_c_full = np.empty((N, K), dtype=np.float64)
    mu_v_full = np.empty((N, K), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, N, batch_size):
            xb = _to_tensor(X[s:s + batch_size], device)
            out = model(xb)
            if out.get("conv_logits") is not None:
                mu_c = torch.sigmoid(out["conv_logits"]).cpu().numpy().astype(np.float64)
            else:
                mu_c = np.ones((xb.shape[0], K), dtype=np.float64)
            # targeted-reg 校正 per-arm GMV 预测
            prop = torch.softmax(out["propensity_logits"], dim=1).clamp(
                min=model.cfg.propensity_floor)
            eps = float(out["epsilon"].clamp(-model.cfg.targeted_eps_clip,
                                             model.cfg.targeted_eps_clip).item())
            corr = (eps / prop).cpu().numpy().astype(np.float64)
            val_log_norm = out["val_log_norm"].cpu().numpy().astype(np.float64) + corr
            val_log = val_log_norm * log_g_std + log_g_mean
            expected_ov = np.expm1(val_log)
            expected_ov = np.clip(expected_ov, 0.0, None)
            mu_c_full[s:s + batch_size] = mu_c
            mu_v_full[s:s + batch_size] = expected_ov
    mu_g_full = mu_c_full * mu_v_full
    tau_c = mu_c_full - mu_c_full[:, [0]]
    tau_g = mu_g_full - mu_g_full[:, [0]]
    return {"mu_c_full": mu_c_full, "mu_v_full": mu_v_full, "mu_g_full": mu_g_full,
            "tau_c": tau_c, "tau_g": tau_g}
