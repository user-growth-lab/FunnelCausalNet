"""RERUM-style baseline (Revenue Uplift Ranking, KDD 2024) — 简化复现.

RERUM 原文针对营销 revenue uplift（连续型异质处理效应），核心创新：
- (i) Tweedie / log-MSE 上的 ranking-aware loss，让模型在估准 GMV 增量大小的
      同时把高效用样本排到前面；
- (ii) 用 listwise pairwise margin loss（与 ranking-NDCG 相对应）替代单点 MSE，
      使下游 TopK 决策的预算-收益关系更陡。

由于原文未开源完整代码，本模块给出**简化复现**：
- 主结构：shared-encoder + per-arm GMV head（regression）+ per-arm conversion
  head（用于 funnel-aware 等场景），但 conversion head 默认权重 0（RERUM
  原文只关注 revenue 而不显式建模 conversion）；
- loss：log1p(GMV) MSE + λ_rank × pairwise margin ranking loss；
- 不做 funnel decomposition（与 FunnelCausalNet 形成对照）。

API 与 FunnelCausalNet 对齐：返回的 dict 形态与 metrics / pareto_ip_solver 兼容。
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
class RERUMConfig:
    d_in: int = 12
    num_arms: int = 8
    rep_hidden: List[int] = field(default_factory=lambda: [128, 64])
    rep_dim: int = 32
    head_hidden: List[int] = field(default_factory=lambda: [32])
    dropout: float = 0.1
    activation: str = "elu"
    use_conv_head: bool = True            # 也输出 conversion (CVR) 以便对接 §III pareto
    rank_lambda: float = 0.5              # listwise ranking loss weight
    rank_pairs_per_batch: int = 256       # 每 batch 采多少对 (i, j)
    rank_margin: float = 0.0              # margin = 0 ⇒ 纯 sigmoid pairwise


@dataclass
class RERUMTrainConfig:
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


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class RERUMNet(nn.Module):
    """Shared-encoder regression network with optional dual-output heads."""

    def __init__(self, cfg: RERUMConfig) -> None:
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

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        phi = self.encoder(x)
        out = {"phi": phi, "gmv_log_norm": self.gmv_head(phi)}
        if self.conv_head is not None:
            out["conv_logits"] = self.conv_head(phi)
        return out


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------


def _pairwise_ranking_loss(pred: torch.Tensor, target: torch.Tensor,
                            n_pairs: int, margin: float,
                            rng: np.random.Generator) -> torch.Tensor:
    """Listwise pairwise sigmoid loss (RankNet-style).

    pred, target: (B,) — 预测与真实 GMV (log1p_norm)。
    若 target_i > target_j，则希望 pred_i > pred_j。loss = log(1+exp(-(p_i-p_j)))。
    """
    B = pred.shape[0]
    if B < 2:
        return pred.sum() * 0.0
    n_pairs = min(n_pairs, B * (B - 1))
    idx_i = rng.integers(0, B, size=n_pairs)
    idx_j = rng.integers(0, B, size=n_pairs)
    keep = idx_i != idx_j
    if not keep.any():
        return pred.sum() * 0.0
    idx_i = idx_i[keep]; idx_j = idx_j[keep]

    t_i = target[idx_i]; t_j = target[idx_j]
    p_i = pred[idx_i]; p_j = pred[idx_j]
    sign = torch.sign(t_i - t_j)
    diff = p_i - p_j
    loss = torch.nn.functional.softplus(-(sign * diff) + margin)
    valid = (sign != 0)
    if not valid.any():
        return pred.sum() * 0.0
    return loss[valid].mean()


# ---------------------------------------------------------------------------
# Train + predict
# ---------------------------------------------------------------------------


def train_rerum_baseline(
    X: np.ndarray, T: np.ndarray, Y_cvr: np.ndarray, Y_gmv: np.ndarray,
    arch_cfg: RERUMConfig, train_cfg: RERUMTrainConfig,
    device: Optional[torch.device] = None,
) -> Tuple[RERUMNet, dict]:
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

    model = RERUMNet(arch_cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr,
                              weight_decay=train_cfg.weight_decay)
    rng = np.random.default_rng(train_cfg.seed)

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad = 0; history: List[dict] = []
    n_tr = X_tr.shape[0]; T_tr_np = T[tr_idx]

    def _loss(out, t, y_c, y_g_norm):
        N = t.shape[0]; arange = torch.arange(N, device=t.device)
        # GMV regression on factual arm
        g_pred = out["gmv_log_norm"][arange, t]
        L_reg = torch.mean((g_pred - y_g_norm).pow(2))
        L_rank = _pairwise_ranking_loss(g_pred, y_g_norm,
                                         arch_cfg.rank_pairs_per_batch,
                                         arch_cfg.rank_margin, rng)
        L_total = L_reg + arch_cfg.rank_lambda * L_rank
        if out.get("conv_logits") is not None:
            c_logit = out["conv_logits"][arange, t]
            L_c = nn.functional.binary_cross_entropy_with_logits(c_logit, y_c)
            L_total = L_total + 0.3 * L_c
        else:
            L_c = torch.zeros((), device=t.device)
        return L_total, L_reg, L_rank, L_c

    for epoch in range(train_cfg.max_epochs):
        model.train()
        epoch_acc = {"L": 0.0, "L_reg": 0.0, "L_rank": 0.0, "L_c": 0.0}
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
            L, L_reg, L_rank, L_c = _loss(out, tb, ycb, ygb)
            optim.zero_grad()
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optim.step()
            epoch_acc["L"] += L.item(); epoch_acc["L_reg"] += L_reg.item()
            epoch_acc["L_rank"] += L_rank.item(); epoch_acc["L_c"] += L_c.item()
            n_batches += 1
        for k in epoch_acc: epoch_acc[k] /= max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            out_va = model(X_va)
            val_L, _, _, _ = _loss(out_va, T_va, Yc_va, Yg_va_norm)
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


def predict_rerum_baseline(
    model: RERUMNet, X: np.ndarray, info: dict,
    device: Optional[torch.device] = None, batch_size: int = 4096,
) -> Dict[str, np.ndarray]:
    """与 FunnelCausalNet.predict_potentials 接口对齐。

    与 ECUP / Funnel 的差异：RERUM 直接回归 log1p(GMV)，不做 funnel
    decomposition。返回 mu_g_full = expm1(log_pred)，mu_c_full 来自可选的
    conv_head（若关闭则等于 1，意味着 mu_g 已是「期望 GMV」整体）。
    """
    device = device or pick_device()
    model = model.to(device).eval()
    log_g_mean, log_g_std = info["log_g_mean"], info["log_g_std"]
    K = model.cfg.num_arms; N = X.shape[0]

    mu_g_full = np.empty((N, K), dtype=np.float64)
    mu_c_full = np.empty((N, K), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, N, batch_size):
            xb = _to_tensor(X[s:s + batch_size], device)
            out = model(xb)
            g_log_norm = out["gmv_log_norm"].cpu().numpy().astype(np.float64)
            g_log = g_log_norm * log_g_std + log_g_mean
            mu_g = np.expm1(g_log)
            mu_g = np.clip(mu_g, 0.0, None)
            mu_g_full[s:s + batch_size] = mu_g
            if model.conv_head is not None:
                mu_c = torch.sigmoid(out["conv_logits"]).cpu().numpy()
                mu_c_full[s:s + batch_size] = mu_c.astype(np.float64)
            else:
                mu_c_full[s:s + batch_size] = 1.0
    mu_v_full = np.where(mu_c_full > 1e-6, mu_g_full / mu_c_full, mu_g_full)
    tau_c = mu_c_full - mu_c_full[:, [0]]
    tau_g = mu_g_full - mu_g_full[:, [0]]
    return {"mu_c_full": mu_c_full, "mu_v_full": mu_v_full, "mu_g_full": mu_g_full,
            "tau_c": tau_c, "tau_g": tau_g}
