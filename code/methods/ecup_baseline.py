"""ECUP-style baseline (Liu et al., SIGIR 2024) — 简化复现版本.

Liu et al. ECUP 原文针对多档优惠券场景识别两类偏差：
- (i) chain-bias       — 多任务串联（CTR → CTCVR → ...）让低层任务的因果信号
                         失真；ECUP 用 task-prior 信息流缓解。
- (ii) treatment-unadaptive — 共享 encoder 学不到 K+1 档治疗的差异；ECUP 用
                              treatment-aware adapter 把 treatment embedding 注入
                              各 head。

由于原文未开源完整代码且面向 MT-LIFT (无 GMV)，本模块给出**简化复现**：
- 主结构沿用 FunnelCausalNet 的 shared encoder + per-arm head，
  额外加一个 K+1 维 treatment embedding 表，concat 到 encoder 输出后再过 head
  （treatment-adaptive 版的 adapter 简化）；
- chain-bias 修正用「CTR head 的预测被 detach 后参与 CTCVR head 的输入」
  做轻量复现，避免 CTCVR 反向传播污染 CTR；
- 损失：BCE on conv + MSE on log1p(GMV) on conv=1 子集（与 FunnelCausalNet
  hard mode 一致）。

API 与 FunnelCausalNet 对齐：train_ecup_baseline / predict_ecup_baseline
返回的 dict 形态与现有 metrics / pareto_ip_solver 兼容。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# 复用 FunnelCausalNet 的工具函数（pick_device、_to_tensor、_train_val_split、
# _balanced_arm_batch 等）。
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
class ECUPConfig:
    d_in: int = 12
    num_arms: int = 8
    rep_hidden: List[int] = field(default_factory=lambda: [128, 64])
    rep_dim: int = 32
    head_hidden: List[int] = field(default_factory=lambda: [32])
    treatment_emb_dim: int = 8           # treatment-adaptive embedding 维度
    dropout: float = 0.1
    activation: str = "elu"
    enable_chain_bias_fix: bool = True   # 是否对 CTR pred 做 detach 再传给 GMV head
    chain_detach_alpha: float = 0.5      # detach 比例：1=完全 detach，0=无修正


@dataclass
class ECUPTrainConfig:
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
    alpha_v: float = 0.3                 # weight on L_v (与 FunnelCausalNet hard 对齐)


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class ECUPNet(nn.Module):
    """Treatment-adaptive shared-encoder dual-head network (ECUP-style)."""

    def __init__(self, cfg: ECUPConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = _mlp(cfg.d_in, cfg.rep_hidden, cfg.rep_dim,
                            activation=cfg.activation, dropout=cfg.dropout)
        self.treatment_emb = nn.Embedding(cfg.num_arms, cfg.treatment_emb_dim)
        adapter_in = cfg.rep_dim + cfg.treatment_emb_dim

        self.conv_head = _mlp(adapter_in, cfg.head_hidden, cfg.num_arms,
                              activation=cfg.activation, dropout=cfg.dropout)
        # GMV head 的输入额外 concat 一个 CTR-pred scalar (chain-bias 修正路径)
        gmv_in = adapter_in + 1
        self.val_head = _mlp(gmv_in, cfg.head_hidden, cfg.num_arms,
                             activation=cfg.activation, dropout=cfg.dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        N = x.shape[0]
        phi = self.encoder(x)                               # (N, rep_dim)

        # 每用户用 mean treatment embedding 作为 treatment-adaptive 适配
        # （也可以按 t 取实际 emb，但这里简化为「每用户一份共享适配」让所有
        # arm 的预测使用同一 phi+t_emb_mean，再由 K+1 维输出区分）。
        t_emb_mean = self.treatment_emb.weight.mean(dim=0).expand(N, -1)
        phi_adapt = torch.cat([phi, t_emb_mean], dim=1)     # (N, rep_dim+emb_dim)

        conv_logits = self.conv_head(phi_adapt)             # (N, K+1)

        if self.cfg.enable_chain_bias_fix:
            # chain-bias 修正：CTR head 的预测被 detach 后作为 GMV head 输入特征
            ctr_signal = torch.sigmoid(conv_logits.detach()
                                        ) * self.cfg.chain_detach_alpha
        else:
            ctr_signal = torch.sigmoid(conv_logits)
        # 取 ctr_signal 在 max-arm 上的 scalar 作为 chain feature（最具代表性）
        ctr_max = ctr_signal[:, -1:].clone()                # (N, 1)

        gmv_in = torch.cat([phi_adapt, ctr_max], dim=1)
        val_log_norm = self.val_head(gmv_in)                # (N, K+1)

        return {
            "conv_logits": conv_logits,
            "val_log_norm": val_log_norm,
            "phi": phi,
        }


# ---------------------------------------------------------------------------
# Train + predict
# ---------------------------------------------------------------------------


def train_ecup_baseline(
    X: np.ndarray, T: np.ndarray, Y_cvr: np.ndarray, Y_gmv: np.ndarray,
    arch_cfg: ECUPConfig, train_cfg: ECUPTrainConfig,
    device: Optional[torch.device] = None,
) -> Tuple[ECUPNet, dict]:
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

    model = ECUPNet(arch_cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr,
                              weight_decay=train_cfg.weight_decay)
    rng = np.random.default_rng(train_cfg.seed)

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad = 0; history: List[dict] = []
    n_tr = X_tr.shape[0]; T_tr_np = T[tr_idx]

    def _loss(out, t, y_c, y_g_norm):
        N = t.shape[0]; arange = torch.arange(N, device=t.device)
        L_c = nn.functional.binary_cross_entropy_with_logits(
            out["conv_logits"][arange, t], y_c)
        obs_val = out["val_log_norm"][arange, t]
        mask = (y_c > 0.5)
        if mask.any():
            L_v = torch.mean((obs_val[mask] - y_g_norm[mask]).pow(2))
        else:
            L_v = torch.zeros((), device=t.device)
        return L_c + train_cfg.alpha_v * L_v, L_c, L_v

    for epoch in range(train_cfg.max_epochs):
        model.train()
        epoch_acc = {"L": 0.0, "L_c": 0.0, "L_v": 0.0}; n_batches = 0
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
            L, L_c, L_v = _loss(out, tb, ycb, ygb)
            optim.zero_grad()
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optim.step()
            epoch_acc["L"] += L.item(); epoch_acc["L_c"] += L_c.item()
            epoch_acc["L_v"] += L_v.item(); n_batches += 1
        for k in epoch_acc: epoch_acc[k] /= max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            out_va = model(X_va)
            val_L, _, _ = _loss(out_va, T_va, Yc_va, Yg_va_norm)
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


def predict_ecup_baseline(
    model: ECUPNet, X: np.ndarray, info: dict,
    device: Optional[torch.device] = None, batch_size: int = 4096,
) -> Dict[str, np.ndarray]:
    """与 FunnelCausalNet.predict_potentials 接口对齐：
    返回 mu_c_full, mu_v_full, mu_g_full, tau_c, tau_g (per arm / vs control)。
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
            mu_c = torch.sigmoid(out["conv_logits"]).cpu().numpy().astype(np.float64)
            val_log_norm = out["val_log_norm"].cpu().numpy().astype(np.float64)
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
