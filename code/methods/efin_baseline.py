"""EFIN baseline (Liu et al. — KDD 2023, "Explicit Feature Interaction-aware Uplift Network") — 简化复现.

EFIN 原文 (KDD 2023) 主要 contribution:
- (i) intent-aware self-attention: 让模型在表征层挑选 treatment-relevant feature;
- (ii) treatment-aware cross-network: 让 representation 与 treatment-embedding
  做显式 multiplicative interaction (与 ECUP 的 "concat mean t_emb" 不同;
  EFIN 是 *per-arm* 的显式 cross interaction);
- (iii) counterfactual perturbation: 用 multi-task heads 同时学 factual 和
  counterfactual outcome.

由于原文未开源完整代码且面向 Tencent ad-platform binary uplift, 本模块给出
**简化复现**, 复刻其与 ECUP 的关键差异 (per-arm explicit feature-treatment
interaction):

- shared encoder phi(x);
- per-arm treatment embedding t_emb[k] (与 ECUP 一致);
- 对 *每个 arm k*, 通过一个共享的 cross-attention 投影 W_t * t_emb[k] 与
  phi(x) 做 element-wise product 得到 interaction_k = phi(x) * (W_t @ t_emb[k]);
- 每个 arm 的 head 接收 [phi(x); t_emb[k]; interaction_k] -> per-arm conv
  logit / GMV pred;
- 与 ECUP 的差异: ECUP 用 mean-treatment-embedding concat (所有 arm 共享同
  一份 t_emb 输入), EFIN 是 *per-arm* explicit interaction, 显式让不同 arm
  的 representation 不同.

API 与 funnel_causal_net / ecup_baseline / rerum_baseline / cfrnet_baseline /
dragonnet_baseline 对齐。
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
class EFINConfig:
    d_in: int = 12
    num_arms: int = 8
    rep_hidden: List[int] = field(default_factory=lambda: [128, 64])
    rep_dim: int = 32
    head_hidden: List[int] = field(default_factory=lambda: [32])
    treatment_emb_dim: int = 16
    dropout: float = 0.1
    activation: str = "elu"
    use_intent_attention: bool = True   # (i) intent-aware self-attention
    use_cross_interaction: bool = True  # (ii) per-arm explicit cross interaction
    intent_attention_heads: int = 2     # in self-attention block


@dataclass
class EFINTrainConfig:
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
    alpha_v: float = 0.3                # weight on L_v on conv=1 子集


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class _IntentAttention(nn.Module):
    """Lightweight self-attention over feature dimensions for intent gating.

    Treats each rep_dim slot as a token; uses 1-step multi-head self-attention
    to learn a soft mask over slots highlighting treatment-relevant features.
    """

    def __init__(self, rep_dim: int, num_heads: int = 2) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(rep_dim, num_heads=num_heads,
                                          batch_first=True)
        self.gate = nn.Sequential(nn.Linear(rep_dim, rep_dim), nn.Sigmoid())

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        # phi: (B, rep_dim) -> (B, 1, rep_dim) -> attention -> (B, rep_dim)
        z = phi.unsqueeze(1)
        z, _ = self.attn(z, z, z, need_weights=False)
        z = z.squeeze(1)
        gate = self.gate(z)
        return phi * gate


class EFINNet(nn.Module):
    """Shared encoder + intent-attention + per-arm explicit feature-treatment interaction."""

    def __init__(self, cfg: EFINConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = _mlp(cfg.d_in, cfg.rep_hidden, cfg.rep_dim,
                            activation=cfg.activation, dropout=cfg.dropout)
        if cfg.use_intent_attention:
            self.intent_attn = _IntentAttention(cfg.rep_dim,
                                                num_heads=cfg.intent_attention_heads)
        else:
            self.intent_attn = None
        self.treatment_emb = nn.Embedding(cfg.num_arms, cfg.treatment_emb_dim)
        if cfg.use_cross_interaction:
            # 把 t_emb 投射到 rep_dim 上, 与 phi 做 element-wise product
            self.cross_proj = nn.Linear(cfg.treatment_emb_dim, cfg.rep_dim)
        else:
            self.cross_proj = None
        # per-arm head input: [phi (rep_dim); t_emb (emb_dim); interaction (rep_dim if用)]
        head_in = cfg.rep_dim + cfg.treatment_emb_dim
        if cfg.use_cross_interaction:
            head_in += cfg.rep_dim
        # 给 conv / GMV 各 K+1 个 head, 保留 fan-out=num_arms 但每个 arm 接收
        # 自己的 head_input_k -> 我们把 num_arms 个 head 用一个 (K+1, head_in -> 1) 实现
        self.conv_heads = nn.ModuleList(
            [_mlp(head_in, cfg.head_hidden, 1,
                  activation=cfg.activation, dropout=cfg.dropout)
             for _ in range(cfg.num_arms)])
        self.gmv_heads = nn.ModuleList(
            [_mlp(head_in, cfg.head_hidden, 1,
                  activation=cfg.activation, dropout=cfg.dropout)
             for _ in range(cfg.num_arms)])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.shape[0]
        phi = self.encoder(x)
        if self.intent_attn is not None:
            phi = self.intent_attn(phi)
        K = self.cfg.num_arms
        # 收集所有 arm 的 head input, 一次性 stack 后过 K+1 个 head
        conv_logits = torch.empty((B, K), device=x.device)
        val_log_norm = torch.empty((B, K), device=x.device)
        for k in range(K):
            t_k_emb = self.treatment_emb.weight[k].unsqueeze(0).expand(B, -1)  # (B, emb)
            head_in_parts = [phi, t_k_emb]
            if self.cross_proj is not None:
                proj_k = self.cross_proj(t_k_emb)        # (B, rep_dim)
                interaction_k = phi * proj_k             # element-wise product
                head_in_parts.append(interaction_k)
            head_in = torch.cat(head_in_parts, dim=1)
            conv_logits[:, k] = self.conv_heads[k](head_in).squeeze(-1)
            val_log_norm[:, k] = self.gmv_heads[k](head_in).squeeze(-1)
        return {"conv_logits": conv_logits, "val_log_norm": val_log_norm, "phi": phi}


# ---------------------------------------------------------------------------
# Train + predict
# ---------------------------------------------------------------------------


def train_efin_baseline(
    X: np.ndarray, T: np.ndarray, Y_cvr: np.ndarray, Y_gmv: np.ndarray,
    arch_cfg: EFINConfig, train_cfg: EFINTrainConfig,
    device: Optional[torch.device] = None,
) -> Tuple[EFINNet, dict]:
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

    model = EFINNet(arch_cfg).to(device)
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
        g_pred = out["val_log_norm"][arange, t]
        mask = (y_c > 0.5)
        if mask.any():
            L_v = torch.mean((g_pred[mask] - y_g_norm[mask]).pow(2))
        else:
            L_v = torch.zeros((), device=t.device)
        return L_c + train_cfg.alpha_v * L_v, L_c, L_v

    for epoch in range(train_cfg.max_epochs):
        model.train()
        epoch_acc = {"L": 0.0, "L_c": 0.0, "L_v": 0.0}
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


def predict_efin_baseline(
    model: EFINNet, X: np.ndarray, info: dict,
    device: Optional[torch.device] = None, batch_size: int = 4096,
) -> Dict[str, np.ndarray]:
    """与 funnel/ECUP/RERUM/CFRNet/DragonNet predict 接口对齐.

    EFIN 推断:
    - mu_c = sigmoid(conv_logits)              (per-arm CVR)
    - mu_v = expm1(denorm(val_log_norm))       (per-arm conv=1 期望 GMV)
    - mu_g = mu_c * mu_v                       (funnel composition)
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
