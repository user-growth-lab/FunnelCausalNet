"""CFRNet baseline (Shalit, Johansson, Sontag — ICML 2017) — 多档简化复现.

CFRNet 原文针对二值 treatment：

- shared encoder Phi(x): 投影到表征空间;
- per-arm head h_t(Phi(x)): 每个 arm 一个独立 outcome head;
- IPM 正则: 在表征空间上对齐 control / treatment 的边缘分布
  L_total = sum_i w_i * (Y_i - h_{T_i}(Phi(x_i)))^2 + alpha * IPM(P_phi^c, P_phi^t)

多档（K+1 arm）场景下的扩展原则与 ICML'17 sec.3 一致, 但需要选择一种
"multi-arm IPM"。我们采用与 SITE / Perfect-Match 类似的做法:

    IPM_total = (1 / K) * sum_{t=1..K} IPM(P_phi^0, P_phi^t)

即把 control arm 当作 anchor, 对每个 treatment arm 算一次表征对齐损失再
平均, 这等价于在二值 CFRNet 上对每个 treatment 做一次 marginal CFR。

IPM 内核选择 linear-MMD 的平方 (见 Gretton 2012); 实现上用 batch 内的
mean(phi) 差的 L2 平方:

    IPM(c, t) = || mean_{i:T_i=0}(phi_i) - mean_{i:T_i=t}(phi_i) ||_2^2

之所以选 linear-MMD: (i) O(B * d) 不需 kernel matrix; (ii) 在 ICML'17
作者后续 SITE / Perfect-Match 中也证实在 multi-tier 半合成上稳定; (iii)
我们的 batch 是 arm-balanced 的 (rep code _arm_balanced_batch), 每个 arm
都有覆盖, 不会出现退化为 0 的情况。

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
class CFRNetConfig:
    d_in: int = 12
    num_arms: int = 8
    rep_hidden: List[int] = field(default_factory=lambda: [128, 64])
    rep_dim: int = 32
    head_hidden: List[int] = field(default_factory=lambda: [32])
    dropout: float = 0.1
    activation: str = "elu"
    ipm_alpha: float = 1.0          # IPM penalty weight (lambda in原文)
    use_conv_head: bool = True      # 也输出 CVR，供统一漏斗评估接口使用


@dataclass
class CFRNetTrainConfig:
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
    alpha_v: float = 0.3            # weight on L_v 在 conv=1 子集 (与 ECUP 对齐)


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class CFRNetModel(nn.Module):
    """Shared-encoder per-arm dual-head network with linear-MMD IPM regularizer."""

    def __init__(self, cfg: CFRNetConfig) -> None:
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
        out: Dict[str, torch.Tensor] = {
            "phi": phi,
            "val_log_norm": self.gmv_head(phi),
        }
        if self.conv_head is not None:
            out["conv_logits"] = self.conv_head(phi)
        return out


# ---------------------------------------------------------------------------
# Linear-MMD IPM
# ---------------------------------------------------------------------------


def _linear_mmd_ipm(phi: torch.Tensor, t: torch.Tensor, num_arms: int,
                    control_idx: int = 0) -> torch.Tensor:
    """Multi-arm IPM: 平均的 (control vs arm-t) linear-MMD^2.

    若某个 arm 在 batch 内不出现, 跳过该 arm; 全部跳过时返回 0.
    """
    mask_c = (t == control_idx)
    if mask_c.sum() == 0:
        return phi.sum() * 0.0
    phi_c_mean = phi[mask_c].mean(dim=0)
    accum = torch.zeros((), device=phi.device); cnt = 0
    for k in range(num_arms):
        if k == control_idx:
            continue
        mask_k = (t == k)
        if mask_k.sum() == 0:
            continue
        phi_k_mean = phi[mask_k].mean(dim=0)
        diff = phi_c_mean - phi_k_mean
        accum = accum + (diff * diff).sum()
        cnt += 1
    if cnt == 0:
        return phi.sum() * 0.0
    return accum / float(cnt)


# ---------------------------------------------------------------------------
# Train + predict
# ---------------------------------------------------------------------------


def train_cfrnet_baseline(
    X: np.ndarray, T: np.ndarray, Y_cvr: np.ndarray, Y_gmv: np.ndarray,
    arch_cfg: CFRNetConfig, train_cfg: CFRNetTrainConfig,
    device: Optional[torch.device] = None,
) -> Tuple[CFRNetModel, dict]:
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

    model = CFRNetModel(arch_cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr,
                              weight_decay=train_cfg.weight_decay)
    rng = np.random.default_rng(train_cfg.seed)

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad = 0; history: List[dict] = []
    n_tr = X_tr.shape[0]; T_tr_np = T[tr_idx]

    def _loss(out, t, y_c, y_g_norm):
        N = t.shape[0]; arange = torch.arange(N, device=t.device)
        g_pred = out["val_log_norm"][arange, t]
        # GMV 头只在 conv=1 子集训练 (与 funnel/ECUP 对齐), 否则 mass 被 0 主导
        mask = (y_c > 0.5)
        if mask.any():
            L_v = torch.mean((g_pred[mask] - y_g_norm[mask]).pow(2))
        else:
            L_v = torch.zeros((), device=t.device)
        L_ipm = _linear_mmd_ipm(out["phi"], t, model.cfg.num_arms)
        L_total = train_cfg.alpha_v * L_v + arch_cfg.ipm_alpha * L_ipm
        if out.get("conv_logits") is not None:
            c_logit = out["conv_logits"][arange, t]
            L_c = nn.functional.binary_cross_entropy_with_logits(c_logit, y_c)
            L_total = L_total + L_c
        else:
            L_c = torch.zeros((), device=t.device)
        return L_total, L_c, L_v, L_ipm

    for epoch in range(train_cfg.max_epochs):
        model.train()
        epoch_acc = {"L": 0.0, "L_c": 0.0, "L_v": 0.0, "L_ipm": 0.0}
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
            L, L_c, L_v, L_ipm = _loss(out, tb, ycb, ygb)
            optim.zero_grad()
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optim.step()
            epoch_acc["L"] += L.item(); epoch_acc["L_c"] += L_c.item()
            epoch_acc["L_v"] += L_v.item(); epoch_acc["L_ipm"] += L_ipm.item()
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


def predict_cfrnet_baseline(
    model: CFRNetModel, X: np.ndarray, info: dict,
    device: Optional[torch.device] = None, batch_size: int = 4096,
) -> Dict[str, np.ndarray]:
    """与 FunnelCausalNet.predict_potentials / ECUP / RERUM 对齐:
    返回 mu_c_full, mu_v_full, mu_g_full, tau_c, tau_g (per arm / vs control).

    与 ECUP 的差异: CFRNet 是 IPM-balanced representation; mu_g = mu_c * mu_v
    的 funnel composition 仍然成立 (因 conv head 与 GMV head 各自 per-arm 训练,
    且 GMV head 在 conv=1 mask 上学的是 E[GMV | conv=1, x, t]).
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
