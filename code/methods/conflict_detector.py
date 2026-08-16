"""Conflict-Aware ranking detector for §II 第 4 项 「排序冲突高风险集合 C(K)」.

C(K) = { i :
  |r_c(i) - r_g(i)| > δ_rank          (排名分歧大)
  ∧  min(r_c(i), r_g(i)) ≤ K/100      (在 TopK 决策边界内)
  ∧  max(w_c(i), w_g(i)) > δ_width    (至少一个目标的区间宽度高)
}

其中 r_c, r_g 分别是 τ_CVR 与 τ_GMV 排名百分位（0=最高，1=最低）；w_c / w_g
是 τ_CVR / τ_GMV 区间宽度的标准化值。

本模块对推理输出 (tau_c, tau_g, width_c, width_g) 在指定档位 t 上做检测，
返回布尔掩码 + 诊断统计。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class ConflictDetectorConfig:
    arm_for_decision: int = 7         # 在哪个 arm 上做 TopK 决策诊断（默认 14% 力度）
    top_k_pct: float = 10.0           # TopK 边界，单位 %
    delta_rank: float = 0.20          # 排名分歧阈值（百分位差）
    delta_width_quantile: float = 0.75  # 宽度阈值用 75% 分位（标准化后）
    conflict_mode: str = "both"       # "both" = |Δrank|>δ_rank（Type A + B 双向冲突，
                                       #   §II 第 4 项原始定义，半合成 ablation 用）；
                                       # "type_a_only" = rank_g - rank_c > δ_rank
                                       #   （仅"高 CVR rank ∩ 低 GMV rank"用户，业务上对应
                                       #   可用于审核低净收益子群）。


def _percentile_rank(x: np.ndarray) -> np.ndarray:
    """Per-user percentile rank over 1D array (0=largest, 1=smallest)."""
    order = np.argsort(-x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x)) / max(len(x) - 1, 1)
    return ranks


def detect_conflict_users(
    tau_c_hat: np.ndarray,             # (N, K+1) point estimate (or column slice)
    tau_g_hat: np.ndarray,             # (N, K+1)
    width_c: np.ndarray,               # (N, K+1) interval width on τ_CVR
    width_g: np.ndarray,               # (N, K+1) interval width on τ_GMV
    cfg: ConflictDetectorConfig,
) -> Dict[str, np.ndarray]:
    """Compute the conflict-user set C(K) at the chosen decision arm.

    Returns dict:
        mask          : (N,) bool array of conflict users
        rank_c, rank_g: (N,) percentile ranks of τ_c / τ_g at arm t
        rank_diff     : (N,) |r_c - r_g|
        width_c_norm  : (N,) normalized width_c at arm t (max-norm)
        width_g_norm  : (N,) normalized width_g at arm t
        n_conflict, frac_conflict : scalars
        config_snapshot : echo of cfg fields
    """
    t = cfg.arm_for_decision
    tau_c_t = tau_c_hat[:, t]
    tau_g_t = tau_g_hat[:, t]
    w_c_t = width_c[:, t]
    w_g_t = width_g[:, t]

    rank_c = _percentile_rank(tau_c_t)
    rank_g = _percentile_rank(tau_g_t)
    if cfg.conflict_mode == "type_a_only":
        # rank uses 0=highest convention; "high CVR rank ∩ low GMV rank" =
        # rank_c is small (close to 0) AND rank_g is large (close to 1).
        # So rank_g - rank_c > δ → Type A conflict (high CVR top, low GMV bottom).
        rank_diff = rank_g - rank_c
    else:
        rank_diff = np.abs(rank_c - rank_g)

    w_c_norm = w_c_t / max(w_c_t.max(), 1e-9)
    w_g_norm = w_g_t / max(w_g_t.max(), 1e-9)

    # Saturation guard: CVR-side conformal width often saturates at the
    # probability-scale boundary (lo→0, hi→1 at α≤0.10), making w_c_norm
    # ≈ constant 1.0 → np.maximum compresses w_g signal entirely.  Detect
    # saturation (q10 of normalized w_c > 0.95) and fall back to w_g only.
    w_c_saturated = bool(float(np.quantile(w_c_norm, 0.10)) > 0.95)
    w_used = w_g_norm if w_c_saturated else np.maximum(w_c_norm, w_g_norm)

    width_thr = float(np.quantile(w_used, cfg.delta_width_quantile))

    in_topk_band = np.minimum(rank_c, rank_g) <= (cfg.top_k_pct / 100.0)
    rank_diverge = rank_diff > cfg.delta_rank
    width_high = w_used > width_thr

    mask = in_topk_band & rank_diverge & width_high

    return {
        "mask": mask,
        "rank_c": rank_c,
        "rank_g": rank_g,
        "rank_diff": rank_diff,
        "width_c_norm": w_c_norm,
        "width_g_norm": w_g_norm,
        "width_thr_used": width_thr,
        "w_c_saturated": w_c_saturated,
        "n_conflict": int(mask.sum()),
        "frac_conflict": float(mask.mean()),
        "config_snapshot": {
            "arm_for_decision": cfg.arm_for_decision,
            "top_k_pct": cfg.top_k_pct,
            "delta_rank": cfg.delta_rank,
            "delta_width_quantile": cfg.delta_width_quantile,
            "conflict_mode": cfg.conflict_mode,
        },
    }


def evaluate_conflict_recall(
    detected_mask: np.ndarray,
    tau_c_gt: np.ndarray,
    tau_g_gt: np.ndarray,
    cfg: ConflictDetectorConfig,
) -> Dict[str, float]:
    """Compare detected C(K) against ground-truth conflict users (rank divergence
    + TopK band on the *ground truth* effects).

    Ground truth C* is defined identically but using gt τ values; width filter
    is dropped (no width on ground truth). This gives a direct
    Precision/Recall/F1 measurement on the ranking-conflict aspect.
    """
    t = cfg.arm_for_decision
    rank_c_gt = _percentile_rank(tau_c_gt[:, t])
    rank_g_gt = _percentile_rank(tau_g_gt[:, t])
    rank_diff_gt = np.abs(rank_c_gt - rank_g_gt)
    in_topk_gt = np.minimum(rank_c_gt, rank_g_gt) <= (cfg.top_k_pct / 100.0)

    gt_mask = (rank_diff_gt > cfg.delta_rank) & in_topk_gt

    tp = int((detected_mask & gt_mask).sum())
    fp = int((detected_mask & ~gt_mask).sum())
    fn = int((~detected_mask & gt_mask).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_gt_conflict": int(gt_mask.sum()),
        "n_detected": int(detected_mask.sum()),
    }
