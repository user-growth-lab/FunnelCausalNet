"""Uplift / business metrics shared across E1 / E4 runners.

包含:
- ``compute_auuc_qini_business``: 业务 RCT 数据上的 AUUC + Qini AUC
  (二值化 control vs max-arm; 多档场景 max-arm = NUM_ARMS - 1)
- ``compute_topk_ate_business``: TopK Cell ATE (PEHE 业务 proxy)
- ``compute_delta_roi_curve``: ΔROI = Σ_i incremental_GMV / Σ_i cost
  (用 RCT 经验组均值，要求 cost = ratio × E[Y_gmv | T=k] 公式)

设计：
- 业务 RCT 数据上 PEHE 不可直接计算 (无 ground-truth ITE)，用 AUUC + TopK
  ATE 作为 ranking + heterogeneity proxy
- ΔROI 仅在有明确 cost_ratio 定义的数据集上有意义；公开 RCT
  (Hillstrom / Criteo Uplift v2.1) 上 cost_ratio 是合成的，建议只报 AUUC +
  TopK ATE，不报 ΔROI
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# AUUC / Qini AUC
# ---------------------------------------------------------------------------


def compute_auuc_qini_business(
    tau_pred_max: np.ndarray,
    T_obs: np.ndarray,
    Y: np.ndarray,
    max_arm: int = 1,
    rng_seed: int = 42,
) -> Dict[str, float]:
    """业务 RCT 数据上的 AUUC + Qini AUC，多档场景下二值化为 (control vs max-arm).

    业内标准做法 (参考 mt-lift / Criteo Uplift v2.1 论文):
      1. 取测试集中 T ∈ {0, max_arm} 的子集
      2. 按 tau_pred_max 降序排
      3. 沿排序累积 cumulative uplift = mean(Y|T=max,前 i 名) - mean(Y|T=0,前 i 名)
      4. AUUC = trapz(uplift, fraction); Qini = trapz(absolute_uplift, n_treated)

    Returns
    -------
    dict with AUUC, Qini, AUUC_baseline (random ranking baseline)
    """
    mask = (T_obs == 0) | (T_obs == max_arm)
    if not mask.any():
        return {"AUUC": float("nan"), "Qini": float("nan"),
                "AUUC_baseline": float("nan")}
    tau_sub = tau_pred_max[mask]
    T_sub = T_obs[mask].copy()
    Y_sub = Y[mask].copy()
    T_bin = (T_sub == max_arm).astype(np.int64)

    n = len(tau_sub)
    n_t = int(T_bin.sum())
    n_c = n - n_t
    if n_t == 0 or n_c == 0:
        return {"AUUC": float("nan"), "Qini": float("nan"),
                "AUUC_baseline": float("nan")}

    order = np.argsort(-tau_sub)
    Y_o = Y_sub[order]
    T_o = T_bin[order]

    cum_t = np.cumsum(Y_o * T_o)
    cum_c = np.cumsum(Y_o * (1 - T_o))
    cum_n_t = np.cumsum(T_o)
    cum_n_c = np.cumsum(1 - T_o)

    eps = 1e-9
    mean_t = cum_t / np.maximum(cum_n_t, eps)
    mean_c = cum_c / np.maximum(cum_n_c, eps)
    uplift_curve = (mean_t - mean_c) * (cum_n_t > 0) * (cum_n_c > 0)
    fractions = np.arange(1, n + 1) / n
    auuc = float(np.trapezoid(uplift_curve, fractions))

    qini_curve = (cum_t - cum_c * (n_t / n)) / n
    qini = float(np.trapezoid(qini_curve, fractions))

    rng = np.random.default_rng(rng_seed)
    rand_perm = rng.permutation(n)
    Y_r = Y_sub[rand_perm]; T_r = T_bin[rand_perm]
    cum_t_r = np.cumsum(Y_r * T_r); cum_c_r = np.cumsum(Y_r * (1 - T_r))
    cum_nt_r = np.cumsum(T_r); cum_nc_r = np.cumsum(1 - T_r)
    mean_t_r = cum_t_r / np.maximum(cum_nt_r, eps)
    mean_c_r = cum_c_r / np.maximum(cum_nc_r, eps)
    base_curve = (mean_t_r - mean_c_r) * (cum_nt_r > 0) * (cum_nc_r > 0)
    auuc_base = float(np.trapezoid(base_curve, fractions))

    return {"AUUC": auuc, "Qini": qini, "AUUC_baseline": auuc_base}


# ---------------------------------------------------------------------------
# TopK Cell ATE (PEHE business proxy)
# ---------------------------------------------------------------------------


def compute_topk_ate_business(
    tau_pred_max: np.ndarray,
    T_obs: np.ndarray,
    Y: np.ndarray,
    K_fracs: Optional[List[float]] = None,
    max_arm: int = 1,
    rng_seed: int = 42,
) -> Dict[str, float]:
    """TopK Cell ATE: 模型选出的高 ITE 用户群是否真的有更高 RCT ATE.

    PEHE 在 RCT 数据上的业界 proxy: 取 model 预测 τ̂(x, max_arm) 排序的
    top K%，看在该子集上的真实 RCT ATE = E[Y|T=max] - E[Y|T=0]，与
    random K% 的 ATE 对比 (uplift gain)。

    Returns dict with keys "ATE_TopK_{K}", "ATE_TopK_{K}_random", "ATE_TopK_{K}_lift".
    """
    if K_fracs is None:
        K_fracs = [0.05, 0.10, 0.20]
    mask = (T_obs == 0) | (T_obs == max_arm)
    tau_sub = tau_pred_max[mask]
    T_sub = T_obs[mask].copy()
    Y_sub = Y[mask].copy()
    T_bin = (T_sub == max_arm).astype(np.int64)

    n = len(tau_sub)
    if n == 0:
        return {f"ATE_TopK_{int(K*100)}": float("nan") for K in K_fracs}

    order = np.argsort(-tau_sub)
    Y_o = Y_sub[order]; T_o = T_bin[order]

    rng = np.random.default_rng(rng_seed)
    rand_perm = rng.permutation(n)
    Y_r = Y_sub[rand_perm]; T_r = T_bin[rand_perm]

    out: Dict[str, float] = {}
    for K in K_fracs:
        k = max(1, int(n * K))
        Y_k = Y_o[:k]; T_k = T_o[:k]
        if T_k.sum() == 0 or (k - T_k.sum()) == 0:
            ate_topk = float("nan")
        else:
            ate_topk = float(Y_k[T_k == 1].mean() - Y_k[T_k == 0].mean())
        Y_kr = Y_r[:k]; T_kr = T_r[:k]
        if T_kr.sum() == 0 or (k - T_kr.sum()) == 0:
            ate_rand = float("nan")
        else:
            ate_rand = float(Y_kr[T_kr == 1].mean() - Y_kr[T_kr == 0].mean())

        Kpct = int(K * 100)
        out[f"ATE_TopK_{Kpct}"] = ate_topk
        out[f"ATE_TopK_{Kpct}_random"] = ate_rand
        out[f"ATE_TopK_{Kpct}_lift"] = (
            (ate_topk - ate_rand) / max(abs(ate_rand), 1e-6) * 100
            if not (np.isnan(ate_topk) or np.isnan(ate_rand))
            else float("nan")
        )
    return out


__all__ = [
    "compute_auuc_qini_business",
    "compute_topk_ate_business",
]
