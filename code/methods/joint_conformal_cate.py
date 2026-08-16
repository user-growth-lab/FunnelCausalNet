"""Joint Conformal CATE for double outcome (CVR + GMV) under funnel structure.

Implements 创新点 II §II.5 定理 1（Bonferroni split-CP）的工程化版本：
- Split conformal calibration on factual (T_obs) data.
- Per-arm CQR (Conformalized Quantile Regression) on conv probability and on
  log1p(GMV | conv=1) value head.
- Bonferroni 联合：α = α_c + α_v 拆分两个 outcome；进一步对 4 个 source
  intervals (conv@t, conv@0, val@t, val@0) 用 α/4 拼接 GMV uplift 区间。
- 输出 per-arm interval band [τ_c^lo, τ_c^hi], [τ_g^lo, τ_g^hi]。

数学对应 §II.5：
    定理 1（边际联合覆盖率）：alpha = alpha_c + alpha_v ≤ 1 ⇒
        P(Y_c* ∈ Î_c ∧ Y_g* ∈ Î_v) ≥ 1 - alpha
    推论 1.1（τ 区间覆盖）：4 个 source 区间各取 α/4 ⇒
        P(τ_c ∈ Î_τc ∧ τ_g ∈ Î_τg) ≥ 1 - alpha
    命题 3（漏斗分解保留）：区间算术 envelope:
        τ_g^lo(x,t) = q_c^lo(x,t)·q_v^lo(x,t) - q_c^hi(x,0)·q_v^hi(x,0)
        τ_g^hi(x,t) = q_c^hi(x,t)·q_v^hi(x,t) - q_c^lo(x,0)·q_v^lo(x,0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from .funnel_causal_net import (
    FunnelCausalNet,
    predict_potentials,
    predict_quantile_bounds,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class JointCPConfig:
    """Hyper-parameters for joint conformal CATE.

    alpha_total = alpha_c + alpha_v ≤ 1; 默认 0.05 + 0.05 = 0.1 (90% joint coverage).
    """

    alpha_c: float = 0.05
    alpha_v: float = 0.05


# ---------------------------------------------------------------------------
# CQR offset computation
# ---------------------------------------------------------------------------


def _split_cp_offset(scores: np.ndarray, alpha: float) -> float:
    """Vovk / Lei-Candès split-CP 标准分位：
        q = ceil((n+1)*(1-alpha)) / n 分位数 of scores.

    Returns +inf if scores is empty (defensive).
    """
    n = scores.shape[0]
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(np.partition(scores, k - 1)[k - 1])


def _compute_arm_offset(
    quantile_lo: np.ndarray,
    quantile_hi: np.ndarray,
    target: np.ndarray,
    alpha_per_outcome: float,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Standard CQR score: s_i = max(q_lo - y, y - q_hi).

    Returns the (1-α) split-CP offset Q̂.

    Parameters
    ----------
    quantile_lo / hi : (N,) factual-arm raw quantile predictions
    target           : (N,) factual outcome on the same scale
    alpha_per_outcome : split level for the single outcome
    mask             : optional bool mask (e.g. conv=1 for OV head)
    """
    if mask is not None:
        ql = quantile_lo[mask]; qh = quantile_hi[mask]; y = target[mask]
    else:
        ql = quantile_lo; qh = quantile_hi; y = target
    scores = np.maximum(ql - y, y - qh)
    return _split_cp_offset(scores, alpha_per_outcome)


# ---------------------------------------------------------------------------
# Calibration on a held-out set
# ---------------------------------------------------------------------------


@dataclass
class CalibratedJointCP:
    """Calibrated offsets for conv and OV heads on log1p(GMV) scale.

    All offsets are stored *per arm* so that arm-specific miscoverage is
    isolated; the raw quantile heads are arm-conditioned, so this matches.
    """

    alpha_c: float
    alpha_v: float
    offset_conv: np.ndarray              # (K+1,) split-CP offset for CVR per arm
    offset_val_log: np.ndarray           # (K+1,) split-CP offset on log1p-GMV per arm
    cal_n_per_arm: np.ndarray            # (K+1,) calibration sample count per arm
    cal_n_pos_per_arm: np.ndarray        # (K+1,) conv=1 calibration count per arm

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {
            "alpha_c": self.alpha_c,
            "alpha_v": self.alpha_v,
            "offset_conv": self.offset_conv,
            "offset_val_log": self.offset_val_log,
            "cal_n_per_arm": self.cal_n_per_arm,
            "cal_n_pos_per_arm": self.cal_n_pos_per_arm,
        }


def calibrate_joint_cp(
    model: FunnelCausalNet,
    info: dict,
    X_cal: np.ndarray,
    T_cal: np.ndarray,
    Y_cvr_cal: np.ndarray,
    Y_gmv_cal: np.ndarray,
    cfg: JointCPConfig,
    device: Optional[torch.device] = None,
) -> CalibratedJointCP:
    """Compute per-arm CQR offsets for conv head and OV head.

    Important: the per-arm split level for the τ_g interval (命题 3) is α/4
    (4 source intervals), but for marginal Y coverage (定理 1) only α_c / α_v
    are needed. We compute the conservative α/4 offsets directly so that
    downstream τ_g band has the §II.5 命题 3 guarantee.
    """
    qbands = predict_quantile_bounds(model, X_cal, info, device=device)
    K = model.cfg.num_arms
    Y_g_log1p = np.log1p(np.maximum(Y_gmv_cal, 0.0)).astype(np.float64)

    # For τ-band Bonferroni (命题 3): 4 source intervals → each gets α/4
    alpha_per_source_c = cfg.alpha_c / 2.0       # conv@t and conv@0 share α_c
    alpha_per_source_v = cfg.alpha_v / 2.0       # val@t and val@0 share α_v

    offset_conv = np.zeros(K, dtype=np.float64)
    offset_val_log = np.zeros(K, dtype=np.float64)
    cal_n = np.zeros(K, dtype=np.int64)
    cal_n_pos = np.zeros(K, dtype=np.int64)

    arange_N = np.arange(X_cal.shape[0])
    for arm in range(K):
        mask_arm = (T_cal == arm)
        cal_n[arm] = int(mask_arm.sum())
        if cal_n[arm] == 0:
            offset_conv[arm] = float("inf")
            offset_val_log[arm] = float("inf")
            continue

        # Conv head CQR offset
        ql_c = qbands["conv_lo"][arange_N, T_cal][mask_arm]
        qh_c = qbands["conv_hi"][arange_N, T_cal][mask_arm]
        y_c = Y_cvr_cal[mask_arm].astype(np.float64)
        offset_conv[arm] = _compute_arm_offset(
            ql_c, qh_c, y_c, alpha_per_source_c,
        )

        # OV head CQR offset on log1p scale, restricted to conv=1
        mask_pos = mask_arm & (Y_cvr_cal == 1)
        cal_n_pos[arm] = int(mask_pos.sum())
        if cal_n_pos[arm] > 0:
            ql_v = qbands["val_log1p_lo"][arange_N, T_cal][mask_pos]
            qh_v = qbands["val_log1p_hi"][arange_N, T_cal][mask_pos]
            y_v = Y_g_log1p[mask_pos]
            offset_val_log[arm] = _compute_arm_offset(
                ql_v, qh_v, y_v, alpha_per_source_v,
            )
        else:
            offset_val_log[arm] = float("inf")

    return CalibratedJointCP(
        alpha_c=cfg.alpha_c,
        alpha_v=cfg.alpha_v,
        offset_conv=offset_conv,
        offset_val_log=offset_val_log,
        cal_n_per_arm=cal_n,
        cal_n_pos_per_arm=cal_n_pos,
    )


# ---------------------------------------------------------------------------
# Inference: produce per-user joint intervals for tau_c and tau_g
# ---------------------------------------------------------------------------


def predict_joint_intervals(
    model: FunnelCausalNet,
    info: dict,
    cal: CalibratedJointCP,
    X: np.ndarray,
    device: Optional[torch.device] = None,
) -> Dict[str, np.ndarray]:
    """Apply calibrated offsets to test quantile bounds and synthesize τ bands.

    Returns dict (all (N, K+1) arrays unless noted):
        conv_lo, conv_hi             : conformalized CVR bounds per arm, ∈ [0, 1]
        val_lo, val_hi               : conformalized OV expectation bounds per arm
                                       (raw RMB scale, after expm1 + LogNormal correction)
        tau_c_lo, tau_c_hi           : τ_CVR bounds vs control arm (Bonferroni 命题 3)
        tau_g_lo, tau_g_hi           : τ_GMV bounds via interval arithmetic
                                       (§II.5 命题 3 envelope)
        width_c, width_g             : per-user (N, K+1) interval widths (for §II 第 4 项 C(K))
        mu_c_full, mu_g_full         : point estimates (passed through for downstream)
    """
    pred = predict_potentials(model, X, info, device=device)
    qbands = predict_quantile_bounds(model, X, info, device=device)
    K = model.cfg.num_arms
    N = X.shape[0]

    # ---- Conv head conformalized bounds (probability scale) ----
    conv_lo = np.clip(qbands["conv_lo"] - cal.offset_conv[None, :], 0.0, 1.0)
    conv_hi = np.clip(qbands["conv_hi"] + cal.offset_conv[None, :], 0.0, 1.0)

    # Defensive monotonicity: ensure lo ≤ hi after clipping
    swap = conv_lo > conv_hi
    conv_lo_fix = np.where(swap, conv_hi, conv_lo)
    conv_hi_fix = np.where(swap, conv_lo, conv_hi)
    conv_lo, conv_hi = conv_lo_fix, conv_hi_fix

    # ---- OV head conformalized bounds (log1p space → expm1 → expectation) ----
    log_sigma = float(model.log_sigma.detach().cpu().numpy())
    sigma_corr = 0.5 * (log_sigma ** 2)
    val_log_lo = qbands["val_log1p_lo"] - cal.offset_val_log[None, :]
    val_log_hi = qbands["val_log1p_hi"] + cal.offset_val_log[None, :]
    val_lo = np.clip(np.expm1(val_log_lo + sigma_corr), 0.0, None)
    val_hi = np.clip(np.expm1(val_log_hi + sigma_corr), 0.0, None)
    swap_v = val_lo > val_hi
    val_lo_fix = np.where(swap_v, val_hi, val_lo)
    val_hi_fix = np.where(swap_v, val_lo, val_hi)
    val_lo, val_hi = val_lo_fix, val_hi_fix

    # ---- τ_c bounds: τ_c^lo = conv_lo[t] - conv_hi[0], τ_c^hi = conv_hi[t] - conv_lo[0] ----
    tau_c_lo = conv_lo - conv_hi[:, [0]]
    tau_c_hi = conv_hi - conv_lo[:, [0]]
    tau_c_lo[:, 0] = 0.0; tau_c_hi[:, 0] = 0.0   # control arm has zero τ by definition

    # ---- τ_g bounds via interval arithmetic (§II.5 命题 3) ----
    # μ_g(x, t) = conv(x, t) · val(x, t); both ≥ 0.
    # τ_g^lo = lo[t] · lo[t] - hi[0] · hi[0], τ_g^hi = hi[t] · hi[t] - lo[0] · lo[0]
    g_lo_per_arm = conv_lo * val_lo                              # (N, K+1)
    g_hi_per_arm = conv_hi * val_hi                              # (N, K+1)
    tau_g_lo = g_lo_per_arm - g_hi_per_arm[:, [0]]
    tau_g_hi = g_hi_per_arm - g_lo_per_arm[:, [0]]
    tau_g_lo[:, 0] = 0.0; tau_g_hi[:, 0] = 0.0

    width_c = tau_c_hi - tau_c_lo
    width_g = tau_g_hi - tau_g_lo

    return {
        "conv_lo": conv_lo,
        "conv_hi": conv_hi,
        "val_lo": val_lo,
        "val_hi": val_hi,
        "tau_c_lo": tau_c_lo,
        "tau_c_hi": tau_c_hi,
        "tau_g_lo": tau_g_lo,
        "tau_g_hi": tau_g_hi,
        "width_c": width_c,
        "width_g": width_g,
        "mu_c_full": pred["mu_c_full"],
        "mu_g_full": pred["mu_g_full"],
    }


# ---------------------------------------------------------------------------
# Coverage diagnostics on a held-out test set
# ---------------------------------------------------------------------------


def empirical_coverage(
    intervals: Dict[str, np.ndarray],
    T_test: np.ndarray,
    Y_cvr_test: np.ndarray,
    Y_gmv_test: np.ndarray,
) -> Dict[str, float]:
    """Compute empirical marginal + joint coverage on the *factual* arm.

    For factual (T_obs) coverage check we look at conv@T and val@T bands
    (not the τ bands) since we only observe Y at one arm per user.
    """
    N = T_test.shape[0]
    arange_N = np.arange(N)

    p_lo = intervals["conv_lo"][arange_N, T_test]
    p_hi = intervals["conv_hi"][arange_N, T_test]
    cov_c = float(((Y_cvr_test >= p_lo - 1e-9) & (Y_cvr_test <= p_hi + 1e-9)).mean())

    v_lo = intervals["val_lo"][arange_N, T_test]
    v_hi = intervals["val_hi"][arange_N, T_test]
    mask_pos = (Y_cvr_test == 1)
    if mask_pos.any():
        cov_v_pos = float(
            ((Y_gmv_test[mask_pos] >= v_lo[mask_pos] - 1e-9)
             & (Y_gmv_test[mask_pos] <= v_hi[mask_pos] + 1e-9)).mean()
        )
    else:
        cov_v_pos = float("nan")

    # Joint coverage: conv interval covers AND (if conv=1) val interval covers.
    # OV is conditional on conv=1 by construction, so for conv=0 users only
    # the conv-coverage requirement applies. Bonferroni union 给出
    # P(both relevant intervals cover) ≥ 1 - α_c - α_v.
    conv_ok = (Y_cvr_test >= p_lo - 1e-9) & (Y_cvr_test <= p_hi + 1e-9)
    val_ok_when_pos = (Y_gmv_test >= v_lo - 1e-9) & (Y_gmv_test <= v_hi + 1e-9)
    joint_ok = conv_ok & np.where(Y_cvr_test == 1, val_ok_when_pos, True)
    cov_joint = float(joint_ok.mean())

    return {
        "cov_conv": cov_c,
        "cov_val_pos_only": cov_v_pos,
        "cov_joint": cov_joint,
    }
