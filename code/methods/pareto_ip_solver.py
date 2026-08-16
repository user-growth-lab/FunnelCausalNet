"""Multi-tier Pareto IP solver for §III 多档 Pareto 决策层 (创新点 III).

实现 §III.5 的形式化算法：
- LP 松弛 + 舍入（定理 3 / 推论 3.1，FPTAS 友好）—— scipy.linprog
- Lagrange 对偶 + λ 二分（定理 2 / 推论 2.2，强对偶 → 零间隙）—— 自实现 O(N·K·log(1/ε))
- λ-标量化扫 Pareto 前沿（定理 1 完备性）

支持：
- reward_mode ∈ {'point', 'lcb', 'topk_baseline'}
  对应"用点估计 / 不确定性下界 / 简单贪心 baseline"三种分配策略对比
  （'lcb' 对应 §III.5 命题 4：不确定性下界 Pareto 一致性）
- 约束类型：总预算 + per-arm 容量上限
  （per-arm 上限对应业务"高档不能发太多"约束）

数学问题（标准形式）：
    max_x  ∑_i ∑_k r_{i,k} · x_{i,k}                    (主目标)
    s.t.   ∑_k x_{i,k} = 1                ∀i           (每用户分恰好一档)
           ∑_i ∑_k cost_{i,k} · x_{i,k} ≤ B            (总预算)
           ∑_i x_{i,k} ≤ U_k             ∀k            (每档容量上限)
           x_{i,k} ∈ {0, 1}              (IP) or [0,1] (LP)

reward 矩阵 r_{i,k} 由 reward_mode 决定（见上）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class IPSolverConfig:
    """Hyper-parameters for Pareto IP solver."""

    # ----- 多目标加权 -----
    lambda_cvr: float = 0.0        # reward = τ_g + λ_cvr · τ_c (default 仅看 GMV)

    # ----- 预算约束 -----
    budget_total: Optional[float] = None        # 总预算 B；None 表示无限制
    arm_capacity: Optional[np.ndarray] = None   # (K+1,) 每档最大用户数；None 表示无限制

    # ----- Lagrange 二分搜索 -----
    lagrange_tol: float = 1e-3                  # |spent - B| / B 收敛阈值
    lagrange_max_iter: int = 60                 # log2(1e18) ≈ 60，远够用
    lagrange_lambda_max: float = 1e6

    # ----- 推荐策略 -----
    reward_mode: str = "point"                  # 'point' | 'lcb' | 'topk_baseline'


@dataclass
class AssignmentResult:
    """Output of any solver."""

    arms: np.ndarray               # (N,) int, 每用户的分配档位
    total_reward: float            # ∑_i r_{i, a_i}
    total_cost: float              # ∑_i cost_{i, a_i}
    arm_counts: np.ndarray         # (K+1,) int
    feasible: bool                 # 是否满足所有硬约束
    solver_time: float             # 求解耗时（秒）
    solver_name: str               # 'lagrange_bisect' | 'lp_relax_round' | 'topk_baseline'
    extra: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reward construction
# ---------------------------------------------------------------------------


def build_reward_matrix(
    tau_g: np.ndarray,                          # (N, K+1)
    tau_c: Optional[np.ndarray] = None,         # (N, K+1) optional
    tau_g_lo: Optional[np.ndarray] = None,      # (N, K+1) optional, for lcb mode
    cfg: Optional[IPSolverConfig] = None,
) -> np.ndarray:
    """根据 reward_mode 构造 reward 矩阵 r ∈ R^{N×(K+1)}。"""
    if cfg is None:
        cfg = IPSolverConfig()
    mode = cfg.reward_mode

    if mode == "point":
        r = tau_g.copy()
    elif mode == "lcb":
        if tau_g_lo is None:
            raise ValueError("reward_mode='lcb' requires tau_g_lo from joint CP layer")
        r = tau_g_lo.copy()
    elif mode == "topk_baseline":
        # baseline: 把 reward 退化为只关心 τ_g 而不联合 τ_c；topk_baseline 求解器
        # 会忽略 reward_matrix 直接按 max-arm-τ_g 排序选高档，这里返回点估计供
        # 求解器内部用作排序依据。
        r = tau_g.copy()
    else:
        raise ValueError(f"unknown reward_mode={mode!r}")

    if cfg.lambda_cvr != 0.0 and tau_c is not None and mode != "topk_baseline":
        r = r + cfg.lambda_cvr * tau_c

    return r


def build_cost_matrix(
    discount_per_arm: np.ndarray,               # (K+1,) e.g. [0, 0.02, ..., 0.14]
    expected_gmv_per_arm: np.ndarray,           # (N, K+1) e.g. mu_g_full from FCN
) -> np.ndarray:
    """每用户每档的期望 coupon 成本 = discount_k × E[GMV | x, t=k]。

    控制组（k=0, discount=0）成本恒为 0。
    """
    return discount_per_arm[None, :] * expected_gmv_per_arm


# ---------------------------------------------------------------------------
# Solver 1: TopK baseline (greedy, no joint optimization)
# ---------------------------------------------------------------------------


def solve_topk_baseline(
    reward: np.ndarray,             # (N, K+1)
    cost: np.ndarray,               # (N, K+1)
    cfg: IPSolverConfig,
) -> AssignmentResult:
    """Baseline: 每用户取 argmax_k r_{i,k}（无联合优化），然后用两步贪心修复：
    (1) 若设了 per-arm cap，对超 cap 的 arm 取该 arm 上 reward 最低的用户降到次优；
    (2) 若设了总预算，按 ROI = reward / cost 降序保留，剩余降回 k=0。

    与 Lagrange / LP 的对照点：本方法不做 reward × cost 的联合 trade-off，
    所以预算 binding 时会被严格 dominate（这正是 §III 创新点的 motivation）。
    """
    t0 = time.time()
    N, Kp1 = reward.shape
    arange_N = np.arange(N)
    arms = reward.argmax(axis=1)

    if cfg.arm_capacity is not None:
        for k in range(Kp1):
            cap = int(cfg.arm_capacity[k])
            sel = np.where(arms == k)[0]
            if len(sel) <= cap:
                continue
            scores = reward[sel, k]
            order = np.argsort(scores)
            drop = sel[order[: len(sel) - cap]]
            for i in drop:
                row = reward[i].copy()
                row[k] = -np.inf
                arms[i] = int(row.argmax())

    if cfg.budget_total is not None:
        cost_per_user = cost[arange_N, arms]
        rew_per_user = reward[arange_N, arms]
        if cost_per_user.sum() > cfg.budget_total:
            order = np.argsort(-rew_per_user / np.maximum(cost_per_user, 1e-9))
            cum = 0.0
            keep_mask = np.zeros(N, dtype=bool)
            for idx in order:
                if cum + cost_per_user[idx] <= cfg.budget_total + 1e-9:
                    keep_mask[idx] = True
                    cum += cost_per_user[idx]
            arms = np.where(keep_mask, arms, 0)

    arm_counts = np.bincount(arms, minlength=Kp1)
    final_cost = float(cost[arange_N, arms].sum())
    final_reward = float(reward[arange_N, arms].sum())
    feasible_budget = (cfg.budget_total is None
                       or final_cost <= cfg.budget_total + 1e-6)
    feasible_cap = (cfg.arm_capacity is None
                    or (arm_counts <= cfg.arm_capacity + 1e-6).all())
    return AssignmentResult(
        arms=arms, total_reward=final_reward, total_cost=final_cost,
        arm_counts=arm_counts, feasible=feasible_budget and feasible_cap,
        solver_time=time.time() - t0, solver_name="topk_baseline",
    )


# ---------------------------------------------------------------------------
# Solver 2: Lagrange dual + λ bisection
# ---------------------------------------------------------------------------


def _lagrange_oracle(
    reward: np.ndarray, cost: np.ndarray, lam: float,
    arm_capacity: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float]:
    """对每用户独立做 a_i = argmax_k (r_{i,k} - λ · cost_{i,k})。

    若有 per-arm 容量约束，做 fractional 修复（按 reduced cost 降序为 over-cap
    arm 取 top-U_k 用户保留，其他降一档至次优）。这是一阶启发式，对预算-only
    设置完全 exact（§III.5 定理 2），对带 per-arm cap 是工程近似。
    """
    N, Kp1 = reward.shape
    reduced = reward - lam * cost
    arms = reduced.argmax(axis=1)

    if arm_capacity is not None:
        for k in range(Kp1):
            cap = int(arm_capacity[k])
            sel = np.where(arms == k)[0]
            if len(sel) <= cap:
                continue
            scores = reduced[sel, k]
            order = np.argsort(-scores)
            keep = sel[order[:cap]]
            drop = sel[order[cap:]]
            for i in drop:
                row = reduced[i].copy()
                row[k] = -np.inf
                arms[i] = int(row.argmax())

    arange_N = np.arange(N)
    spent = float(cost[arange_N, arms].sum())
    rew = float(reward[arange_N, arms].sum())
    return arms, spent, rew


def solve_lagrange_bisect(
    reward: np.ndarray, cost: np.ndarray, cfg: IPSolverConfig,
) -> AssignmentResult:
    """Lagrange 对偶 + λ ≥ 0 二分搜索使 total_cost ≈ B。

    单调性：λ ↑ ⇒ reduced cost 中预算项权重 ↑ ⇒ 最优 arm 偏向低 cost ⇒
    spent ↓。所以 spent(λ) 是 λ 的非增阶梯函数，二分 well-defined。
    """
    t0 = time.time()
    if cfg.budget_total is None:
        # 无预算约束 → λ=0 即最优
        arms, spent, rew = _lagrange_oracle(reward, cost, 0.0, cfg.arm_capacity)
        Kp1 = reward.shape[1]
        return AssignmentResult(
            arms=arms, total_reward=rew, total_cost=spent,
            arm_counts=np.bincount(arms, minlength=Kp1), feasible=True,
            solver_time=time.time() - t0, solver_name="lagrange_bisect",
            extra={"lambda_star": 0.0, "n_iter": 0},
        )

    B = cfg.budget_total

    # λ = 0：花最贵；如果还在预算内说明无约束最优本身就可行
    arms_lo, spent_lo, rew_lo = _lagrange_oracle(reward, cost, 0.0, cfg.arm_capacity)
    if spent_lo <= B + cfg.lagrange_tol * max(B, 1.0):
        Kp1 = reward.shape[1]
        return AssignmentResult(
            arms=arms_lo, total_reward=rew_lo, total_cost=spent_lo,
            arm_counts=np.bincount(arms_lo, minlength=Kp1),
            feasible=True, solver_time=time.time() - t0,
            solver_name="lagrange_bisect",
            extra={"lambda_star": 0.0, "n_iter": 0,
                   "note": "unconstrained optimum already within budget"},
        )

    lo, hi = 0.0, cfg.lagrange_lambda_max
    arms_best = arms_lo; spent_best = spent_lo; rew_best = rew_lo
    lam_star = 0.0
    n_iter = 0
    for it in range(cfg.lagrange_max_iter):
        mid = 0.5 * (lo + hi)
        arms_m, spent_m, rew_m = _lagrange_oracle(reward, cost, mid, cfg.arm_capacity)
        n_iter = it + 1
        if spent_m > B:
            lo = mid                           # 还是超支，需要更大 λ
        else:
            arms_best = arms_m; spent_best = spent_m; rew_best = rew_m; lam_star = mid
            hi = mid                           # 已可行，收紧
        if abs(hi - lo) < cfg.lagrange_tol * max(1.0, lam_star + 1.0):
            break

    Kp1 = reward.shape[1]
    return AssignmentResult(
        arms=arms_best, total_reward=rew_best, total_cost=spent_best,
        arm_counts=np.bincount(arms_best, minlength=Kp1),
        feasible=(spent_best <= B + cfg.lagrange_tol * max(B, 1.0)),
        solver_time=time.time() - t0, solver_name="lagrange_bisect",
        extra={"lambda_star": lam_star, "n_iter": n_iter,
               "lo": lo, "hi": hi},
    )


# ---------------------------------------------------------------------------
# Solver 3: LP relaxation + rounding (scipy.linprog)
# ---------------------------------------------------------------------------


def solve_lp_relax_round(
    reward: np.ndarray, cost: np.ndarray, cfg: IPSolverConfig,
    method: str = "highs",
) -> AssignmentResult:
    """LP 松弛 + 舍入。对应 §III.5 定理 3：rounding gap ≤ K · max_k cost_k.

    LP 形式：max c^T x  s.t.  A_eq x = b_eq, A_ub x ≤ b_ub, 0 ≤ x ≤ 1.
    变量 x 维度 = N · (K+1) 平摊向量，按 (i, k) → i*(K+1)+k 编号。

    舍入策略：对每个用户 i，取 argmax_k x_{i,k}^*。可能造成预算 / cap 轻微违反，
    再做一次贪心 swap 修复（降级溢出 arm 上 marginal reward 最低的用户）。
    """
    t0 = time.time()
    N, Kp1 = reward.shape
    nv = N * Kp1

    c_obj = -reward.flatten()                  # linprog 是 minimize

    # ∑_k x_{i,k} = 1
    A_eq = np.zeros((N, nv), dtype=np.float64)
    for i in range(N):
        A_eq[i, i * Kp1:(i + 1) * Kp1] = 1.0
    b_eq = np.ones(N, dtype=np.float64)

    A_ub_rows = []
    b_ub_rows = []
    if cfg.budget_total is not None:
        A_ub_rows.append(cost.flatten())
        b_ub_rows.append(cfg.budget_total)
    if cfg.arm_capacity is not None:
        for k in range(Kp1):
            row = np.zeros(nv, dtype=np.float64)
            row[k::Kp1] = 1.0
            A_ub_rows.append(row)
            b_ub_rows.append(float(cfg.arm_capacity[k]))
    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_rows) if b_ub_rows else None

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0.0, 1.0)] * nv, method=method)
    if not res.success:
        return AssignmentResult(
            arms=np.zeros(N, dtype=np.int64), total_reward=0.0, total_cost=0.0,
            arm_counts=np.bincount(np.zeros(N, dtype=np.int64), minlength=Kp1),
            feasible=False, solver_time=time.time() - t0, solver_name="lp_relax_round",
            extra={"linprog_status": res.message},
        )

    X = res.x.reshape(N, Kp1)
    arms = X.argmax(axis=1)
    arange_N = np.arange(N)

    if cfg.budget_total is not None:
        cost_per_user = cost[arange_N, arms]
        spent = float(cost_per_user.sum())
        if spent > cfg.budget_total:
            order = np.argsort(cost_per_user - reward[arange_N, arms])  # 牺牲少→优先降
            for i in order[::-1]:
                if spent <= cfg.budget_total:
                    break
                if arms[i] == 0:
                    continue
                spent -= cost[i, arms[i]]
                arms[i] = 0
                spent += cost[i, 0]

    if cfg.arm_capacity is not None:
        for k in range(Kp1):
            sel = np.where(arms == k)[0]
            if len(sel) <= int(cfg.arm_capacity[k]):
                continue
            scores = reward[sel, k] - cost[sel, k]
            order = np.argsort(scores)
            drop = sel[order[: len(sel) - int(cfg.arm_capacity[k])]]
            for i in drop:
                row = reward[i].copy()
                if cfg.budget_total is not None:
                    pass
                row[k] = -np.inf
                arms[i] = int(row.argmax())

    spent_final = float(cost[arange_N, arms].sum())
    rew_final = float(reward[arange_N, arms].sum())
    feasible_budget = (cfg.budget_total is None
                       or spent_final <= cfg.budget_total + 1e-6)
    arm_counts = np.bincount(arms, minlength=Kp1)
    feasible_cap = (cfg.arm_capacity is None
                    or (arm_counts <= cfg.arm_capacity + 1e-6).all())
    return AssignmentResult(
        arms=arms, total_reward=rew_final, total_cost=spent_final,
        arm_counts=arm_counts, feasible=feasible_budget and feasible_cap,
        solver_time=time.time() - t0, solver_name="lp_relax_round",
        extra={"lp_objective": -res.fun, "linprog_status": res.message},
    )


# ---------------------------------------------------------------------------
# Pareto frontier scan
# ---------------------------------------------------------------------------


@dataclass
class ParetoPoint:
    lam: float
    total_tau_c: float
    total_tau_g: float
    total_cost: float
    arms: np.ndarray
    solver_extra: Dict


def scan_pareto_frontier(
    tau_g: np.ndarray, tau_c: np.ndarray, cost: np.ndarray,
    cfg: IPSolverConfig,
    lambda_grid: Optional[np.ndarray] = None,
    solver: str = "lagrange_bisect",
) -> List[ParetoPoint]:
    """对 λ_cvr ∈ lambda_grid 做 reward = τ_g + λ_cvr·τ_c 的标量化扫描，
    每个 λ 解一个 IP 子问题，得到 (Total τ_c, Total τ_g) 平面上的 Pareto 候选点。

    §III.5 定理 1 保证：当 λ 遍历 [0, ∞] 时，所得解集即 Pareto 前沿
    （凸包补强后是完备的）。
    """
    if lambda_grid is None:
        lambda_grid = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
    arange_N = np.arange(tau_g.shape[0])

    points: List[ParetoPoint] = []
    for lam in lambda_grid:
        sub_cfg = IPSolverConfig(
            lambda_cvr=float(lam),
            budget_total=cfg.budget_total,
            arm_capacity=cfg.arm_capacity,
            lagrange_tol=cfg.lagrange_tol,
            lagrange_max_iter=cfg.lagrange_max_iter,
            lagrange_lambda_max=cfg.lagrange_lambda_max,
            reward_mode=cfg.reward_mode,
        )
        reward = build_reward_matrix(tau_g, tau_c, cfg=sub_cfg)
        if solver == "lagrange_bisect":
            res = solve_lagrange_bisect(reward, cost, sub_cfg)
        elif solver == "lp_relax_round":
            res = solve_lp_relax_round(reward, cost, sub_cfg)
        else:
            raise ValueError(f"unknown solver={solver!r}")

        total_tau_c = float(tau_c[arange_N, res.arms].sum())
        total_tau_g = float(tau_g[arange_N, res.arms].sum())
        points.append(ParetoPoint(
            lam=float(lam),
            total_tau_c=total_tau_c, total_tau_g=total_tau_g,
            total_cost=res.total_cost, arms=res.arms,
            solver_extra=res.extra,
        ))
    return points
