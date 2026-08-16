"""E4 多档 Pareto 决策层 pipeline (创新点 III 应用层).

按 §5.1 E4 设计：在 Criteo-MT7 8 档 + 真实 ground-truth 反事实下，对比四种分配
策略在「等预算 ATE_GMV」与「Pareto 前沿覆盖面积」两类指标上的表现：

策略对比：
- baseline_topk      ：按 τ̂_g(x, max_arm) 降序，预算内贪心选 max-arm
- funnel_ip_point    ：本方法 — FunnelCausalNet + Lagrange 二分（reward=τ̂_g 点估计）
- funnel_ip_lp       ：本方法 — 同上但用 LP relaxation + 舍入（避免 λ 二分离散跳跃）
- funnel_ip_lcb      ：本方法 — 同 lagrange 但 reward=τ_g_lo 不确定性下界 (§III.5 命题 4)
- baseline_random    ：随机均匀分配 8 档（控制 baseline，受 budget 约束截断）

评估：在测试集上对每个策略求出 arms[i]，然后用 ground-truth tau_gmv[i, arms[i]]
求总 GMV 增量（这是用反事实 ITE 算出的真实收益，不是预测值）。

Pareto 前沿：扫 λ_cvr ∈ [0, 100] 在 funnel_ip_point 上得到 (Total τ_c, Total τ_g)
平面上的曲线，与 baseline_topk 的单点解作对比。

输出：
    results/e4/e4_pareto_<timestamp>.csv  长表
    results/e4/e4_pareto_<timestamp>_summary.csv 按 (budget, strategy) groupby
    results/e4/e4_pareto_<timestamp>.json
    results/e4/pareto_frontier_<timestamp>.csv  λ-扫描的 Pareto 点

用法：
    python3 code/experiments/run_e4_pareto.py
    python3 code/experiments/run_e4_pareto.py --quick    # N=5K, 1 seed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "code"))

from methods.funnel_causal_net import (  # noqa: E402
    FunnelArchConfig,
    FunnelLossConfig,
    FunnelTrainConfig,
    predict_potentials,
    train_funnel_net,
)
from methods.joint_conformal_cate import (  # noqa: E402
    JointCPConfig,
    calibrate_joint_cp,
    predict_joint_intervals,
)
from methods.pareto_ip_solver import (  # noqa: E402
    IPSolverConfig,
    build_cost_matrix,
    build_reward_matrix,
    scan_pareto_frontier,
    solve_lagrange_bisect,
    solve_lp_relax_round,
    solve_topk_baseline,
)
from semisynth.criteo_mt7_generator import (  # noqa: E402
    GenConfig,
    NUM_ARMS,
    PRESETS,
    _load_criteo_X,
    generate,
)


DISCOUNT_GRID = np.linspace(0.0, 0.14, NUM_ARMS).astype(np.float64)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_bundle(N: int, seed: int, anticorr: float = 0.6) -> dict:
    cfg = GenConfig(n_samples=N, seed=seed)
    for k, v in PRESETS["realistic"].items():
        setattr(cfg, k, v)
    cfg.conflict_anticorr = anticorr
    csv_path = _PROJECT_ROOT / "data" / "criteo-uplift" / "criteo-uplift-v2.1.csv.gz"
    X = _load_criteo_X(csv_path, N, seed)
    return generate(cfg, X)


def _train_full_pipeline(bundle: dict, seed: int, max_epochs: int):
    """训 FunnelCausalNet (hard + quantile heads) + Conformal calibration。"""
    X = bundle["X"]; T = bundle["T"]
    Y_cvr = bundle["Y_cvr"]; Y_gmv = bundle["Y_gmv"]
    N = X.shape[0]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_te = int(N * 0.2); n_cal = int(N * 0.2)
    tr = idx[:N - n_te - n_cal]
    cal = idx[N - n_te - n_cal:N - n_te]
    te = idx[N - n_te:]

    # quantile_lo=0.20 而非 0.05：零膨胀 GMV 下 5% 分位下界几乎全负，会让
    # LCB-mode 的 reward 全 ≤ 0，IP solver 退化为全员 control（无意义）。
    # §III.5 命题 4 未规定具体 alpha，这里取 80% 单边 LCB（双边 60%）作为
    # 「鲁棒但不过度悲观」的工程默认。
    arch = FunnelArchConfig(
        d_in=X.shape[1], num_arms=NUM_ARMS,
        rep_hidden=[128, 64], rep_dim=32, head_hidden=[32],
        dropout=0.1, use_anchor=True, learn_log_sigma=True,
        quantile_heads=True, quantile_lo=0.20, quantile_hi=0.80,
        pinball_weight=0.5,
    )
    loss = FunnelLossConfig(alpha=0.3, beta=1.0, gamma=0.5,
                            funnel_mode="hard")
    train_cfg = FunnelTrainConfig(lr=1e-3, batch_size=512, max_epochs=max_epochs,
                                  patience=8, seed=seed, verbose=False)

    model, info = train_funnel_net(X[tr], T[tr], Y_cvr[tr], Y_gmv[tr],
                                   arch, loss, train_cfg)
    cp_cfg = JointCPConfig(alpha_c=0.05, alpha_v=0.05)
    cal_obj = calibrate_joint_cp(model, info, X[cal], T[cal],
                                 Y_cvr[cal], Y_gmv[cal], cp_cfg)
    intervals = predict_joint_intervals(model, info, cal_obj, X[te])

    # 用 predict_potentials 出 ranking 中心点（条件期望），避免 quantile interval
    # 中点对零膨胀重尾数据的 ranking 信号污染。CP intervals 仍保留用于 LCB 决策。
    pred_center = predict_potentials(model, X[te], info, funnel_mode="hard")
    tau_c_pred = pred_center["tau_c"]               # (n_te, NUM_ARMS)
    tau_g_pred = pred_center["tau_g"]               # (n_te, NUM_ARMS)
    tau_g_lo = intervals["tau_g_lo"]                # CP 下界仍保留供 LCB 策略
    mu_g_full = pred_center["mu_g_full"]

    cost = build_cost_matrix(DISCOUNT_GRID, mu_g_full)

    # RCT per-arm sample mean of Y_gmv (post-hoc calibration anchor)。
    # 在 train data 上按 T 分组取均值，作为 funnel net 预测 μ_g per arm 的
    # 「绝对水平」无偏锚点。funnel hard mode 的 multiplicative composition 在
    # zero-inflated control arm 上系统性低估 μ_g[control]（实测 -41%），
    # 导致下游 IP solver 的 ROI 排序偏向低档。anchored 策略用此 RCT 均值
    # 做加性 shift 校正绝对水平，保留 funnel 学到的 user-级 heterogeneity。
    rct_per_arm_mean = np.array([
        float(Y_gmv[tr][T[tr] == k].mean()) if (T[tr] == k).any() else 0.0
        for k in range(NUM_ARMS)
    ])

    return {
        "te": te, "tau_c_pred": tau_c_pred, "tau_g_pred": tau_g_pred,
        "tau_g_lo": tau_g_lo, "mu_g_full": mu_g_full, "cost": cost,
        "rct_per_arm_mean": rct_per_arm_mean,
        "intervals": intervals, "model": model, "info": info,
    }


def _evaluate_strategy_on_gt(arms: np.ndarray, tau_g_gt_te: np.ndarray,
                              tau_c_gt_te: np.ndarray,
                              cost_per_user_per_arm: np.ndarray) -> Dict:
    """在测试集上用 ground-truth ITE 计算分配策略的真实收益。

    arms: (n_te,) 每用户分配的档位 (0..K)
    tau_g_gt_te / tau_c_gt_te: (n_te, K+1) ground-truth ITE per arm
    cost_per_user_per_arm: (n_te, K+1) 期望成本（已经内嵌折扣率 × E[GMV]）
    """
    n_te = len(arms)
    arange = np.arange(n_te)
    realized_tau_g = float(tau_g_gt_te[arange, arms].sum())
    realized_tau_c = float(tau_c_gt_te[arange, arms].sum())
    realized_cost = float(cost_per_user_per_arm[arange, arms].sum())
    arm_dist = np.bincount(arms, minlength=NUM_ARMS).tolist()
    return {
        "true_total_tau_g": realized_tau_g,
        "true_total_tau_c": realized_tau_c,
        "realized_cost": realized_cost,
        "arm_dist": arm_dist,
        "frac_max_arm": float((arms == NUM_ARMS - 1).mean()),
        "frac_control": float((arms == 0).mean()),
    }


# ---------------------------------------------------------------------------
# Per-trial run
# ---------------------------------------------------------------------------


def _run_trial(N: int, seed: int, budget_fracs: List[float],
               max_epochs: int) -> List[Dict]:
    bundle = _build_bundle(N, seed)
    pipe = _train_full_pipeline(bundle, seed, max_epochs)
    te = pipe["te"]
    tau_c_pred = pipe["tau_c_pred"]; tau_g_pred = pipe["tau_g_pred"]
    tau_g_lo = pipe["tau_g_lo"]
    cost = pipe["cost"]
    mu_g_full = pipe["mu_g_full"]
    rct_per_arm_mean = pipe["rct_per_arm_mean"]

    # Anchored μ_g：加性 shift 校正每 arm 的绝对水平（保留 user 级 heterogeneity）。
    # mu_g_anchored[i, k] = mu_g_pred[i, k] + (rct_mean[k] - funnel_mean[k])
    funnel_per_arm_mean = mu_g_full.mean(axis=0)
    arm_shift = rct_per_arm_mean - funnel_per_arm_mean
    mu_g_anchored = mu_g_full + arm_shift[None, :]
    mu_g_anchored = np.clip(mu_g_anchored, 0.0, None)
    tau_g_anchored = mu_g_anchored - mu_g_anchored[:, [0]]
    cost_anchored = build_cost_matrix(DISCOUNT_GRID, mu_g_anchored)

    tau_c_gt_te = bundle["tau_cvr"][te]
    tau_g_gt_te = bundle["tau_gmv"][te]

    # 评估专用的 cost matrix（用 ground-truth μ_g 算 discount × E[GMV]），
    # 让所有策略的 realized_cost 在同一物理尺度上可比；solver 输入仍用各自
    # 模型预测的 cost，不破坏 solver 的优化前提。
    mu_g_gt_te = bundle["mu_gmv_full"][te]
    cost_eval = build_cost_matrix(DISCOUNT_GRID, mu_g_gt_te)

    # free_cost = "全员发 max-arm" 的总成本；budget 用 free_cost 的比例定义，
    # 与模型预测无关。比 .argmax(1) 更稳定，且公平地对所有策略适用。
    free_cost = float(cost[:, -1].sum())
    free_cost_anchored = float(cost_anchored[:, -1].sum())
    rng = np.random.default_rng(seed)

    rows = []
    for budget_frac in budget_fracs:
        B = float(free_cost * budget_frac)
        # 不设 per-arm cap，仅总预算（让 baseline 与 Funnel+IP 公平）
        cfg_pt = IPSolverConfig(budget_total=B, reward_mode="point",
                                lambda_cvr=0.0)
        cfg_lcb = IPSolverConfig(budget_total=B, reward_mode="lcb",
                                 lambda_cvr=0.0)

        # Strategy 1: TopK baseline (greedy on tau_g_pred, no joint optim)
        rew_pt = build_reward_matrix(tau_g_pred, tau_c_pred, cfg=cfg_pt)
        res_topk = solve_topk_baseline(rew_pt, cost, cfg_pt)

        # Strategy 2: Funnel + IP (point, Lagrange)
        res_ip_pt = solve_lagrange_bisect(rew_pt, cost, cfg_pt)

        # Strategy 2b: Funnel + IP (point, LP relaxation + rounding)
        # 与 lagrange 同 reward / cfg，仅替换 solver；LP 在 budget 紧约束下可
        # 生成 user 间「混合策略」（一些 max-arm + 一些 control），而 Lagrange
        # 二分搜索在某些 λ 下让 user 集体跳到中间档，导致次优。
        res_ip_lp = solve_lp_relax_round(rew_pt, cost, cfg_pt)

        # Strategy 2c: Funnel + IP + RCT-anchored μ_g (post-hoc calibration)
        # 用 train RCT 的 per-arm 均值做加性 shift，校正 funnel net 在 zero-
        # inflated control 上的系统性 underestimate（实测控制组 μ_g 偏低 41%）。
        # 用 anchored cost 而不是原 cost 来定义预算，确保 free_cost 也用同一
        # μ_g 尺度，使 budget_frac 含义一致。
        B_anchored = float(free_cost_anchored * budget_frac)
        cfg_anchored = IPSolverConfig(budget_total=B_anchored,
                                       reward_mode="point", lambda_cvr=0.0)
        rew_anchored = build_reward_matrix(tau_g_anchored, tau_c_pred,
                                            cfg=cfg_anchored)
        res_ip_anchored = solve_lp_relax_round(rew_anchored, cost_anchored,
                                                cfg_anchored)

        # Strategy 3: Funnel + IP (LCB)
        rew_lcb = build_reward_matrix(tau_g_pred, tau_c_pred,
                                      tau_g_lo=tau_g_lo, cfg=cfg_lcb)
        res_ip_lcb = solve_lagrange_bisect(rew_lcb, cost, cfg_lcb)

        # Strategy 4: Random uniform with budget enforcement (公平性补丁)。
        # 先随机分配 K+1 档，若总成本 > B，按 cost 降序回滚最贵的用户至 control，
        # 直到满足预算。这样 random 与其他策略一样严格遵守 hard budget。
        arms_random = rng.integers(0, NUM_ARMS, size=len(te))
        arange_te = np.arange(len(te))
        cost_random = cost[arange_te, arms_random]
        spent = float(cost_random.sum())
        if spent > B:
            # 按 cost 从高到低回滚到 control，直到 ≤ B
            order = np.argsort(-cost_random)
            running = spent
            for idx in order:
                if running <= B:
                    break
                running -= float(cost_random[idx])
                arms_random[idx] = 0
                cost_random[idx] = 0.0

        for strat_name, arms_strat, solver_time in [
            ("baseline_topk", res_topk.arms, res_topk.solver_time),
            ("funnel_ip_point", res_ip_pt.arms, res_ip_pt.solver_time),
            ("funnel_ip_lp", res_ip_lp.arms, res_ip_lp.solver_time),
            ("funnel_ip_anchored", res_ip_anchored.arms, res_ip_anchored.solver_time),
            ("funnel_ip_lcb", res_ip_lcb.arms, res_ip_lcb.solver_time),
            ("baseline_random", arms_random, 0.0),
        ]:
            # 用 ground-truth μ_g 的 cost_eval 统一计算 realized_cost，让不同
            # 策略在同一物理尺度可比（避免 anchored 用大 cost 矩阵看似花更多）。
            ev = _evaluate_strategy_on_gt(arms_strat, tau_g_gt_te,
                                           tau_c_gt_te, cost_eval)
            rows.append({
                "N": N, "seed": seed, "budget_frac": budget_frac, "B": B,
                "strategy": strat_name,
                "true_total_tau_g": ev["true_total_tau_g"],
                "true_total_tau_c": ev["true_total_tau_c"],
                "realized_cost": ev["realized_cost"],
                "frac_max_arm": ev["frac_max_arm"],
                "frac_control": ev["frac_control"],
                "arm_dist": ev["arm_dist"],
                "solver_time": solver_time,
            })
    return rows


# ---------------------------------------------------------------------------
# Pareto frontier scan
# ---------------------------------------------------------------------------


def _run_pareto_scan(N: int, seed: int, budget_frac: float,
                     max_epochs: int) -> List[Dict]:
    bundle = _build_bundle(N, seed)
    pipe = _train_full_pipeline(bundle, seed, max_epochs)
    te = pipe["te"]
    tau_c_pred = pipe["tau_c_pred"]; tau_g_pred = pipe["tau_g_pred"]
    cost = pipe["cost"]

    tau_c_gt_te = bundle["tau_cvr"][te]
    tau_g_gt_te = bundle["tau_gmv"][te]

    free_cost = float(cost[:, -1].sum())
    B = float(free_cost * budget_frac)

    cfg = IPSolverConfig(budget_total=B, reward_mode="point", lambda_cvr=0.0)
    lam_grid = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    points = scan_pareto_frontier(tau_g_pred, tau_c_pred, cost, cfg,
                                   lambda_grid=lam_grid,
                                   solver="lagrange_bisect")

    rows = []
    for p in points:
        arange = np.arange(len(te))
        true_tau_g = float(tau_g_gt_te[arange, p.arms].sum())
        true_tau_c = float(tau_c_gt_te[arange, p.arms].sum())
        rows.append({
            "N": N, "seed": seed, "budget_frac": budget_frac, "B": B,
            "lambda_cvr": p.lam,
            "pred_total_tau_c": p.total_tau_c,
            "pred_total_tau_g": p.total_tau_g,
            "true_total_tau_c": true_tau_c,
            "true_total_tau_g": true_tau_g,
            "realized_cost": p.total_cost,
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="N=5K, 1 seed, single budget_frac=0.5")
    ap.add_argument("--N", type=int, default=20_000)
    ap.add_argument("--max_epochs", type=int, default=30)
    args = ap.parse_args()

    if args.quick:
        N = 5_000; seeds = [0]; budget_fracs = [0.10]
    else:
        # 紧预算 grid：让 IP 必须做精细选择，预算不能松到任何 user 都能拿 max-arm。
        # 旧版 [0.2, 0.5, 0.8, 1.0] 利用率只有 41-65%，IP 退化为 unconstrained TopK。
        N = args.N; seeds = [0, 1, 2]; budget_fracs = [0.05, 0.10, 0.20, 0.50]

    print(f"== E4 Pareto decision runner ({'quick' if args.quick else 'default'}) ==")
    print(f"  N: {N},  seeds: {seeds},  budget_fracs: {budget_fracs}")
    print(f"  strategies: baseline_topk, funnel_ip_point, funnel_ip_lcb, baseline_random")
    print(f"  max_epochs: {args.max_epochs}\n")

    all_rows = []
    pareto_rows = []
    n_total = len(seeds)
    t_start = time.time()
    for trial_idx, seed in enumerate(seeds, 1):
        t0 = time.time()
        rows = _run_trial(N, seed, budget_fracs, args.max_epochs)
        all_rows.extend(rows)

        # Pareto scan: 仅在 budget_frac=0.5 上做（cost & 时间）
        prows = _run_pareto_scan(N, seed, 0.5, args.max_epochs)
        pareto_rows.extend(prows)

        dt = time.time() - t0
        eta = (time.time() - t_start) / trial_idx * (n_total - trial_idx)
        print(f"  [{trial_idx}/{n_total}] N={N} seed={seed}: trial+pareto done in {dt:.1f}s  "
              f"ETA={eta/60:.1f}min")

    df = pd.DataFrame(all_rows)
    df_pareto = pd.DataFrame(pareto_rows)

    summary = df.groupby(["budget_frac", "strategy"], as_index=False).agg(
        true_tau_g_mean=("true_total_tau_g", "mean"),
        true_tau_g_std=("true_total_tau_g", "std"),
        true_tau_c_mean=("true_total_tau_c", "mean"),
        realized_cost_mean=("realized_cost", "mean"),
        frac_max_arm_mean=("frac_max_arm", "mean"),
        frac_control_mean=("frac_control", "mean"),
        solver_time_mean=("solver_time", "mean"),
    )

    out_dir = _PROJECT_ROOT / "results" / "e4"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    long_csv = out_dir / f"e4_pareto_{ts}.csv"
    summary_csv = out_dir / f"e4_pareto_{ts}_summary.csv"
    pareto_csv = out_dir / f"pareto_frontier_{ts}.csv"
    json_path = out_dir / f"e4_pareto_{ts}.json"

    df.to_csv(long_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    df_pareto.to_csv(pareto_csv, index=False)
    json_path.write_text(json.dumps({
        "config": {"N": N, "seeds": seeds, "budget_fracs": budget_fracs,
                   "max_epochs": args.max_epochs},
        "rows": all_rows,
        "summary": summary.to_dict(orient="records"),
        "pareto_rows": pareto_rows,
    }, indent=2, ensure_ascii=False))

    print(f"\n== Summary ==\n{summary.to_string(index=False)}")

    print("\n[verify §III ATE_GMV @ equal-budget: funnel_ip_* vs baselines]")
    print("  (cost columns use ground-truth μ_g for fair cross-strategy comparison)")
    for budget_frac in budget_fracs:
        sub = summary[summary["budget_frac"] == budget_frac].set_index("strategy")

        def _g(name):
            return sub.loc[name, "true_tau_g_mean"] if name in sub.index else float("nan")
        def _c(name):
            return sub.loc[name, "realized_cost_mean"] if name in sub.index else float("nan")

        anc = _g("funnel_ip_anchored"); lp = _g("funnel_ip_lp")
        pt = _g("funnel_ip_point"); topk = _g("baseline_topk")
        rnd = _g("baseline_random")
        anc_c = _c("funnel_ip_anchored"); rnd_c = _c("baseline_random")

        gain_anc_vs_rnd = (anc - rnd) / max(abs(rnd), 1e-6) * 100
        verdict_anc = "✓" if anc > rnd else "✗"
        roi_anc = anc / max(anc_c, 1e-6); roi_rnd = rnd / max(rnd_c, 1e-6)

        print(f"  budget={budget_frac:.2f}:  "
              f"anchored={anc:8.1f}/{anc_c:7.0f} (ROI={roi_anc:.2f})  "
              f"lp={lp:8.1f}  topk={topk:8.1f}  "
              f"random={rnd:8.1f}/{rnd_c:7.0f} (ROI={roi_rnd:.2f})  "
              f"anc_vs_rnd={gain_anc_vs_rnd:+5.1f}%  {verdict_anc}")

    print(f"\n  long-form CSV    : {long_csv}")
    print(f"  summary CSV      : {summary_csv}")
    print(f"  Pareto CSV       : {pareto_csv}")
    print(f"  JSON dump        : {json_path}")
    print(f"\n  total wall-clock : {(time.time() - t_start)/60:.2f} min")


if __name__ == "__main__":
    main()
