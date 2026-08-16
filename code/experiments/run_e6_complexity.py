"""E6 计算开销与可扩展性 runner（创新点 III §III.5 (f) 复杂度表实证）。

测量 4 项 wall-clock time 随 (样本量 N, 档位数 K) 的扩展性：
- (1) FunnelCausalNet 训练时间（含 early stopping）
- (2) 推理时间 (predict_potentials, batched forward)
- (3) Conformal split-CP 标定时间 (calibrate_joint_cp + predict_joint_intervals)
- (4) IP solver 求解时间（TopK baseline / Lagrange bisect / LP relaxation 三种）

设计取舍：
- mt7 generator 硬编码 8 档，因此 (1)/(2)/(3) 训练-推理-标定均在 K=8 下测；
- K 维度仅作用于 (4) IP solver——通过 stride 在 8 个 arm 中抽 K+1 档子集
  （K=2: arms [0,7]; K=4: arms [0,2,4,7]; K=8: 全部），用于实证 §III.5 (f)
  对 IP solver 给出的 O(N·K·log(1/ε)) 复杂度。
- 每个 (N, seed) super-trial 共享同一 bundle 与 funnel net，最大化 IO/训练复用。
- LP relaxation 在 N≥200K 时主动 skip（A_eq 矩阵 N × N(K+1) 致 OOM），仅用 N≤100K
  验证 §III.5 (e) 定理 3 LP-rounding gap 与 §III.5 定理 2 Lagrange 强对偶的对照。

输出：
    results/e6/e6_complexity_<ts>.csv          long form: (N, K, seed, component, time_s, extra)
    results/e6/e6_complexity_<ts>_summary.csv  groupby (N, K, component) mean ± std
    results/e6/e6_complexity_<ts>.json         snapshot

用法：
    python3 code/experiments/run_e6_complexity.py --quick   # N=10K × K=2 × 1 seed
    python3 code/experiments/run_e6_complexity.py           # default
    python3 code/experiments/run_e6_complexity.py --paper   # 5 N × 3 K × 3 seeds = 45 trials

需要按 ``data/README.md`` 准备公开 Criteo 特征文件；outcomes 与 treatments
由半合成生成器构造。全量特征载入约需 1.4GB 内存，计时结果依赖运行环境。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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
    solve_lagrange_bisect,
    solve_lp_relax_round,
    solve_topk_baseline,
)
from semisynth.criteo_mt7_generator import (  # noqa: E402
    CRITEO_FEATURE_COLS,
    GenConfig,
    NUM_ARMS,
    PRESETS,
    generate,
)


# ---------------------------------------------------------------------------
# Experiment grids
# ---------------------------------------------------------------------------

DISCOUNT_GRID = np.linspace(0.0, 0.14, NUM_ARMS).astype(np.float64)

# K 维度通过 arm 子集模拟（仅作用于 IP solver）。
# K=2 → control + max-arm；K=4 → 4 等距 arm；K=8 → 全 8 档。
ARM_SUBSETS: Dict[int, np.ndarray] = {
    2: np.array([0, NUM_ARMS - 1], dtype=np.int64),
    4: np.array([0, 2, 4, NUM_ARMS - 1], dtype=np.int64),
    8: np.arange(NUM_ARMS, dtype=np.int64),
}

# 当 N ≥ LP_SKIP_N 时 skip LP relaxation（避免 N(K+1) × N(K+1) 稠密矩阵 OOM）。
LP_SKIP_N = 200_000

# IP solver 测试用的预算占比（mid-range，对应 §5.5(c) E4 budget=0.20）。
BUDGET_FRAC = 0.20

SCALE_GRIDS: Dict[str, Dict] = {
    "quick": {
        "sample_sizes": [10_000],
        "K_list": [2],
        "seeds": [0],
    },
    "default": {
        "sample_sizes": [10_000, 50_000, 100_000],
        "K_list": [2, 4, 8],
        "seeds": [0],
    },
    "paper": {
        "sample_sizes": [10_000, 50_000, 100_000, 500_000, 1_000_000],
        "K_list": [2, 4, 8],
        "seeds": [0, 1, 2],
    },
}


# ---------------------------------------------------------------------------
# Cached Criteo X loader (跨 super-trial 复用 14M-row CSV，省 ~30s/trial IO)
# ---------------------------------------------------------------------------

_X_FULL_CACHE: Optional[np.ndarray] = None


def _get_X_subsample(N: int, seed: int) -> np.ndarray:
    """Stratified random sub-sample of full Criteo X feature matrix。

    全量 14M 行 X 在第一次调用时一次性载入内存（约 1.4GB），后续 super-trial
    直接 numpy index，避免重复 read_csv。
    """
    global _X_FULL_CACHE
    if _X_FULL_CACHE is None:
        csv_path = _PROJECT_ROOT / "data" / "criteo-uplift" / "criteo-uplift-v2.1.csv.gz"
        print(f"  [cache] loading full Criteo X (~14M rows) into memory once...")
        t0 = time.time()
        df = pd.read_csv(csv_path, usecols=CRITEO_FEATURE_COLS)
        _X_FULL_CACHE = df.to_numpy(dtype=np.float64)
        print(f"  [cache] loaded {len(_X_FULL_CACHE):,} rows in {time.time()-t0:.1f}s")
    n_total = len(_X_FULL_CACHE)
    if N >= n_total:
        # 不足 N 时直接返回全量（实际不会触发，N 上限 1M < 14M）
        return _X_FULL_CACHE.copy()
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n_total, size=N, replace=False))
    return _X_FULL_CACHE[idx].copy()


# ---------------------------------------------------------------------------
# Single super-trial: build bundle, train funnel net, time 4 components
# ---------------------------------------------------------------------------


def _build_bundle(N: int, seed: int) -> dict:
    cfg = GenConfig(n_samples=N, seed=seed)
    for k, v in PRESETS["realistic"].items():
        setattr(cfg, k, v)
    cfg.conflict_anticorr = 0.6
    X = _get_X_subsample(N, seed)
    return generate(cfg, X)


def _run_super_trial(
    N: int, seed: int, K_list: List[int], max_epochs: int,
) -> List[Dict]:
    """单个 (N, seed) super-trial：跑一次完整 pipeline，测 4 项 component 时间。

    返回 long-form rows（每个 component × K 一行）。
    """
    rows: List[Dict] = []

    # ---- Step 0: bundle 生成 (不计入 component time，仅辅助记录) ----
    t0 = time.perf_counter()
    bundle = _build_bundle(N, seed)
    t_gen = time.perf_counter() - t0
    print(f"  [gen]      {t_gen:>7.1f}s  (data + DGP)")

    X = bundle["X"]
    T = bundle["T"]
    Y_cvr = bundle["Y_cvr"]
    Y_gmv = bundle["Y_gmv"]

    # 60% train / 20% calibration / 20% test split（与 §5.7 一致）
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_te = int(N * 0.2)
    n_cal = int(N * 0.2)
    tr = idx[: N - n_te - n_cal]
    cal = idx[N - n_te - n_cal : N - n_te]
    te = idx[N - n_te :]

    # ---- Step 1: 训练 (Component "train") ----
    arch = FunnelArchConfig(
        d_in=X.shape[1], num_arms=NUM_ARMS,
        rep_hidden=[128, 64], rep_dim=32, head_hidden=[32],
        dropout=0.1, use_anchor=True, learn_log_sigma=True,
        quantile_heads=True, quantile_lo=0.05, quantile_hi=0.95,
        pinball_weight=0.5,
    )
    loss = FunnelLossConfig(
        alpha=0.3, beta=1.0, gamma=0.5, funnel_mode="hard",
        enable_consistency=True, enable_monotonic=True,
    )
    train_cfg = FunnelTrainConfig(
        lr=1e-3, batch_size=512, max_epochs=max_epochs,
        patience=8, seed=seed, verbose=False,
    )

    t0 = time.perf_counter()
    model, info = train_funnel_net(
        X[tr], T[tr], Y_cvr[tr], Y_gmv[tr], arch, loss, train_cfg,
    )
    t_train = time.perf_counter() - t0
    epochs_used = int(info.get("epochs_used", max_epochs))
    print(f"  [train]    {t_train:>7.1f}s  (epochs_used={epochs_used}, n_tr={len(tr):,})")
    rows.append({
        "N": N, "K": NUM_ARMS, "seed": seed,
        "component": "train", "time_s": t_train,
        "extra": f"epochs_used={epochs_used}|n_tr={len(tr)}",
    })

    # ---- Step 2: 推理 (Component "inference") ----
    # warm-up 一次（GPU/MPS lazy init + JIT compile）
    _ = predict_potentials(model, X[te][:128], info, funnel_mode="hard")
    t0 = time.perf_counter()
    pred = predict_potentials(model, X[te], info, funnel_mode="hard")
    t_infer = time.perf_counter() - t0
    print(f"  [infer]    {t_infer:>7.1f}s  (n_te={len(te):,})")
    rows.append({
        "N": N, "K": NUM_ARMS, "seed": seed,
        "component": "inference", "time_s": t_infer,
        "extra": f"n_te={len(te)}",
    })

    # ---- Step 3: Conformal calibration + interval prediction ----
    cp_cfg = JointCPConfig(alpha_c=0.05, alpha_v=0.05)
    t0 = time.perf_counter()
    cal_obj = calibrate_joint_cp(
        model, info, X[cal], T[cal], Y_cvr[cal], Y_gmv[cal], cp_cfg,
    )
    intervals = predict_joint_intervals(model, info, cal_obj, X[te])
    t_cp = time.perf_counter() - t0
    print(f"  [conf_cal] {t_cp:>7.1f}s  (n_cal={len(cal):,})")
    rows.append({
        "N": N, "K": NUM_ARMS, "seed": seed,
        "component": "conformal_cal", "time_s": t_cp,
        "extra": f"n_cal={len(cal)}|alpha=0.05",
    })

    # ---- Step 4: IP solver 三种求解器 × K_list ----
    tau_g_pred = pred["tau_g"]
    tau_c_pred = pred["tau_c"]
    mu_g_full = pred["mu_g_full"]

    for K in K_list:
        if K not in ARM_SUBSETS:
            print(f"  [ip K={K}]  skipped (no arm-subset defined)")
            continue
        arms_subset = ARM_SUBSETS[K]
        # 子集 reward / cost / discount
        tau_g_sub = tau_g_pred[:, arms_subset].copy()
        tau_c_sub = tau_c_pred[:, arms_subset].copy()
        mu_g_sub = mu_g_full[:, arms_subset].copy()
        discount_sub = DISCOUNT_GRID[arms_subset].copy()
        cost_sub = build_cost_matrix(discount_sub, mu_g_sub)
        free_cost_sub = float(cost_sub[:, -1].sum())
        B_sub = float(free_cost_sub * BUDGET_FRAC)

        cfg_ip = IPSolverConfig(
            budget_total=B_sub, reward_mode="point", lambda_cvr=0.0,
        )
        rew = build_reward_matrix(tau_g_sub, tau_c_sub, cfg=cfg_ip)

        # Solver A: TopK baseline
        t0 = time.perf_counter()
        res_topk = solve_topk_baseline(rew, cost_sub, cfg_ip)
        t_topk = time.perf_counter() - t0
        rows.append({
            "N": N, "K": K, "seed": seed,
            "component": "ip_topk", "time_s": t_topk,
            "extra": f"feasible={res_topk.feasible}",
        })

        # Solver B: Lagrange bisect
        t0 = time.perf_counter()
        res_lag = solve_lagrange_bisect(rew, cost_sub, cfg_ip)
        t_lag = time.perf_counter() - t0
        n_iter = int((res_lag.extra or {}).get("n_iter", 0))
        rows.append({
            "N": N, "K": K, "seed": seed,
            "component": "ip_lagrange", "time_s": t_lag,
            "extra": f"n_iter={n_iter}|feasible={res_lag.feasible}",
        })

        # Solver C: LP relaxation + rounding（N<LP_SKIP_N 才跑）
        if N >= LP_SKIP_N:
            rows.append({
                "N": N, "K": K, "seed": seed,
                "component": "ip_lp", "time_s": float("nan"),
                "extra": f"skipped:N>={LP_SKIP_N}_OOM_risk",
            })
            print(f"  [ip K={K}]  topk={t_topk:.3f}s  lag={t_lag:.3f}s  lp=skip(OOM)")
        else:
            try:
                t0 = time.perf_counter()
                res_lp = solve_lp_relax_round(rew, cost_sub, cfg_ip)
                t_lp = time.perf_counter() - t0
                rows.append({
                    "N": N, "K": K, "seed": seed,
                    "component": "ip_lp", "time_s": t_lp,
                    "extra": f"feasible={res_lp.feasible}",
                })
                print(f"  [ip K={K}]  topk={t_topk:.3f}s  lag={t_lag:.3f}s  lp={t_lp:.3f}s")
            except (MemoryError, ValueError) as e:
                rows.append({
                    "N": N, "K": K, "seed": seed,
                    "component": "ip_lp", "time_s": float("nan"),
                    "extra": f"failed:{type(e).__name__}",
                })
                print(f"  [ip K={K}]  topk={t_topk:.3f}s  lag={t_lag:.3f}s  lp=fail({type(e).__name__})")

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="N=10K × K=2 × 1 seed (1 super-trial, sanity check)")
    ap.add_argument("--paper", action="store_true",
                    help="5 N × 3 K × 3 seeds = 45 trial-rows (paper grid)")
    ap.add_argument("--max_epochs", type=int, default=20,
                    help="FunnelCausalNet max_epochs (with early stopping)")
    args = ap.parse_args()

    grid_name = "quick" if args.quick else ("paper" if args.paper else "default")
    grid = SCALE_GRIDS[grid_name]

    # Force unbuffered stdout: 即使被 pipe (e.g. tee) 也保证 line-flush
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

    # 提前创建 long-form CSV 用于 incremental save，避免后台跑被中断时全部丢失
    out_dir = _PROJECT_ROOT / "results" / "e6"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    long_csv = out_dir / f"e6_complexity_{ts}.csv"
    summary_csv = out_dir / f"e6_complexity_{ts}_summary.csv"
    json_path = out_dir / f"e6_complexity_{ts}.json"
    csv_columns = ["N", "K", "seed", "component", "time_s", "extra"]
    pd.DataFrame(columns=csv_columns).to_csv(long_csv, index=False)

    print(f"== E6 complexity runner ({grid_name} mode) ==", flush=True)
    print(f"  sample_sizes : {grid['sample_sizes']}", flush=True)
    print(f"  K_list       : {grid['K_list']}", flush=True)
    print(f"  seeds        : {grid['seeds']}", flush=True)
    print(f"  max_epochs   : {args.max_epochs}", flush=True)
    print(f"  LP skip N    : {LP_SKIP_N}", flush=True)
    print(f"  budget_frac  : {BUDGET_FRAC}", flush=True)
    print(f"  long_csv     : {long_csv} (incremental save enabled)\n", flush=True)

    rows: List[Dict] = []
    n_super = len(grid["sample_sizes"]) * len(grid["seeds"])
    super_idx = 0
    t_start = time.time()

    for N in grid["sample_sizes"]:
        for seed in grid["seeds"]:
            super_idx += 1
            print(f"[{super_idx}/{n_super}] N={N:>7d} seed={seed} "
                  f"K_list={grid['K_list']}", flush=True)
            t0 = time.time()
            super_rows = _run_super_trial(N, seed, grid["K_list"], args.max_epochs)
            rows.extend(super_rows)
            # ----- Incremental append: 立即把本 super-trial 的 rows 落盘 -----
            pd.DataFrame(super_rows, columns=csv_columns).to_csv(
                long_csv, mode="a", header=False, index=False,
            )
            elapsed = time.time() - t0
            eta_min = (time.time() - t_start) / super_idx * (n_super - super_idx) / 60.0
            print(f"  super-trial wall-clock: {elapsed:.1f}s  "
                  f"ETA: {eta_min:.1f}min  (saved to {long_csv.name})\n",
                  flush=True)

    df = pd.DataFrame(rows)

    summary = (
        df.groupby(["N", "K", "component"], as_index=False)
          .agg(
              time_s_mean=("time_s", "mean"),
              time_s_std=("time_s", "std"),
              n_seeds=("seed", "count"),
          )
    )

    summary.to_csv(summary_csv, index=False)
    json_path.write_text(json.dumps({
        "grid": grid_name,
        "max_epochs": args.max_epochs,
        "LP_skip_N": LP_SKIP_N,
        "budget_frac": BUDGET_FRAC,
        "sample_sizes": grid["sample_sizes"],
        "K_list": grid["K_list"],
        "seeds": grid["seeds"],
        "rows": rows,
        "summary": summary.to_dict(orient="records"),
        "wall_clock_total_min": (time.time() - t_start) / 60.0,
    }, indent=2, ensure_ascii=False))

    # 分两段打印 pivot：与 K 无关的训练/推理/标定 vs 依赖 K 的 IP solver
    K_INDEPENDENT = {"train", "inference", "conformal_cal"}
    mask_no_K = summary["component"].isin(K_INDEPENDENT)

    print(f"\n== Summary 1: training / inference / conformal calibration (K=8 fixed) ==")
    if mask_no_K.any():
        pivot_no_K = summary[mask_no_K].pivot_table(
            index="N", columns="component",
            values="time_s_mean", aggfunc="first",
        ).round(3)
        # 列序与论文表 7 一致
        col_order = [c for c in ["train", "inference", "conformal_cal"]
                     if c in pivot_no_K.columns]
        pivot_no_K = pivot_no_K[col_order]
        print(pivot_no_K)

    print(f"\n== Summary 2: IP solver wall-clock (s) by (N, K) ==")
    mask_ip = summary["component"].str.startswith("ip_")
    if mask_ip.any():
        pivot_ip = summary[mask_ip].pivot_table(
            index=["N", "K"], columns="component",
            values="time_s_mean", aggfunc="first",
        ).round(3)
        col_order = [c for c in ["ip_topk", "ip_lagrange", "ip_lp"]
                     if c in pivot_ip.columns]
        pivot_ip = pivot_ip[col_order]
        print(pivot_ip)

    print(f"\n  long-form CSV : {long_csv}")
    print(f"  summary CSV   : {summary_csv}")
    print(f"  JSON dump     : {json_path}")
    print(f"  total wall-clock : {(time.time() - t_start)/60.0:.2f} min")


if __name__ == "__main__":
    main()
