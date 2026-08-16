"""E2 Funnel 约束 ablation runner (Step 5 — pipeline 验证版).

按 §5.4 设计三组对照（D 组 Focal-ZILN 留作完整实验时再加）：
- A: Direct Regression (funnel_mode='direct')      — 仅 anchor 学 log1p(y_g)
- B: Soft Funnel       (funnel_mode='soft')        — anchor 学 y_g + (anchor - decomp)^2 软正则
- C: Hard Funnel       (funnel_mode='hard')        — mu_g = mu_c × mu_v 严格分解

输出：
    results/e2/e2_ablation_<timestamp>.csv  长表 (mode, N, seed, metric, value)
    results/e2/e2_ablation_<timestamp>.json 结构化 dict
    results/e2/e2_ablation_<timestamp>_summary.csv  按 (mode, N) groupby 的 mean ± std

样本规模 / seed 数量为 pipeline 验证版（缩减），论文版需扩大到 §5.4 (b) 表中的规模。

用法：
    python3 code/experiments/run_e2_ablation.py
    python3 code/experiments/run_e2_ablation.py --quick     # 仅 N=2K 快速通跑
    python3 code/experiments/run_e2_ablation.py --full      # 完整论文规模 (慢)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
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
from semisynth.criteo_mt7_generator import (  # noqa: E402
    GenConfig,
    NUM_ARMS,
    PRESETS,
    _load_criteo_X,
    generate,
)


# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------


SCALE_GRIDS = {
    "quick": {"sample_sizes": [2_000], "seeds": [0]},
    "default": {"sample_sizes": [2_000, 10_000, 30_000], "seeds": [0, 1, 2]},
    "paper":  {"sample_sizes": [2_000, 5_000, 10_000, 20_000, 50_000, 100_000],
               "seeds": [0, 1, 2, 3, 4]},
    "full":  {"sample_sizes": [2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 500_000],
              "seeds": [0, 1, 2, 3, 4]},
}


MODES = [
    ("A_direct", "direct"),
    ("B_soft",   "soft"),
    ("C_hard",   "hard"),
    ("D_ziln",   "ziln"),     # VALOR-style focal-BCE + LogNormal NLL
]


# ---------------------------------------------------------------------------
# Single training trial
# ---------------------------------------------------------------------------


def _build_bundle(N: int, seed: int) -> dict:
    cfg = GenConfig(n_samples=N, seed=seed)
    for k, v in PRESETS["realistic"].items():
        setattr(cfg, k, v)
    cfg.conflict_anticorr = 0.6
    csv_path = _PROJECT_ROOT / "data" / "criteo-uplift" / "criteo-uplift-v2.1.csv.gz"
    X = _load_criteo_X(csv_path, N, seed)
    return generate(cfg, X)


def _train_eval_once(N: int, seed: int, mode_name: str, funnel_mode: str,
                     max_epochs: int) -> Dict[str, float]:
    bundle = _build_bundle(N, seed)
    X = bundle["X"]; T = bundle["T"]
    Y_cvr = bundle["Y_cvr"]; Y_gmv = bundle["Y_gmv"]
    tau_g_gt = bundle["tau_gmv"]                 # (N, K+1)
    tau_c_gt = bundle["tau_cvr"]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_te = max(int(N * 0.2), 200)
    tr, te = idx[:N - n_te], idx[N - n_te:]

    arch_cfg = FunnelArchConfig(
        d_in=X.shape[1], num_arms=NUM_ARMS,
        rep_hidden=[128, 64], rep_dim=32, head_hidden=[32],
        dropout=0.1, use_anchor=True, learn_log_sigma=True,
        quantile_heads=False,
    )
    if funnel_mode == "hard":
        alpha_v = 0.3; gamma_m = 0.5; enable_consistency = True; enable_mono = True
    elif funnel_mode == "ziln":
        # NLL 内已含 1/(2σ_norm²) ≈ 5x MSE 系数 (σ_raw=0.8, log_g_std≈2.5)；
        # 对应原 hard mode 的 effective L_v 权重 ≈ 0.3 × 1.0 = 0.3，
        # 这里取 alpha_v=0.06 让 ziln 总权重落在 0.06 × 5 ≈ 0.3 附近以保持公平。
        alpha_v = 0.06; gamma_m = 0.5; enable_consistency = False; enable_mono = True
    else:                         # 'direct' / 'soft'
        alpha_v = 0.05; gamma_m = 0.0; enable_consistency = True; enable_mono = False
    loss_cfg = FunnelLossConfig(
        alpha=alpha_v, beta=1.0, gamma=gamma_m,
        enable_consistency=enable_consistency, enable_monotonic=enable_mono,
        funnel_mode=funnel_mode,
    )
    train_cfg = FunnelTrainConfig(
        lr=1e-3, batch_size=512, max_epochs=max_epochs,
        patience=8, seed=seed, verbose=False,
    )

    t0 = time.time()
    model, info = train_funnel_net(
        X[tr], T[tr], Y_cvr[tr], Y_gmv[tr],
        arch_cfg, loss_cfg, train_cfg,
    )
    train_time = time.time() - t0

    # ziln mode 走与 hard 相同的预测路径（mu_g = mu_c × mu_v）
    pred_mode = "hard" if funnel_mode == "ziln" else funnel_mode
    pred = predict_potentials(model, X[te], info, funnel_mode=pred_mode)
    tau_g_hat = pred["tau_g"]; tau_c_hat = pred["tau_c"]

    tau_g_gt_te = tau_g_gt[te]
    tau_c_gt_te = tau_c_gt[te]

    pehe_g = float(np.sqrt(np.mean((tau_g_hat - tau_g_gt_te) ** 2)))
    pehe_c = float(np.sqrt(np.mean((tau_c_hat - tau_c_gt_te) ** 2)))

    ate_g_hat = tau_g_hat.mean(axis=0); ate_g_gt = tau_g_gt_te.mean(axis=0)
    ate_c_hat = tau_c_hat.mean(axis=0); ate_c_gt = tau_c_gt_te.mean(axis=0)
    ate_g_err_max = float(np.max(np.abs(ate_g_hat - ate_g_gt)))
    ate_c_err_max = float(np.max(np.abs(ate_c_hat - ate_c_gt)))

    funnel_violation = float(np.mean(np.abs(
        pred["mu_g_full"] - pred["mu_c_full"] * pred["mu_v_full"]
    )))

    last_arm = NUM_ARMS - 1
    auuc_g = _auuc_at(tau_g_hat[:, last_arm], tau_g_gt_te[:, last_arm])

    return {
        "mode": mode_name,
        "funnel_mode": funnel_mode,
        "N": N,
        "seed": seed,
        "epochs_used": int(info["epochs_used"]),
        "train_time": train_time,
        "PEHE_GMV": pehe_g,
        "PEHE_CVR": pehe_c,
        "ATE_GMV_err_max": ate_g_err_max,
        "ATE_CVR_err_max": ate_c_err_max,
        "AUUC_GMV_top_arm": auuc_g,
        "funnel_violation_mean": funnel_violation,
    }


def _auuc_at(tau_hat: np.ndarray, tau_gt: np.ndarray) -> float:
    """简化 AUUC：按预测排序后累计真值，归一化为 [0, 1]。
    退化到二档 case 的 standard AUUC，对多档此处仅看「max-arm 上的排序质量」。"""
    order = np.argsort(-tau_hat)
    cum_gt = np.cumsum(tau_gt[order])
    if cum_gt[-1] <= 1e-9:
        return 0.0
    auuc = np.trapezoid(cum_gt) / (len(tau_hat) * cum_gt[-1])
    return float(auuc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--max_epochs", type=int, default=20)
    args = ap.parse_args()

    grid_name = ("quick" if args.quick else "paper" if args.paper
                 else "full" if args.full else "default")
    grid = SCALE_GRIDS[grid_name]
    print(f"== E2 ablation runner ({grid_name} mode) ==")
    print(f"  sample_sizes: {grid['sample_sizes']}")
    print(f"  seeds: {grid['seeds']}")
    print(f"  modes: {[m[0] for m in MODES]}")
    print(f"  max_epochs: {args.max_epochs}\n")

    rows: List[Dict] = []
    n_trials = len(grid["sample_sizes"]) * len(grid["seeds"]) * len(MODES)
    trial_idx = 0
    t_start = time.time()
    for N in grid["sample_sizes"]:
        for seed in grid["seeds"]:
            for mode_name, funnel_mode in MODES:
                trial_idx += 1
                t0 = time.time()
                row = _train_eval_once(N, seed, mode_name, funnel_mode, args.max_epochs)
                dt = time.time() - t0
                eta = (time.time() - t_start) / trial_idx * (n_trials - trial_idx)
                print(f"  [{trial_idx}/{n_trials}] N={N:>6d} seed={seed} mode={mode_name:9s}  "
                      f"PEHE_G={row['PEHE_GMV']:7.2f}  PEHE_C={row['PEHE_CVR']:.4f}  "
                      f"ATE_G_err={row['ATE_GMV_err_max']:6.2f}  AUUC={row['AUUC_GMV_top_arm']:.3f}  "
                      f"violate={row['funnel_violation_mean']:.3g}  "
                      f"time={dt:.1f}s  ETA={eta/60:.1f}min")
                rows.append(row)

    df = pd.DataFrame(rows)

    summary = df.groupby(["mode", "N"], as_index=False).agg(
        PEHE_GMV_mean=("PEHE_GMV", "mean"), PEHE_GMV_std=("PEHE_GMV", "std"),
        PEHE_CVR_mean=("PEHE_CVR", "mean"), PEHE_CVR_std=("PEHE_CVR", "std"),
        ATE_GMV_err_max_mean=("ATE_GMV_err_max", "mean"),
        AUUC_mean=("AUUC_GMV_top_arm", "mean"),
        funnel_viol_mean=("funnel_violation_mean", "mean"),
        train_time_mean=("train_time", "mean"),
    )

    out_dir = _PROJECT_ROOT / "results" / "e2"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    long_csv = out_dir / f"e2_ablation_{ts}.csv"
    summary_csv = out_dir / f"e2_ablation_{ts}_summary.csv"
    json_path = out_dir / f"e2_ablation_{ts}.json"

    df.to_csv(long_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    json_path.write_text(json.dumps({
        "grid": grid_name,
        "max_epochs": args.max_epochs,
        "modes": [m[0] for m in MODES],
        "rows": rows,
        "summary": summary.to_dict(orient="records"),
    }, indent=2, ensure_ascii=False))

    print(f"\n== Summary ==\n{summary.to_string(index=False)}")

    print("\n[verify §I.5 Theorem 1 trend: PEHE_GMV(C_hard) < PEHE_GMV(A_direct)]")
    for N in grid["sample_sizes"]:
        sub = summary[summary["N"] == N].set_index("mode")
        if "C_hard" in sub.index and "A_direct" in sub.index:
            pehe_c = sub.loc["C_hard", "PEHE_GMV_mean"]
            pehe_a = sub.loc["A_direct", "PEHE_GMV_mean"]
            ratio = pehe_c / pehe_a if pehe_a > 0 else float("nan")
            verdict = "✓" if ratio < 1.0 else "✗"
            print(f"  N={N:>6d}: PEHE_GMV C_hard / A_direct = {ratio:.3f}  {verdict}  "
                  f"(C={pehe_c:.2f}, A={pehe_a:.2f})")

    print("\n[verify §I.5 Theorem 2 likelihood equivalence: |D_ziln - C_hard| / C_hard < 0.05]")
    for N in grid["sample_sizes"]:
        sub = summary[summary["N"] == N].set_index("mode")
        if "D_ziln" in sub.index and "C_hard" in sub.index:
            pehe_d = sub.loc["D_ziln", "PEHE_GMV_mean"]
            pehe_c = sub.loc["C_hard", "PEHE_GMV_mean"]
            rel_gap = abs(pehe_d - pehe_c) / max(pehe_c, 1e-6)
            verdict = "✓" if rel_gap < 0.05 else "✗" if rel_gap > 0.20 else "≈"
            print(f"  N={N:>6d}: |D_ziln - C_hard| / C_hard = {rel_gap*100:5.2f}%  {verdict}  "
                  f"(D={pehe_d:.2f}, C={pehe_c:.2f})")

    print(f"\n  long-form CSV    : {long_csv}")
    print(f"  summary CSV      : {summary_csv}")
    print(f"  JSON dump        : {json_path}")
    print(f"\n  total wall-clock : {(time.time() - t_start)/60:.2f} min")


if __name__ == "__main__":
    main()
