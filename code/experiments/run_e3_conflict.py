"""E3 冲突检测 + 冲突强度扫描 pipeline (创新点 II 应用层).

按 §5.1 / §5.4 设计：
- (i) 在 Criteo-MT7 anticorr ∈ {0.0, 0.3, 0.6, 0.9} 四个配置上跑 FunnelCausalNet
      + Joint Conformal CATE，得到 (tau_c, tau_g, width_c, width_g)。
- (ii) 用 conflict_detector.detect_conflict_users 输出 C(K) 集合。
- (iii) 与 ground-truth 冲突用户集合（用 tau_cvr_gt vs tau_gmv_gt 排名分歧 + TopK
      边界两个判据计算）对比，报 Precision / Recall / F1。

输出：
    results/e3/e3_conflict_<timestamp>.csv  (rho_conf, seed, P, R, F1, n_detected, ...)
    results/e3/e3_conflict_<timestamp>_summary.csv
    results/e3/e3_conflict_<timestamp>.json

用法：
    python3 code/experiments/run_e3_conflict.py
    python3 code/experiments/run_e3_conflict.py --quick    # anticorr=0.6 only, 1 seed
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
    train_funnel_net,
)
from methods.joint_conformal_cate import (  # noqa: E402
    JointCPConfig,
    calibrate_joint_cp,
    predict_joint_intervals,
)
from methods.conflict_detector import (  # noqa: E402
    ConflictDetectorConfig,
    detect_conflict_users,
    evaluate_conflict_recall,
)
from semisynth.criteo_mt7_generator import (  # noqa: E402
    GenConfig,
    NUM_ARMS,
    PRESETS,
    _load_criteo_X,
    generate,
)


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------


def _build_bundle(N: int, seed: int, anticorr: float, preset: str = "realistic") -> dict:
    cfg = GenConfig(n_samples=N, seed=seed)
    for k, v in PRESETS[preset].items():
        setattr(cfg, k, v)
    cfg.conflict_anticorr = anticorr
    csv_path = _PROJECT_ROOT / "data" / "criteo-uplift" / "criteo-uplift-v2.1.csv.gz"
    X = _load_criteo_X(csv_path, N, seed)
    return generate(cfg, X)


def _run_trial(N: int, seed: int, anticorr: float, max_epochs: int,
               cd_cfg: ConflictDetectorConfig, preset: str = "realistic") -> Dict:
    bundle = _build_bundle(N, seed, anticorr, preset=preset)
    X = bundle["X"]; T = bundle["T"]
    Y_cvr = bundle["Y_cvr"]; Y_gmv = bundle["Y_gmv"]
    tau_c_gt = bundle["tau_cvr"]; tau_g_gt = bundle["tau_gmv"]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_te = int(N * 0.2); n_cal = int(N * 0.2)
    tr = idx[:N - n_te - n_cal]
    cal = idx[N - n_te - n_cal:N - n_te]
    te = idx[N - n_te:]

    arch_cfg = FunnelArchConfig(
        d_in=X.shape[1], num_arms=NUM_ARMS,
        rep_hidden=[128, 64], rep_dim=32, head_hidden=[32],
        dropout=0.1, use_anchor=True, learn_log_sigma=True,
        quantile_heads=True, quantile_lo=0.05, quantile_hi=0.95,
        pinball_weight=0.5,
    )
    loss_cfg = FunnelLossConfig(
        alpha=0.3, beta=1.0, gamma=0.5,
        enable_consistency=True, enable_monotonic=True,
        funnel_mode="hard",
    )
    train_cfg = FunnelTrainConfig(
        lr=1e-3, batch_size=512, max_epochs=max_epochs,
        patience=8, seed=seed, verbose=False,
    )

    t0 = time.time()
    model, info = train_funnel_net(X[tr], T[tr], Y_cvr[tr], Y_gmv[tr],
                                   arch_cfg, loss_cfg, train_cfg)
    train_time = time.time() - t0

    cp_cfg = JointCPConfig(alpha_c=0.05, alpha_v=0.05)
    cal_obj = calibrate_joint_cp(model, info,
                                 X[cal], T[cal], Y_cvr[cal], Y_gmv[cal], cp_cfg)
    intervals = predict_joint_intervals(model, info, cal_obj, X[te])

    tau_c_pred = (intervals["tau_c_lo"] + intervals["tau_c_hi"]) / 2.0
    tau_g_pred = (intervals["tau_g_lo"] + intervals["tau_g_hi"]) / 2.0

    detection = detect_conflict_users(
        tau_c_pred, tau_g_pred,
        intervals["width_c"], intervals["width_g"], cd_cfg,
    )

    metrics = evaluate_conflict_recall(
        detection["mask"], tau_c_gt[te], tau_g_gt[te], cd_cfg,
    )

    rho = float(np.corrcoef(tau_c_gt[te, NUM_ARMS - 1],
                            tau_g_gt[te, NUM_ARMS - 1])[0, 1])

    return {
        "rho_conf": anticorr,
        "seed": seed,
        "N": N,
        "preset": preset,
        "rho_pearson_max_arm": rho,
        "n_detected": detection["n_conflict"],
        "frac_detected": detection["frac_conflict"],
        "n_gt_conflict": metrics["n_gt_conflict"],
        "tp": metrics["tp"], "fp": metrics["fp"], "fn": metrics["fn"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "train_time": train_time,
        "epochs_used": int(info["epochs_used"]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="anticorr=0.9 only, 1 seed, N=current default")
    ap.add_argument("--N", type=int, default=20_000)
    ap.add_argument("--max_epochs", type=int, default=30)
    ap.add_argument("--preset", choices=list(PRESETS.keys()), default="realistic",
                    help="DGP preset: realistic / amplified / conflict (E3 推荐 conflict)")
    ap.add_argument("--seeds", type=str, default=None,
                    help="Comma-sep seeds, e.g. '0,1,2'. Overrides default.")
    ap.add_argument("--anticorr-grid", type=str, default=None,
                    help="Comma-sep anticorr values, e.g. '0.0,0.3,0.6,0.9'.")
    ap.add_argument("--delta-rank", type=float, default=0.20,
                    help="Conflict detector rank-divergence threshold")
    ap.add_argument("--delta-width-q", type=float, default=0.50,
                    help="Conflict detector width quantile threshold")
    ap.add_argument("--top-k-pct", type=float, default=10.0,
                    help="TopK band size in percent (default 10%)")
    args = ap.parse_args()

    if args.quick:
        anticorr_grid = [0.9]
        seeds = [0]
    else:
        anticorr_grid = [0.0, 0.3, 0.6, 0.9]
        seeds = [0, 1, 2]
    if args.anticorr_grid is not None:
        anticorr_grid = [float(x) for x in args.anticorr_grid.split(",") if x.strip()]
    if args.seeds is not None:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    cd_cfg = ConflictDetectorConfig(
        arm_for_decision=NUM_ARMS - 1, top_k_pct=args.top_k_pct,
        delta_rank=args.delta_rank, delta_width_quantile=args.delta_width_q,
    )

    print(f"== E3 conflict detection runner ({'quick' if args.quick else 'default'}) ==")
    print(f"  preset: {args.preset}")
    print(f"  anticorr_grid: {anticorr_grid}")
    print(f"  seeds: {seeds},  N: {args.N}")
    print(f"  conflict_detector: arm={cd_cfg.arm_for_decision}, "
          f"topK={cd_cfg.top_k_pct}%, δ_rank={cd_cfg.delta_rank}, "
          f"δ_width_q={cd_cfg.delta_width_quantile}")
    print(f"  max_epochs: {args.max_epochs}\n")

    rows: List[Dict] = []
    n_total = len(anticorr_grid) * len(seeds)
    trial_idx = 0
    t_start = time.time()

    for rho_conf in anticorr_grid:
        for seed in seeds:
            trial_idx += 1
            t0 = time.time()
            row = _run_trial(args.N, seed, rho_conf, args.max_epochs, cd_cfg,
                             preset=args.preset)
            dt = time.time() - t0
            eta = (time.time() - t_start) / trial_idx * (n_total - trial_idx)
            print(f"  [{trial_idx}/{n_total}] ρ_conf={rho_conf:.1f}  seed={seed}  "
                  f"ρ_pearson={row['rho_pearson_max_arm']:+.3f}  "
                  f"n_gt={row['n_gt_conflict']:>4d}  n_det={row['n_detected']:>4d}  "
                  f"P={row['precision']:.3f}  R={row['recall']:.3f}  F1={row['f1']:.3f}  "
                  f"time={dt:.1f}s  ETA={eta/60:.1f}min")
            rows.append(row)

    df = pd.DataFrame(rows)

    summary = df.groupby("rho_conf", as_index=False).agg(
        rho_pearson_mean=("rho_pearson_max_arm", "mean"),
        n_gt_conflict_mean=("n_gt_conflict", "mean"),
        n_detected_mean=("n_detected", "mean"),
        precision_mean=("precision", "mean"), precision_std=("precision", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        train_time_mean=("train_time", "mean"),
    )

    out_dir = _PROJECT_ROOT / "results" / "e3"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    long_csv = out_dir / f"e3_conflict_{ts}.csv"
    summary_csv = out_dir / f"e3_conflict_{ts}_summary.csv"
    json_path = out_dir / f"e3_conflict_{ts}.json"

    df.to_csv(long_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    json_path.write_text(json.dumps({
        "preset": args.preset,
        "anticorr_grid": anticorr_grid,
        "seeds": seeds,
        "N": args.N,
        "max_epochs": args.max_epochs,
        "conflict_detector_cfg": cd_cfg.__dict__,
        "rows": rows,
        "summary": summary.to_dict(orient="records"),
    }, indent=2, ensure_ascii=False))

    print(f"\n== Summary ==\n{summary.to_string(index=False)}")

    print("\n[verify §II 第 4 项 C(K) 集合检测能在高 ρ_conf 下找到非空冲突 + F1 > 0]")
    for rho_conf in anticorr_grid:
        sub = summary[summary["rho_conf"] == rho_conf].iloc[0]
        verdict = "✓" if sub["f1_mean"] > 0.05 else "≈" if sub["f1_mean"] > 0 else "✗"
        print(f"  ρ_conf={rho_conf:.1f}: F1={sub['f1_mean']:.3f} ± {sub['f1_std']:.3f}  "
              f"(P={sub['precision_mean']:.3f}, R={sub['recall_mean']:.3f}, "
              f"n_det={sub['n_detected_mean']:.0f})  {verdict}")

    print(f"\n  long-form CSV    : {long_csv}")
    print(f"  summary CSV      : {summary_csv}")
    print(f"  JSON dump        : {json_path}")
    print(f"\n  total wall-clock : {(time.time() - t_start)/60:.2f} min")


if __name__ == "__main__":
    main()
