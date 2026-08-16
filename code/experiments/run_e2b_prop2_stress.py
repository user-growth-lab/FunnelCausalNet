"""E2.5 — Prop 2 zero-inflation stress test.

Sweeps the zero-inflation parameter (cvr_baseline_logit_mean) of the semi-
synthetic Criteo-MT7 generator and contrasts hard funnel composition (C_hard)
against direct GMV regression (A_direct). Goal: empirically validate Prop 2
prediction that the funnel composite advantage grows with zero mass (1-p)·μ_v²/σ_v².

Outputs:
    results/e2b_prop2_stress/<timestamp>_long.csv  long-form per-seed
    results/e2b_prop2_stress/<timestamp>_summary.csv  groupby (level, mode) means/stds

Usage:
    python3 code/experiments/run_e2b_prop2_stress.py --quick   # smoke test
    python3 code/experiments/run_e2b_prop2_stress.py           # full

Requires the public Criteo feature file at the relative path documented in
``data/README.md``; outcomes and treatments are generated semi-synthetically.
"""
from __future__ import annotations

import argparse
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
from semisynth.criteo_mt7_generator import (  # noqa: E402
    GenConfig,
    NUM_ARMS,
    PRESETS,
    _load_criteo_X,
    generate,
)


ZERO_INFLATION_GRID = [
    ("Z_high",   -3.5),
    ("Z_mid_hi", -2.4),
    ("Z_mid_lo", -1.5),
    ("Z_low",    -0.5),
]

MODES = [
    ("A_direct", "direct"),
    ("C_hard",   "hard"),
]


def _build_bundle(N: int, seed: int, baseline_logit: float) -> dict:
    cfg = GenConfig(n_samples=N, seed=seed)
    for k, v in PRESETS["realistic"].items():
        setattr(cfg, k, v)
    cfg.cvr_baseline_logit_mean = baseline_logit
    cfg.conflict_anticorr = 0.6
    csv_path = _PROJECT_ROOT / "data" / "criteo-uplift" / "criteo-uplift-v2.1.csv.gz"
    X = _load_criteo_X(csv_path, N, seed)
    return generate(cfg, X)


def _auuc_at(tau_hat: np.ndarray, tau_gt: np.ndarray) -> float:
    order = np.argsort(-tau_hat)
    cum = np.cumsum(tau_gt[order])
    avg = float(tau_gt.mean())
    if abs(avg) < 1e-9:
        return float("nan")
    n = len(tau_gt)
    return float(cum.mean() / (n * avg))


def _train_eval_once(N: int, seed: int, baseline_logit: float, mode_name: str,
                     funnel_mode: str, max_epochs: int) -> Dict[str, float]:
    bundle = _build_bundle(N, seed, baseline_logit)
    X = bundle["X"]; T = bundle["T"]
    Y_cvr = bundle["Y_cvr"]; Y_gmv = bundle["Y_gmv"]
    tau_g_gt = bundle["tau_gmv"]
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
    else:
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

    model, info = train_funnel_net(
        X[tr], T[tr], Y_cvr[tr], Y_gmv[tr],
        arch_cfg, loss_cfg, train_cfg,
    )

    pred_mode = funnel_mode
    pred = predict_potentials(model, X[te], info, funnel_mode=pred_mode)
    tau_g_hat = pred["tau_g"]; tau_c_hat = pred["tau_c"]
    tau_g_gt_te = tau_g_gt[te]
    tau_c_gt_te = tau_c_gt[te]

    pehe_g = float(np.sqrt(np.mean((tau_g_hat - tau_g_gt_te) ** 2)))
    pehe_c = float(np.sqrt(np.mean((tau_c_hat - tau_c_gt_te) ** 2)))
    ate_g_err_max = float(np.max(np.abs(tau_g_hat.mean(axis=0) - tau_g_gt_te.mean(axis=0))))

    last_arm = NUM_ARMS - 1
    auuc_g = _auuc_at(tau_g_hat[:, last_arm], tau_g_gt_te[:, last_arm])

    p_obs = float(Y_cvr[te].mean())
    sigma_v_eff = float(np.std(np.log1p(Y_gmv[te][Y_cvr[te] > 0])) if (Y_cvr[te] > 0).sum() > 10 else float("nan"))

    return {
        "p_observed": p_obs,
        "sigma_log1p_v": sigma_v_eff,
        "PEHE_GMV": pehe_g,
        "PEHE_CVR": pehe_c,
        "ATE_GMV_err_max": ate_g_err_max,
        "AUUC_GMV_top_arm": auuc_g,
        "epochs_used": int(info["epochs_used"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=20_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--max_epochs", type=int, default=20)
    ap.add_argument("--quick", action="store_true",
                    help="quick: N=5K, seeds=[0,1], 8 epochs, 2 levels only")
    args = ap.parse_args()

    grid = list(ZERO_INFLATION_GRID)
    if args.quick:
        args.N = 5_000
        args.seeds = [0, 1]
        args.max_epochs = 8
        grid = [grid[0], grid[3]]

    out_dir = _PROJECT_ROOT / "results" / "e2b_prop2_stress"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_long = out_dir / f"{ts}_long.csv"
    out_summary = out_dir / f"{ts}_summary.csv"

    rows: List[Dict] = []
    total = len(grid) * len(MODES) * len(args.seeds)
    done = 0
    t0 = time.time()
    print(f"[E2.5 Prop 2 stress test] N={args.N} seeds={args.seeds} epochs={args.max_epochs} total={total}")
    print(f"  ZERO_INFLATION_GRID: {grid}")
    print(f"  MODES: {MODES}\n")

    for level_name, baseline_logit in grid:
        for mode_name, funnel_mode in MODES:
            for seed in args.seeds:
                done += 1
                run_t0 = time.time()
                try:
                    metrics = _train_eval_once(args.N, seed, baseline_logit,
                                                mode_name, funnel_mode, args.max_epochs)
                except Exception as exc:
                    print(f"  [{done}/{total}] {level_name}/{mode_name}/seed{seed} FAILED: {exc}")
                    import traceback; traceback.print_exc()
                    continue
                wall = time.time() - run_t0
                row = {
                    "level": level_name,
                    "baseline_logit": baseline_logit,
                    "mode": mode_name,
                    "funnel_mode": funnel_mode,
                    "seed": seed,
                    "N": args.N,
                    **metrics,
                    "wall_s": round(wall, 1),
                }
                rows.append(row)
                cum = time.time() - t0
                eta = cum / done * (total - done)
                print(f"  [{done}/{total}] {level_name}({baseline_logit:+.1f})/{mode_name}/seed{seed}: "
                      f"p={metrics['p_observed']:.3f} σ_v={metrics['sigma_log1p_v']:.3f} "
                      f"PEHE_g={metrics['PEHE_GMV']:.2f} AUUC_g={metrics['AUUC_GMV_top_arm']:.3f} "
                      f"ATE_err={metrics['ATE_GMV_err_max']:.2f} ({wall:.0f}s; ETA {eta/60:.1f}min)")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("no rows produced; check the Criteo input path and preceding errors")
    df.to_csv(out_long, index=False)

    summ = df.groupby(["level", "baseline_logit", "mode"]).agg(
        p_obs=("p_observed", "mean"),
        sigma_log1p_v=("sigma_log1p_v", "mean"),
        PEHE_GMV_mean=("PEHE_GMV", "mean"),
        PEHE_GMV_std=("PEHE_GMV", "std"),
        AUUC_GMV_mean=("AUUC_GMV_top_arm", "mean"),
        AUUC_GMV_std=("AUUC_GMV_top_arm", "std"),
        ATE_err_mean=("ATE_GMV_err_max", "mean"),
        ATE_err_std=("ATE_GMV_err_max", "std"),
        n_seeds=("seed", "count"),
    ).reset_index()
    summ.to_csv(out_summary, index=False)

    print(f"\n=== SUMMARY ({len(df)} runs) ===")
    print(summ.to_string(index=False))

    print(f"\n=== Funnel advantage: PEHE(C_hard) / PEHE(A_direct) by zero-inflation level ===")
    print(f"  Prop 2 prediction: ratio decreases as 1-p increases")
    for lvl, baseline_logit in grid:
        try:
            a = summ[(summ.level == lvl) & (summ["mode"] == "A_direct")]["PEHE_GMV_mean"].iloc[0]
            c = summ[(summ.level == lvl) & (summ["mode"] == "C_hard")]["PEHE_GMV_mean"].iloc[0]
            p = summ[(summ.level == lvl) & (summ["mode"] == "C_hard")]["p_obs"].iloc[0]
            sv = summ[(summ.level == lvl) & (summ["mode"] == "C_hard")]["sigma_log1p_v"].iloc[0]
            print(f"  {lvl:10s} (logit={baseline_logit:+.1f}, p_obs={p:.3f}, σ_v={sv:.3f}): "
                  f"A={a:.2f} → C={c:.2f}  ratio={c/a:.3f}  benefit={(1-c/a)*100:+.1f}%")
        except Exception as exc:
            print(f"  {lvl}: incomplete data ({exc})")

    print(f"\n[OK] long: {out_long}")
    print(f"[OK] summary: {out_summary}")


if __name__ == "__main__":
    main()
