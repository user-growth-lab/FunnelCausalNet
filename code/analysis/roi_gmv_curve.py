"""ΔROI vs ΔGMV 用户级累积曲线分析（业务决策视角）。

业务问题：「当 GMV 增加 10%, 50%, 100%, 200% 时，ΔROI 各是多少？」
答：把每个 (user, arm) 决策按 ranking score 降序加入，累积 (ΔGMV, cost)，
每加一个就是曲线上一个点，得到从最严格到最宽松预算的连续曲线。

横轴：ΔGMV / GMV_control_baseline × 100%
       （baseline = 全员不发券的 ground-truth GMV 总和）
纵轴：ΔROI = ΔGMV / cost
       （cost = DISCOUNT_GRID[k] × Y_gmv[i,k]，即只有产生 GMV 才消耗预算）

策略：
- random：随机给每个用户一个非 control 档位 + 随机加入顺序
- topk：按 τ̂_g[:, max-arm] 降序，只发 max-arm
- lp：每个用户选 best ROI arm（argmax τ̂_g/cost），然后按 best ROI 排序加入
- anchored：同 lp 但用 RCT-anchored shift 后的 τ̂_g_anc

输出：
- results/e4/roi_gmv_curve_mt7_<seed>.png / .pdf
- results/e4/roi_gmv_curve_mt7_targets.csv  关键 GMV 档位的 ΔROI 表

用法：
    python3 code/analysis/roi_gmv_curve.py --N 20000 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


DISCOUNT_GRID = np.linspace(0.0, 0.14, NUM_ARMS).astype(np.float64)


def _build_bundle(N: int, seed: int, anticorr: float = 0.6) -> dict:
    cfg = GenConfig(n_samples=N, seed=seed)
    for k, v in PRESETS["realistic"].items():
        setattr(cfg, k, v)
    cfg.conflict_anticorr = anticorr
    csv_path = _PROJECT_ROOT / "data" / "criteo-uplift" / "criteo-uplift-v2.1.csv.gz"
    X = _load_criteo_X(csv_path, N, seed)
    return generate(cfg, X)


def _train_pipeline(bundle: dict, seed: int, max_epochs: int = 30) -> dict:
    """训 funnel net (hard) + 准备 train/test 索引 + anchored shift。"""
    X = bundle["X"]; T = bundle["T"]
    Y_cvr = bundle["Y_cvr"]; Y_gmv = bundle["Y_gmv"]
    N = X.shape[0]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_te = int(N * 0.3)
    tr = idx[:N - n_te]
    te = idx[N - n_te:]

    arch = FunnelArchConfig(
        d_in=X.shape[1], num_arms=NUM_ARMS,
        rep_hidden=[128, 64], rep_dim=32, head_hidden=[32],
        dropout=0.1, use_anchor=True, learn_log_sigma=True,
        quantile_heads=False,
    )
    loss = FunnelLossConfig(alpha=0.3, beta=1.0, gamma=0.5, funnel_mode="hard")
    train_cfg = FunnelTrainConfig(
        lr=1e-3, batch_size=512, max_epochs=max_epochs,
        patience=8, seed=seed, verbose=False,
    )

    model, info = train_funnel_net(X[tr], T[tr], Y_cvr[tr], Y_gmv[tr],
                                   arch, loss, train_cfg)

    pred_te = predict_potentials(model, X[te], info, funnel_mode="hard")
    tau_g_pred = pred_te["tau_g"]            # (n_te, NUM_ARMS)
    mu_g_pred = pred_te["mu_g_full"]         # (n_te, NUM_ARMS)

    # RCT per-arm 经验组均值（train data 上）
    a_k = np.array([
        float(Y_gmv[tr][T[tr] == k].mean()) if (T[tr] == k).any() else 0.0
        for k in range(NUM_ARMS)
    ])
    bar_mu = mu_g_pred.mean(axis=0)
    delta_k = a_k - bar_mu
    mu_g_anc = mu_g_pred + delta_k[None, :]
    tau_g_anc = mu_g_anc - mu_g_anc[:, 0:1]

    return {
        "te": te, "tau_g_pred": tau_g_pred, "mu_g_pred": mu_g_pred,
        "tau_g_anc": tau_g_anc, "delta_k": delta_k, "a_k": a_k,
    }


def _marginal_greedy_curve(
    score_pred: np.ndarray,
    tau_g_gt_te: np.ndarray,
    cost_gt_te: np.ndarray,
    GMV_control_baseline: float,
    subsample_pts: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """边际升档贪心曲线（fractional knapsack 风格）。

    把每个 (user i, arm k) 当作独立 item，按 score_pred[i,k] 降序遍历。
    每用户从 control(arm=0) 开始，遇到比当前 arm 更高的档就「升档」：
        ΔGMV += tau_g_gt[i, k] - tau_g_gt[i, current_arm[i]]
        Δcost += cost_gt[i, k] - cost_gt[i, current_arm[i]]
    （遇到比当前 arm 更低或相等的就跳过）

    这样能从「全员 control」一路扫到「按 predicted score 给每用户配 best arm」
    （理想模型下 = 全员 max-arm），覆盖 ΔGMV ∈ [0, 全员 max-arm 增量] 整段。

    subsample_pts: 等间隔子采样，避免曲线点数 > 100K 拖慢绘图。
    """
    n_te, K = score_pred.shape
    sp = score_pred.copy()
    sp[:, 0] = -np.inf  # control 不进 ranking

    flat_score = sp.flatten()
    flat_user = np.repeat(np.arange(n_te), K)
    flat_arm = np.tile(np.arange(K), n_te)

    order = np.argsort(-flat_score, kind="stable")

    arm_cur = np.zeros(n_te, dtype=np.int64)
    cum_gmv = 0.0
    cum_cost = 0.0
    xs: List[float] = []
    ys: List[float] = []
    for idx in order:
        i = flat_user[idx]
        k = flat_arm[idx]
        if k <= arm_cur[i] or k == 0:
            continue
        if not np.isfinite(flat_score[idx]):
            break
        d_gmv = float(tau_g_gt_te[i, k] - tau_g_gt_te[i, arm_cur[i]])
        d_cost = float(cost_gt_te[i, k] - cost_gt_te[i, arm_cur[i]])
        cum_gmv += d_gmv
        cum_cost += d_cost
        arm_cur[i] = k
        if cum_cost > 1e-9 and cum_gmv > 1e-9:
            xs.append(cum_gmv / GMV_control_baseline * 100.0)
            ys.append(cum_gmv / cum_cost)

    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    if len(xs_arr) > subsample_pts:
        step = max(1, len(xs_arr) // subsample_pts)
        xs_arr = np.concatenate([xs_arr[::step], xs_arr[-1:]])
        ys_arr = np.concatenate([ys_arr[::step], ys_arr[-1:]])
    return xs_arr, ys_arr


def _topk_marginal_curve(
    tau_g_pred_te: np.ndarray,
    cost_pred_te: np.ndarray,
    tau_g_gt_te: np.ndarray,
    cost_gt_te: np.ndarray,
    GMV_control_baseline: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """topk 策略：只发 max-arm，按 tau_g_pred[i, max_arm]/cost_pred[i, max_arm] 降序。"""
    K = tau_g_pred_te.shape[1]
    rew = tau_g_pred_te[:, K - 1]
    cst = np.clip(cost_pred_te[:, K - 1], 1e-9, None)
    score = rew / cst
    order = np.argsort(-score)
    gmv_seq = tau_g_gt_te[order, K - 1]
    cost_seq = cost_gt_te[order, K - 1]
    cum_gmv = np.cumsum(gmv_seq)
    cum_cost = np.clip(np.cumsum(cost_seq), 1e-9, None)
    return cum_gmv / GMV_control_baseline * 100.0, cum_gmv / cum_cost


def _build_random_curve(
    tau_g_gt_te: np.ndarray, cost_gt_te: np.ndarray,
    GMV_control_baseline: float, seed: int, n_repeat: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """random 策略：把 (i,k) 全部随机打分，再走同样的 marginal greedy。"""
    rng = np.random.RandomState(seed)
    n_te, K = tau_g_gt_te.shape
    all_x: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    target_len = None
    for r in range(n_repeat):
        rand_score = rng.rand(n_te, K)
        x, y = _marginal_greedy_curve(rand_score, tau_g_gt_te, cost_gt_te,
                                       GMV_control_baseline,
                                       subsample_pts=2000)
        all_x.append(x); all_y.append(y)
        if target_len is None:
            target_len = len(x)

    min_len = min(len(x) for x in all_x)
    all_x = np.stack([x[:min_len] for x in all_x])
    all_y = np.stack([y[:min_len] for y in all_y])
    return all_x.mean(axis=0), all_y.mean(axis=0)


def _interp_at_targets(x: np.ndarray, y: np.ndarray, targets: List[float]) -> Dict[float, float]:
    """在 x_pct 单调上升假设下，对每个 target 用线性插值出 y_ROI。"""
    out: Dict[float, float] = {}
    for tgt in targets:
        if tgt < x.min() or tgt > x.max():
            out[tgt] = float("nan")
            continue
        idx = np.searchsorted(x, tgt)
        if idx == 0:
            out[tgt] = float(y[0])
        elif idx >= len(x):
            out[tgt] = float(y[-1])
        else:
            x0, x1 = x[idx - 1], x[idx]
            y0, y1 = y[idx - 1], y[idx]
            if x1 - x0 < 1e-9:
                out[tgt] = float(y0)
            else:
                w = (tgt - x0) / (x1 - x0)
                out[tgt] = float(y0 * (1 - w) + y1 * w)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--out_dir", type=str, default="results/e4")
    args = p.parse_args()

    out_dir = _PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ROI-GMV curve] mt7  N={args.N}  seed={args.seed}")
    bundle = _build_bundle(args.N, args.seed)
    pipe = _train_pipeline(bundle, args.seed, args.epochs)
    te = pipe["te"]
    tau_g_pred = pipe["tau_g_pred"]; mu_g_pred = pipe["mu_g_pred"]
    tau_g_anc = pipe["tau_g_anc"]
    print(f"  delta_k (RCT shift): {np.array2string(pipe['delta_k'], precision=2)}")

    tau_g_gt_te = bundle["tau_gmv"][te]
    mu_g_gt_te = bundle["mu_gmv_full"][te]
    cost_gt_te = DISCOUNT_GRID[None, :] * mu_g_gt_te
    cost_pred_te = DISCOUNT_GRID[None, :] * np.clip(mu_g_pred, 1e-9, None)
    GMV_control_baseline = float(mu_g_gt_te[:, 0].sum())
    print(f"  GMV_control_baseline (sum mu_g[i,0]): {GMV_control_baseline:.1f}")

    n_te = tau_g_gt_te.shape[0]
    arange = np.arange(n_te)

    curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    curves["topk"] = _topk_marginal_curve(
        tau_g_pred, cost_pred_te, tau_g_gt_te, cost_gt_te, GMV_control_baseline,
    )

    score_lp = tau_g_pred / np.clip(cost_pred_te, 1e-9, None)
    score_lp = np.where(np.isfinite(score_lp), score_lp, -np.inf)
    curves["lp"] = _marginal_greedy_curve(
        score_lp, tau_g_gt_te, cost_gt_te, GMV_control_baseline,
    )

    score_anc = tau_g_anc / np.clip(cost_pred_te, 1e-9, None)
    score_anc = np.where(np.isfinite(score_anc), score_anc, -np.inf)
    curves["anchored"] = _marginal_greedy_curve(
        score_anc, tau_g_gt_te, cost_gt_te, GMV_control_baseline,
    )

    curves["random"] = _build_random_curve(
        tau_g_gt_te, cost_gt_te, GMV_control_baseline,
        seed=args.seed, n_repeat=5,
    )

    targets = [1.0, 5.0, 10.0, 20.0, 30.0, 50.0]
    rows = []
    for name, (xc, yc) in curves.items():
        roi_at = _interp_at_targets(xc, yc, targets)
        for tgt, roi in roi_at.items():
            rows.append({"strategy": name, "target_dGMV_pct": tgt, "delta_ROI": roi,
                         "max_dGMV_pct_reached": float(xc.max())})
    df_targets = pd.DataFrame(rows)
    print("\n== ΔROI at target ΔGMV% (interpolated) ==")
    print(df_targets.pivot(index="target_dGMV_pct", columns="strategy",
                           values="delta_ROI").round(3))
    df_targets.to_csv(out_dir / f"roi_gmv_curve_mt7_targets_seed{args.seed}.csv",
                      index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"topk": "tab:orange", "lp": "tab:green",
              "anchored": "tab:red", "random": "tab:gray"}
    styles = {"topk": "-", "lp": "-", "anchored": "-", "random": "--"}
    for name, (xc, yc) in curves.items():
        mask = (xc > 0.5) & (xc < 600)
        ax.plot(xc[mask], yc[mask], color=colors[name], linestyle=styles[name],
                label=name, linewidth=2, alpha=0.85)

    ax.axhline(1.0, color="red", linestyle=":", alpha=0.6, linewidth=1.0)
    ax.axhline(5.0, color="green", linestyle=":", alpha=0.6, linewidth=1.0)

    for tgt in [10, 50, 100, 200]:
        ax.axvline(tgt, color="lightgray", linestyle=":", alpha=0.5, linewidth=0.8)

    ax.set_xlabel(r"$\Delta$GMV / GMV$_{\rm control\_baseline}$  $\times$ 100%")
    ax.set_ylabel(r"$\Delta$ROI = $\Delta$GMV / Cost")
    ax.set_title(f"$\\Delta$ROI vs $\\Delta$GMV cumulative curve  (Criteo-MT7, N={args.N}, seed={args.seed})")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, max(c[0].max() for c in curves.values()) * 1.05)
    ax.set_ylim(0, max(c[1].max() for c in curves.values() if c[1].max() < 50) * 1.1)

    plt.tight_layout()
    png_path = out_dir / f"roi_gmv_curve_mt7_seed{args.seed}.png"
    pdf_path = out_dir / f"roi_gmv_curve_mt7_seed{args.seed}.pdf"
    plt.savefig(png_path, dpi=150)
    plt.savefig(pdf_path)
    plt.close()
    print(f"\n  curve saved: {png_path}")
    print(f"  targets csv: {out_dir / f'roi_gmv_curve_mt7_targets_seed{args.seed}.csv'}")


if __name__ == "__main__":
    main()
