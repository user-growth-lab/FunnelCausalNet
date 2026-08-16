"""Plot E6 complexity scaling curves (论文 §5.8 图 5).

Inputs (auto-detect latest):
  results/e6/e6_complexity_<ts>.csv          long-form (N, K, seed, component, time_s, extra)
  results/e6/e6_complexity_<ts>_summary.csv  agg per (N, K, component)

Outputs:
  results/e6/e6_scaling_curve.{png,pdf}      4-panel figure (log-log)
  results/e6/e6_scaling_table.csv            clean main table for §5.8

子图布局 (2 × 2)：
  (a) 训练时间 vs N（K=8 fixed）—— 验证近线性 scale
  (b) IP solver wall-clock vs N（K=8 fixed）—— 三种 solver 对比，
      验证 LP 急速增长 vs Lagrange / TopK 平稳（对应 §III.5 (e) 定理 3 + (f) 复杂度表）
  (c) IP Lagrange wall-clock vs K（按 N 分层）—— 验证 K 维度的近线性 scale
      （对应 §III.5 (f) 表中 O(N · K · log(1/ε))）
  (d) 推理 + Conformal 标定 vs N —— 表明二者均 < 1s 可忽略
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = _PROJECT_ROOT / "results" / "e6"


def _latest_long_csv() -> Path:
    cands = sorted(RESULTS_DIR.glob("e6_complexity_2*.csv"))
    cands = [p for p in cands if not p.stem.endswith("_summary")]
    if not cands:
        raise FileNotFoundError("No e6_complexity CSV found in results/e6/")
    return cands[-1]


# Component → (display label, color, marker)
COMPONENT_STYLE = {
    "train":         ("Training",                "#d62728", "o"),
    "inference":     ("Inference",               "#1f77b4", "s"),
    "conformal_cal": ("Conformal calibration",   "#2ca02c", "^"),
    "ip_topk":       ("IP TopK baseline",        "#ff7f0e", "D"),
    "ip_lagrange":   ("IP Lagrange (ours)",      "#9467bd", "o"),
    "ip_lp":         ("IP LP relaxation",        "#8c564b", "x"),
}


def _agg(df: pd.DataFrame) -> pd.DataFrame:
    """Group by (N, K, component) → (time_s mean ± std)."""
    return (
        df.dropna(subset=["time_s"])
          .groupby(["N", "K", "component"], as_index=False)
          .agg(time_s_mean=("time_s", "mean"),
               time_s_std=("time_s", "std"),
               n_seeds=("seed", "count"))
    )


def _plot_panel_a(ax: plt.Axes, agg: pd.DataFrame) -> None:
    """Training time vs N (log-log), K=8 fixed."""
    sub = agg[agg["component"] == "train"].sort_values("N")
    if sub.empty:
        ax.text(0.5, 0.5, "(no training data)", ha="center", va="center",
                transform=ax.transAxes)
        return
    label, color, marker = COMPONENT_STYLE["train"]
    N_vals = sub["N"].values
    m = sub["time_s_mean"].values
    s = sub["time_s_std"].fillna(0).values
    ax.errorbar(N_vals, m, yerr=s, fmt=f"-{marker}", color=color, lw=2.0,
                ms=7, capsize=3, label=label)

    # 近线性 reference 虚线 (slope=1 in log-log)
    if len(N_vals) >= 2:
        N_ref = np.array([N_vals[0], N_vals[-1]], dtype=float)
        m_ref = m[0] * (N_ref / N_vals[0])
        ax.plot(N_ref, m_ref, "--", color="gray", alpha=0.7, lw=1.2,
                label="Linear reference (slope=1)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("(a) FunnelCausalNet training time vs $N$ (K=8 fixed)")
    ax.set_xlabel("Sample size $N$")
    ax.set_ylabel("Wall-clock time (s)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3, which="both")


def _plot_panel_b(ax: plt.Axes, agg: pd.DataFrame) -> None:
    """IP solver wall-clock vs N at K=8 (3 solvers compared)."""
    K_full = 8
    for comp in ["ip_topk", "ip_lagrange", "ip_lp"]:
        sub = agg[(agg["component"] == comp) & (agg["K"] == K_full)].sort_values("N")
        if sub.empty:
            continue
        label, color, marker = COMPONENT_STYLE[comp]
        N_vals = sub["N"].values
        m = sub["time_s_mean"].values
        s = sub["time_s_std"].fillna(0).values
        # 用 max(time, 1e-4) 避免 log scale 下 0ms 点丢失
        m_plot = np.maximum(m, 1e-4)
        ax.errorbar(N_vals, m_plot, yerr=s, fmt=f"-{marker}", color=color,
                    lw=2.0, ms=7, capsize=3, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("(b) IP solver wall-clock vs $N$ (K=8 fixed)")
    ax.set_xlabel("Sample size $N$")
    ax.set_ylabel("Wall-clock time (s)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3, which="both")


def _plot_panel_c(ax: plt.Axes, agg: pd.DataFrame) -> None:
    """IP Lagrange wall-clock vs K, stratified by N."""
    sub = agg[agg["component"] == "ip_lagrange"].sort_values(["N", "K"])
    if sub.empty:
        ax.text(0.5, 0.5, "(no ip_lagrange data)", ha="center", va="center",
                transform=ax.transAxes)
        return
    N_vals = sorted(sub["N"].unique())
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(len(N_vals) - 1, 1)) for i in range(len(N_vals))]
    for c, N in zip(colors, N_vals):
        cur = sub[sub["N"] == N]
        K_vals = cur["K"].values
        m = cur["time_s_mean"].values
        s = cur["time_s_std"].fillna(0).values
        m_plot = np.maximum(m, 1e-4)
        N_label = f"N={N//1000}K" if N < 1_000_000 else f"N={N//1_000_000}M"
        ax.errorbar(K_vals, m_plot, yerr=s, fmt="-o", color=c, lw=2.0, ms=6,
                    capsize=3, label=N_label)

    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.set_title("(c) IP Lagrange wall-clock vs $K$ (per-$N$ slice)")
    ax.set_xlabel("Number of treatment arms $K$")
    ax.set_ylabel("Wall-clock time (s)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3, which="both")
    ax.set_xticks(sorted(sub["K"].unique()))


def _plot_panel_d(ax: plt.Axes, agg: pd.DataFrame) -> None:
    """Inference + Conformal calibration vs N (K=8 fixed)."""
    for comp in ["inference", "conformal_cal"]:
        sub = agg[agg["component"] == comp].sort_values("N")
        if sub.empty:
            continue
        label, color, marker = COMPONENT_STYLE[comp]
        N_vals = sub["N"].values
        m = sub["time_s_mean"].values
        s = sub["time_s_std"].fillna(0).values
        m_plot = np.maximum(m, 1e-4)
        ax.errorbar(N_vals, m_plot, yerr=s, fmt=f"-{marker}", color=color,
                    lw=2.0, ms=7, capsize=3, label=label)

    # 1s reference line
    ax.axhline(1.0, ls=":", color="red", alpha=0.6, lw=1.0,
               label="1s reference")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("(d) Inference + Conformal calibration vs $N$")
    ax.set_xlabel("Sample size $N$")
    ax.set_ylabel("Wall-clock time (s)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3, which="both")


def main() -> None:
    long_csv = _latest_long_csv()
    print(f"[plot] reading {long_csv}")
    df = pd.read_csv(long_csv)
    agg = _agg(df)

    # ----------- 4-panel figure -----------
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    _plot_panel_a(axes[0, 0], agg)
    _plot_panel_b(axes[0, 1], agg)
    _plot_panel_c(axes[1, 0], agg)
    _plot_panel_d(axes[1, 1], agg)

    plt.suptitle(
        "E6 Computational Complexity: 4 components × (N, K) scaling",
        fontsize=13, y=1.00,
    )
    plt.tight_layout()

    out_png = RESULTS_DIR / "e6_scaling_curve.png"
    out_pdf = RESULTS_DIR / "e6_scaling_curve.pdf"
    out_csv = RESULTS_DIR / "e6_scaling_table.csv"
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")

    # ----------- clean main table -----------
    table = agg.copy()
    table["time_s_mean"] = table["time_s_mean"].round(4)
    table["time_s_std"] = table["time_s_std"].round(4)
    # 补全 ip_lp 跳过的行（标注 N/A）
    table.to_csv(out_csv, index=False)

    print(f"[plot] wrote {out_png}")
    print(f"[plot] wrote {out_pdf}")
    print(f"[plot] wrote {out_csv}")

    # ----------- text summary for §5.8 -----------
    print("\n=== §5.8 Summary 1: train / inference / conformal_cal (K=8 fixed) ===")
    K_INDEPENDENT = ["train", "inference", "conformal_cal"]
    pivot1 = (
        agg[agg["component"].isin(K_INDEPENDENT)]
        .pivot_table(index="N", columns="component",
                     values="time_s_mean", aggfunc="first")
        .round(3)
    )
    col_order1 = [c for c in K_INDEPENDENT if c in pivot1.columns]
    print(pivot1[col_order1].to_string())

    print("\n=== §5.8 Summary 2: IP solver wall-clock (s) by (N, K) ===")
    pivot2 = (
        agg[agg["component"].str.startswith("ip_")]
        .pivot_table(index=["N", "K"], columns="component",
                     values="time_s_mean", aggfunc="first")
        .round(4)
    )
    col_order2 = [c for c in ["ip_topk", "ip_lagrange", "ip_lp"]
                  if c in pivot2.columns]
    print(pivot2[col_order2].to_string())

    # ----------- key findings text -----------
    train_sub = agg[agg["component"] == "train"].sort_values("N")
    if len(train_sub) >= 2:
        N_min, N_max = train_sub["N"].iloc[0], train_sub["N"].iloc[-1]
        t_min, t_max = train_sub["time_s_mean"].iloc[0], train_sub["time_s_mean"].iloc[-1]
        scale_N = N_max / N_min
        scale_T = t_max / t_min
        slope = np.log(scale_T) / np.log(scale_N)
        print(f"\n[finding 1] Training: N {N_min:,}→{N_max:,} ({scale_N:.0f}×), "
              f"time {t_min:.1f}s→{t_max:.1f}s ({scale_T:.1f}×), log-log slope ≈ {slope:.2f}")

    lag_sub = agg[(agg["component"] == "ip_lagrange") & (agg["K"] == 8)].sort_values("N")
    if len(lag_sub) >= 2:
        t_max = lag_sub["time_s_mean"].iloc[-1] * 1000  # ms
        N_max = lag_sub["N"].iloc[-1]
        print(f"[finding 2] Lagrange IP @ N={N_max:,}, K=8: {t_max:.0f} ms")

    lp_sub = agg[(agg["component"] == "ip_lp") & (agg["K"] == 8)].sort_values("N")
    lag_sub_pair = agg[(agg["component"] == "ip_lagrange") & (agg["K"] == 8)]
    if len(lp_sub) >= 1 and len(lag_sub_pair) >= 1:
        N_lp_max = lp_sub["N"].iloc[-1]
        t_lp = lp_sub["time_s_mean"].iloc[-1]
        lag_at_N = lag_sub_pair[lag_sub_pair["N"] == N_lp_max]["time_s_mean"]
        if len(lag_at_N) > 0:
            t_lag = lag_at_N.iloc[0]
            speedup = t_lp / max(t_lag, 1e-9)
            print(f"[finding 3] LP vs Lagrange @ N={N_lp_max:,}, K=8: "
                  f"{t_lp:.1f}s vs {t_lag*1000:.0f}ms, Lagrange speedup ≈ {speedup:.0f}×")


if __name__ == "__main__":
    main()
