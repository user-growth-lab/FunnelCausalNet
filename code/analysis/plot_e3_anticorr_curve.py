"""Plot E3 conflict-detection ablation curves (论文 §5.6 图 4).

Inputs (auto-detect latest):
  results/e3/e3_conflict_<ts>.csv          long-form (rho_conf, seed, P/R/F1, ...)
  results/e3/e3_conflict_<ts>_summary.csv  agg per rho_conf

Outputs:
  results/e3/e3_anticorr_curve.png         double-panel figure
  results/e3/e3_anticorr_curve.pdf         vector copy for paper
  results/e3/e3_anticorr_table.csv         clean main table
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
RESULTS_DIR = _PROJECT_ROOT / "results" / "e3"


def _latest_long_csv() -> Path:
    cands = sorted(RESULTS_DIR.glob("e3_conflict_2*.csv"))
    cands = [p for p in cands if not p.stem.endswith("_summary")]
    if not cands:
        raise FileNotFoundError("No e3_conflict CSV found in results/e3/")
    return cands[-1]


def main() -> None:
    long_csv = _latest_long_csv()
    print(f"[plot] reading {long_csv}")
    df = pd.read_csv(long_csv)

    # group statistics
    g = df.groupby("rho_conf")
    summary = pd.DataFrame({
        "rho_conf":          g["rho_conf"].first().values,
        "n_seeds":           g.size().values,
        "n_gt_mean":         g["n_gt_conflict"].mean().values,
        "n_det_mean":        g["n_detected"].mean().values,
        "P_mean":            g["precision"].mean().values,
        "P_std":             g["precision"].std(ddof=1).values,
        "R_mean":            g["recall"].mean().values,
        "R_std":             g["recall"].std(ddof=1).values,
        "F1_mean":           g["f1"].mean().values,
        "F1_std":            g["f1"].std(ddof=1).values,
        "rho_pearson_GT":    g["rho_pearson_max_arm"].mean().values,
    }).sort_values("rho_conf").reset_index(drop=True)

    # seed=0 single-seed curve
    s0 = df[df["seed"] == 0].sort_values("rho_conf").reset_index(drop=True)

    # save clean table
    out_csv = RESULTS_DIR / "e3_anticorr_table.csv"
    summary.to_csv(out_csv, index=False)

    # ----------- plotting -----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)

    rho_x = summary["rho_conf"].values

    # Left: 5-seed mean ± std envelope for F1, P, R
    ax = axes[0]
    for col_mean, col_std, label, color in [
        ("F1_mean", "F1_std", "F1",        "#d62728"),
        ("P_mean",  "P_std",  "Precision", "#1f77b4"),
        ("R_mean",  "R_std",  "Recall",    "#2ca02c"),
    ]:
        m = summary[col_mean].values
        s = summary[col_std].fillna(0).values
        ax.plot(rho_x, m, "o-", label=label, color=color, lw=2.0, ms=6)
        ax.fill_between(rho_x, np.maximum(m - s, 0.0), np.minimum(m + s, 1.0),
                        alpha=0.18, color=color)
    ax.set_title("(a) 5-seed mean ± std (mt7 conflict preset, $\\gamma$=0.15)")
    ax.set_xlabel("$\\rho_{\\mathrm{conf}}$ (DGP-injected anti-correlation)")
    ax.set_ylabel("Detection metric")
    ax.set_ylim(-0.02, 1.0)
    ax.set_xticks([0.0, 0.3, 0.6, 0.9])
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)

    # Right: signal-rich seed=0 single curve + 5-seed envelope
    ax = axes[1]
    for col, label, color in [
        ("f1",        "F1",        "#d62728"),
        ("precision", "Precision", "#1f77b4"),
        ("recall",    "Recall",    "#2ca02c"),
    ]:
        ax.plot(s0["rho_conf"].values, s0[col].values,
                "o-", label=label, color=color, lw=2.2, ms=7)
        # individual seeds as light points to show spread
        for sd in df["seed"].unique():
            if sd == 0:
                continue
            ds = df[df["seed"] == sd].sort_values("rho_conf")
            ax.plot(ds["rho_conf"].values, ds[col].values,
                    "o", color=color, alpha=0.18, ms=5)
    ax.set_title("(b) Signal-rich seed=0 (line) + other seeds (dots)")
    ax.set_xlabel("$\\rho_{\\mathrm{conf}}$")
    ax.set_xticks([0.0, 0.3, 0.6, 0.9])
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)

    plt.suptitle(
        "E3 Conflict Detection: Precision / Recall / F1 vs DGP anti-correlation",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()

    out_png = RESULTS_DIR / "e3_anticorr_curve.png"
    out_pdf = RESULTS_DIR / "e3_anticorr_curve.pdf"
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"[plot] wrote {out_png}")
    print(f"[plot] wrote {out_pdf}")
    print(f"[plot] wrote {out_csv}")
    print("\n=== Main table (5-seed mean ± std) ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
