"""E2 paper grid 汇总脚本：将 results/e2/*.csv 转成论文 §5.4 表 + 定理 1/2 验证统计.

输入：results/e2/e2_ablation_paper_<timestamp>.csv  (4 modes × 6 sizes × 5 seeds = 120 rows)
输出：
    results/e2/e2_summary_<timestamp>_table.csv      论文表（含 ± std）
    results/e2/e2_summary_<timestamp>_thm1.csv       定理 1 验证：PEHE_GMV(C_hard) / PEHE_GMV(A_direct) 应随 N 趋于 0
    results/e2/e2_summary_<timestamp>_thm2.csv       定理 2 验证：|D_ziln - C_hard| / C_hard 应随 N → 0
    results/e2/e2_summary_<timestamp>.md             markdown 表（直接贴入 §5.4 (h)）

用法：
    python3 code/experiments/e2_summarize.py                  # 自动找最新 paper csv
    python3 code/experiments/e2_summarize.py --csv <path>     # 指定 csv 路径
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_DIR = _PROJECT_ROOT / "results" / "e2"


def _find_latest_paper_csv() -> Path:
    """Find the largest (paper-grid sized) e2_ablation csv. Excludes summary files."""
    candidates = list(_RESULTS_DIR.glob("e2_ablation_*.csv"))
    candidates = [p for p in candidates
                  if "summary" not in p.stem and "thm" not in p.stem]
    if not candidates:
        raise FileNotFoundError(f"未在 {_RESULTS_DIR} 下找到 e2_ablation_*.csv")
    candidates.sort(key=lambda p: (p.stat().st_size, p.name))
    return candidates[-1]


def _format_mean_std(mean: float, std: float, fmt: str = "{:.2f}") -> str:
    if pd.isna(mean):
        return "—"
    if pd.isna(std):
        return fmt.format(mean)
    return f"{fmt.format(mean)}±{fmt.format(std)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None,
                    help="指定 e2_ablation_paper csv，未指定则取最新")
    args = ap.parse_args()

    csv_path = Path(args.csv) if args.csv else _find_latest_paper_csv()
    print(f"读取: {csv_path}")
    df = pd.read_csv(csv_path)

    if "violation_mean" not in df.columns and "funnel_violation_mean" in df.columns:
        df = df.rename(columns={"funnel_violation_mean": "violation_mean"})

    metrics = ["PEHE_GMV", "PEHE_CVR", "ATE_GMV_err_max", "AUUC_GMV_top_arm",
               "violation_mean"]
    available_metrics = [m for m in metrics if m in df.columns]
    print(f"  可用指标: {available_metrics}")

    # ---------- 表 1: 按 (N, mode) 聚合 mean ± std ----------
    table = df.groupby(["N", "mode"], as_index=False)[available_metrics].agg(["mean", "std"])
    table.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in table.columns]

    rows = []
    for N in sorted(df["N"].unique()):
        for mode in ["A_direct", "B_soft", "C_hard", "D_ziln"]:
            sub = df[(df["N"] == N) & (df["mode"] == mode)]
            if len(sub) == 0:
                continue
            r = {"N": N, "mode": mode, "n_seeds": len(sub)}
            for m in available_metrics:
                r[f"{m}_mean"] = float(sub[m].mean())
                r[f"{m}_std"] = float(sub[m].std()) if len(sub) > 1 else float("nan")
            rows.append(r)
    table_df = pd.DataFrame(rows)

    # ---------- 表 2: 定理 1 验证 - PEHE_GMV(C_hard) / PEHE_GMV(A_direct) ----------
    thm1_rows = []
    for N in sorted(df["N"].unique()):
        a = df[(df["N"] == N) & (df["mode"] == "A_direct")]["PEHE_GMV"]
        c = df[(df["N"] == N) & (df["mode"] == "C_hard")]["PEHE_GMV"]
        if len(a) == 0 or len(c) == 0:
            continue
        ratio_mean = float(c.mean() / a.mean()) if a.mean() > 0 else float("nan")
        # 配对比较：每 seed 上 C/A
        seeds = sorted(set(df[df["N"] == N]["seed"].unique()))
        paired = []
        for s in seeds:
            a_s = df[(df["N"] == N) & (df["mode"] == "A_direct") & (df["seed"] == s)]["PEHE_GMV"]
            c_s = df[(df["N"] == N) & (df["mode"] == "C_hard") & (df["seed"] == s)]["PEHE_GMV"]
            if len(a_s) == 1 and len(c_s) == 1 and float(a_s.iloc[0]) > 0:
                paired.append(float(c_s.iloc[0]) / float(a_s.iloc[0]))
        thm1_rows.append({
            "N": N,
            "n_seeds": len(seeds),
            "PEHE_GMV_A_mean": float(a.mean()),
            "PEHE_GMV_C_mean": float(c.mean()),
            "ratio_mean_of_means": ratio_mean,
            "ratio_paired_mean": float(np.mean(paired)) if paired else float("nan"),
            "ratio_paired_std": float(np.std(paired)) if len(paired) > 1 else float("nan"),
            "verdict_thm1": "✓ ratio<1" if ratio_mean < 1.0 else "✗",
        })
    thm1_df = pd.DataFrame(thm1_rows)

    # ---------- 表 3: 定理 2 验证 - |D_ziln - C_hard| / C_hard ----------
    thm2_rows = []
    for N in sorted(df["N"].unique()):
        c = df[(df["N"] == N) & (df["mode"] == "C_hard")]["PEHE_GMV"]
        d = df[(df["N"] == N) & (df["mode"] == "D_ziln")]["PEHE_GMV"]
        if len(c) == 0 or len(d) == 0:
            continue
        gap_mean = float(abs(d.mean() - c.mean()) / c.mean()) if c.mean() > 0 else float("nan")
        seeds = sorted(set(df[df["N"] == N]["seed"].unique()))
        paired = []
        for s in seeds:
            c_s = df[(df["N"] == N) & (df["mode"] == "C_hard") & (df["seed"] == s)]["PEHE_GMV"]
            d_s = df[(df["N"] == N) & (df["mode"] == "D_ziln") & (df["seed"] == s)]["PEHE_GMV"]
            if len(c_s) == 1 and len(d_s) == 1 and float(c_s.iloc[0]) > 0:
                paired.append(abs(float(d_s.iloc[0]) - float(c_s.iloc[0])) / float(c_s.iloc[0]))
        thm2_rows.append({
            "N": N,
            "n_seeds": len(seeds),
            "PEHE_GMV_C_mean": float(c.mean()),
            "PEHE_GMV_D_mean": float(d.mean()),
            "gap_mean_of_means": gap_mean,
            "gap_paired_mean": float(np.mean(paired)) if paired else float("nan"),
            "gap_paired_std": float(np.std(paired)) if len(paired) > 1 else float("nan"),
            "verdict_thm2": "✓ <0.20" if gap_mean < 0.20 else
                            "≈ <0.50" if gap_mean < 0.50 else "✗",
        })
    thm2_df = pd.DataFrame(thm2_rows)

    # ---------- markdown 论文表（按 §5.4 (h) 风格）----------
    md_lines = ["### §5.4 (h) E2 paper grid 完整结果（{}-seed × 6-N × 4-modes）"
                .format(int(df["seed"].nunique())),
                "",
                "| N | Mode | PEHE_GMV | PEHE_CVR | ATE_GMV_err | AUUC_GMV | violation% |",
                "|---|------|---------:|---------:|------------:|---------:|-----------:|"]
    for _, r in table_df.iterrows():
        N = int(r["N"])
        mode = r["mode"]
        cells = [f"{N:>6,d}", mode]
        for m, fmt in [("PEHE_GMV", "{:.2f}"), ("PEHE_CVR", "{:.4f}"),
                        ("ATE_GMV_err_max", "{:.2f}"),
                        ("AUUC_GMV_top_arm", "{:.3f}"),
                        ("violation_mean", "{:.2f}")]:
            if f"{m}_mean" in r:
                cells.append(_format_mean_std(r[f"{m}_mean"], r[f"{m}_std"], fmt))
            else:
                cells.append("—")
        md_lines.append("| " + " | ".join(cells) + " |")

    md_lines.extend([
        "",
        "**定理 1 验证（PEHE_GMV 比值 C_hard / A_direct，应随 N 单调下降趋于 0）：**",
        "",
        "| N | C_hard | A_direct | ratio (paired) | verdict |",
        "|---|-------:|---------:|---------------:|:-------:|",
    ])
    for _, r in thm1_df.iterrows():
        md_lines.append(
            f"| {int(r['N']):>6,d} | {r['PEHE_GMV_C_mean']:.2f} | "
            f"{r['PEHE_GMV_A_mean']:.2f} | "
            f"{r['ratio_paired_mean']:.3f}±{r['ratio_paired_std']:.3f} | "
            f"{r['verdict_thm1']} |"
        )

    md_lines.extend([
        "",
        "**定理 2 验证（D_ziln 与 C_hard 的相对差距，应随 N → ∞ 收敛到 0）：**",
        "",
        "| N | C_hard | D_ziln | gap=|D−C|/C (paired) | verdict |",
        "|---|-------:|-------:|---------------------:|:-------:|",
    ])
    for _, r in thm2_df.iterrows():
        md_lines.append(
            f"| {int(r['N']):>6,d} | {r['PEHE_GMV_C_mean']:.2f} | "
            f"{r['PEHE_GMV_D_mean']:.2f} | "
            f"{r['gap_paired_mean']:.3f}±{r['gap_paired_std']:.3f} | "
            f"{r['verdict_thm2']} |"
        )

    md = "\n".join(md_lines)

    # ---------- 输出 ----------
    stem = csv_path.stem.replace("e2_ablation_paper_", "e2_summary_")
    if "e2_summary_" not in stem:
        stem = csv_path.stem.replace("e2_ablation_", "e2_summary_")
    table_csv = _RESULTS_DIR / f"{stem}_table.csv"
    thm1_csv = _RESULTS_DIR / f"{stem}_thm1.csv"
    thm2_csv = _RESULTS_DIR / f"{stem}_thm2.csv"
    md_path = _RESULTS_DIR / f"{stem}.md"

    table_df.to_csv(table_csv, index=False)
    thm1_df.to_csv(thm1_csv, index=False)
    thm2_df.to_csv(thm2_csv, index=False)
    md_path.write_text(md, encoding="utf-8")

    print(f"\n  论文表 CSV    : {table_csv}")
    print(f"  定理 1 验证   : {thm1_csv}")
    print(f"  定理 2 验证   : {thm2_csv}")
    print(f"  Markdown 表   : {md_path}\n")
    print("=" * 80)
    print(md)
    print("=" * 80)


if __name__ == "__main__":
    main()
