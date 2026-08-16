"""Criteo-MT7: semi-synthetic 8-arm uplift dataset with funnel-aware
zero-inflated GMV, derived from Criteo-Uplift v2.1 features.

This generator targets the FunnelCausalNet + multi-tier coupon allocation
experimental needs in this study.

Key design choices:
- 8 randomly assigned coupon-discount arms with evenly spaced tiers
  {control, 2%, 4%, 6%, 8%, 10%, 12%, 14%}, an operationally common
  multi-tier discount schedule in e-commerce coupon programs.
- Funnel structure:
      P(conv | x, t)               (CVR head, sigmoid)
      log E[OV | x, t, conv=1]     (OV head, LogNormal)
      GMV = conv · OV              (zero-inflated by construction)
- Heterogeneous treatment sensitivities for CVR and OV that are
  partially anti-correlated; ``conflict_anticorr`` (in [0, 1]) controls
  the strength of the CVR/GMV TopK conflict the conflict diagnostic
  is designed to flag.
- Two presets that fix the dose magnitude:
      REALISTIC : baseline CVR ~ 8% and per-tier dose curves chosen
                  within the operationally common e-commerce coupon
                  range (5%-15% baseline CVR, modest per-tier ATE on
                  conversion, mild free-rider effect on conditional
                  spend); used for the main matrix.
      AMPLIFIED : doubles dose magnitude for sanity-check experiments.

DGP (per sample i ∈ [N], arm t ∈ {0,...,7}):
    d(t)           = discount_pct[t] / 14            ∈ [0, 1]
    α(x), γ(x)     = baseline shifts        (orthogonal random directions in R^d)
    β(x), δ(x)     = treatment-sensitivity shifts (β ⟂ α, δ partially -β)
    sens_cvr(x)    = μ_cvr_dose + β(x)
    sens_ov(x)     = μ_ov_dose  + δ(x)

    logit P(conv=1 | x, t)                = (μ_cvr_base + α(x)) + sens_cvr(x) · d(t)
    log E[OV | x, t, conv=1]              = (μ_log_ov  + γ(x)) + sens_ov(x)  · d(t)
    OV ~ LogNormal(log_mean, σ_log²)
    Y_cvr ~ Bernoulli(P(conv | x, T))
    Y_gmv = Y_cvr · OV

Outputs (saved to ``data/semisynth/criteo-mt7/<preset>_anticorr<v>/``):
  data.npz with X, T, Y_cvr, Y_gmv, mu_cvr_full, mu_gmv_full,
                 Y_cvr_full, Y_gmv_full, tau_cvr, tau_gmv, pi_true.
  meta.json   : dataset metadata + GenConfig snapshot.
  summary.txt : per-arm marginal stats + CVR/GMV conflict diagnostic.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CRITEO_FEATURE_COLS = [f"f{i}" for i in range(12)]

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent

DEFAULT_CSV = _PROJECT_ROOT / "data" / "criteo-uplift" / "criteo-uplift-v2.1.csv.gz"
DEFAULT_OUT_ROOT = _PROJECT_ROOT / "data" / "semisynth" / "criteo-mt7"

# Coupon discount tiers (control + seven non-zero tiers up to 14%).
DISCOUNT_PCT = np.array([0, 2, 4, 6, 8, 10, 12, 14], dtype=np.float64)
NUM_ARMS = len(DISCOUNT_PCT)  # 8 (control + 7 tiers)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GenConfig:
    """Generator hyper-parameters.

    Defaults follow the REALISTIC preset, set so that baseline CVR
    (~8%), per-tier dose curves on conversion, and free-rider effects
    on conditional spend fall inside operationally common e-commerce
    coupon ranges (cf. e.g.\\ Criteo Uplift v2 baseline CVR ~5%-12%;
    Hillstrom baseline CVR ~10%). The exact moments are not matched
    to any specific industrial dataset and the controlled stress
    test (E2b) sweeps the zero-inflation regime independently of
    the default operating point.
    """

    n_samples: int = 500_000
    seed: int = 42

    # CVR head
    cvr_baseline_logit_mean: float = -2.4    # σ(-2.4) ≈ 8.3% baseline CVR
    cvr_alpha_strength: float = 0.6           # X -> baseline CVR logit (heterogeneity)
    cvr_beta_strength: float = 0.4            # X -> per-user CVR-dose sensitivity
    cvr_dose_max: float = 0.6                 # ≈ +5pp CVR at 14% top tier

    # OV head (LogNormal)
    ov_log_mean: float = 6.0                  # exp(6.0 + σ²/2) ≈ 550 currency units
    ov_gamma_strength: float = 0.3            # X -> baseline log(OV)
    ov_delta_strength: float = 0.25           # X -> per-user OV-dose sensitivity
    ov_dose_max: float = -0.05                # ≈ -5% OV at 14% (mild free-rider effect)
    ov_log_sigma: float = 0.8                 # within-user log-OV noise (heavy tail)

    # Conflict structure (used by the Top-K conflict diagnostic)
    conflict_anticorr: float = 0.6            # 0..1: how much δ-direction anti-correlates with β-direction
    latent_realign: bool = False              # True: enforce Corr(β_x, δ_x) = -ρ via z-score + Gram-Schmidt
                                              # (used by 'conflict' preset; off elsewhere to preserve
                                              # bit-stable realistic / amplified DGP)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# Three presets:
#   - realistic: operational-scale dose magnitudes (default; used for E1/E2 main matrix)
#   - amplified: dose effect doubled (signal-robustness ablations)
#   - conflict:  E3 conflict-detection only -- ov_delta dominates GMV heterogeneity so
#                conflict_anticorr really propagates to (tau_c, tau_g) anti-correlation.
#                Diagnostic (diagnose_e3_anticorr.py) shows: under the default realistic
#                preset, anticorr=0.9 yields rho_pearson(tau_c, tau_g) ~ +0.01 (diluted by
#                beta); under the conflict preset it reaches ~-0.63 (TopC<->BotG ~ 37%).
PRESETS: Dict[str, Dict[str, Any]] = {
    "realistic": {
        "cvr_dose_max": 0.6,
        "ov_dose_max": -0.05,
    },
    "amplified": {
        "cvr_dose_max": 1.2,
        "ov_dose_max": -0.10,
    },
    "conflict": {
        "cvr_dose_max": 0.6,
        "ov_dose_max": -0.10,
        "cvr_beta_strength": 0.4,
        "ov_delta_strength": 0.5,
        # 关键：缩小 OV baseline 异质性 γ_x（X→log_OV 维度），避免它在 τ_g
        # 量级上盖过 (β_x, δ_x) 的 conflict 信号；诊断显示 γ=0.30 时
        # τ_g 的 Pearson 相关性跨 seed 抖动 ±0.6，γ=0.15 下抖动收敛到 ±0.15。
        "ov_gamma_strength": 0.15,
        "latent_realign": True,
    },
}


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _standardize(X: np.ndarray) -> np.ndarray:
    """Robust z-score; tolerates NaN/Inf and zero-variance columns."""
    X = np.nan_to_num(X.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd > 1e-8, sd, 1.0)
    Xs = np.clip((X - mu) / sd, -8.0, 8.0)
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(Xs).all():
        raise RuntimeError("standardize produced non-finite values; check input X.")
    return Xs


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else v


def _load_criteo_X(csv_path: Path, n_rows: int, seed: int) -> np.ndarray:
    """Load the 12-d Criteo feature matrix and stratify-shuffle to n_rows."""
    df = pd.read_csv(csv_path, usecols=CRITEO_FEATURE_COLS)
    n_total = len(df)
    if n_rows < n_total:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_total, size=n_rows, replace=False)
        df = df.iloc[np.sort(idx)].reset_index(drop=True)
    return df.to_numpy(dtype=np.float64)


# ---------------------------------------------------------------------------
# Core DGP
# ---------------------------------------------------------------------------


def generate(cfg: GenConfig, X: np.ndarray) -> Dict[str, np.ndarray]:
    """Generate the full ground-truth bundle.

    Parameters
    ----------
    cfg : GenConfig
    X   : (N, d) raw Criteo feature matrix (will be z-scored internally).

    Returns
    -------
    bundle dict (see module docstring for schema).
    """
    # macOS Apple Accelerate BLAS occasionally raises spurious IEEE flags
    # inside matmul SIMD kernels even when inputs and outputs are bounded
    # (see numpy/numpy#22813). Silence them; correctness is unaffected.
    with np.errstate(all="ignore"):
        return _generate_impl(cfg, X)


def _generate_impl(cfg: GenConfig, X: np.ndarray) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    Xs = _standardize(X)
    N, d = Xs.shape

    # ---------- Random direction vectors ----------
    # CVR-sensitivity direction (β); OV-sensitivity direction (δ) is
    # partially anti-correlated with β to plant CVR/GMV TopK conflict.
    beta_dir = _unit(rng.normal(size=d))

    rand_dir = rng.normal(size=d)
    rand_dir = _unit(rand_dir - (rand_dir @ beta_dir) * beta_dir)
    rho = float(np.clip(cfg.conflict_anticorr, 0.0, 1.0))
    delta_dir = _unit(-rho * beta_dir + np.sqrt(1.0 - rho ** 2) * rand_dir)

    # Baseline directions (independent of β/δ)
    alpha_dir = _unit(rng.normal(size=d))
    gamma_dir = _unit(rng.normal(size=d))

    # ---------- Per-user latent scalars ----------
    alpha_x = cfg.cvr_alpha_strength * (Xs @ alpha_dir)        # (N,)
    gamma_x = cfg.ov_gamma_strength  * (Xs @ gamma_dir)        # (N,)

    if cfg.latent_realign:
        # ---------- Latent re-alignment to enforce Corr(β_x, δ_x) = -ρ ----------
        # Direction-level orthogonality (-ρ·β_dir + √(1-ρ²)·rand_dir) only ensures
        # the *vector* inner product equals -ρ; under Criteo's non-Gaussian 12-d X,
        # the realized Pearson(X·β_dir, X·δ_dir) drifts seed-by-seed by ±0.6.
        # Standardise both latents and rebuild δ_x as -ρ·β_x + √(1-ρ²)·resid,
        # guaranteeing Corr(β_x, δ_x) = -ρ exactly (modulo finite-sample noise)
        # across seeds.  Used by `conflict` preset; off by default to keep
        # realistic / amplified preset numerics bit-stable across releases.
        def _zscore(v):
            s = float(v.std())
            return (v - v.mean()) / (s if s > 1e-8 else 1.0)

        beta_x_n  = _zscore(Xs @ beta_dir)
        delta_raw = Xs @ delta_dir
        delta_proj = float(np.dot(delta_raw, beta_x_n)) / N
        delta_resid_n = _zscore(delta_raw - delta_proj * beta_x_n)
        delta_x_n  = (-rho) * beta_x_n + np.sqrt(max(0.0, 1.0 - rho ** 2)) * delta_resid_n
        beta_x  = cfg.cvr_beta_strength  * beta_x_n            # (N,) std ≈ strength
        delta_x = cfg.ov_delta_strength  * delta_x_n           # (N,) std ≈ strength
    else:
        beta_x  = cfg.cvr_beta_strength  * (Xs @ beta_dir)     # (N,)
        delta_x = cfg.ov_delta_strength  * (Xs @ delta_dir)    # (N,)

    sens_cvr = cfg.cvr_dose_max + beta_x                       # (N,)
    sens_ov  = cfg.ov_dose_max  + delta_x                      # (N,)

    doses = DISCOUNT_PCT / DISCOUNT_PCT[-1]                    # (8,) ∈ [0, 1]

    # ---------- Counterfactual CVR ----------
    cvr_logit_full = (
        (cfg.cvr_baseline_logit_mean + alpha_x)[:, None]
        + sens_cvr[:, None] * doses[None, :]
    )                                                          # (N, 8)
    mu_cvr_full = _sigmoid(cvr_logit_full)

    # ---------- Counterfactual log(OV) and expected OV ----------
    log_ov_full = (
        (cfg.ov_log_mean + gamma_x)[:, None]
        + sens_ov[:, None] * doses[None, :]
    )                                                          # (N, 8)
    expected_ov_full = np.exp(log_ov_full + 0.5 * cfg.ov_log_sigma ** 2)

    # ---------- Counterfactual expected GMV (funnel identity) ----------
    mu_gmv_full = mu_cvr_full * expected_ov_full               # (N, 8)

    # ---------- Observed treatment + outcomes ----------
    T = rng.integers(low=0, high=NUM_ARMS, size=N)             # uniform RCT
    p_obs_cvr = mu_cvr_full[np.arange(N), T]
    Y_cvr = (rng.random(N) < p_obs_cvr).astype(np.int64)

    obs_log_ov_mean = log_ov_full[np.arange(N), T]
    obs_log_ov = obs_log_ov_mean + rng.normal(0.0, cfg.ov_log_sigma, size=N)
    Y_ov_if_conv = np.exp(obs_log_ov)
    Y_gmv = Y_cvr.astype(np.float64) * Y_ov_if_conv

    # ---------- Potential outcomes (sampled, all 8 arms) ----------
    Y_cvr_full = (rng.random((N, NUM_ARMS)) < mu_cvr_full).astype(np.int64)
    log_ov_noise_full = rng.normal(0.0, cfg.ov_log_sigma, size=(N, NUM_ARMS))
    Y_ov_full = np.exp(log_ov_full + log_ov_noise_full)
    Y_gmv_full = Y_cvr_full.astype(np.float64) * Y_ov_full

    # ---------- Treatment effects (counterfactual ground-truth) ----------
    tau_cvr = mu_cvr_full - mu_cvr_full[:, [0]]                # (N, 8)
    tau_gmv = mu_gmv_full - mu_gmv_full[:, [0]]                # (N, 8)

    pi_true = np.full(NUM_ARMS, 1.0 / NUM_ARMS, dtype=np.float64)

    return {
        "X": Xs.astype(np.float32),
        "T": T,
        "Y_cvr": Y_cvr,
        "Y_gmv": Y_gmv.astype(np.float32),
        "mu_cvr_full": mu_cvr_full.astype(np.float32),
        "mu_gmv_full": mu_gmv_full.astype(np.float32),
        "Y_cvr_full": Y_cvr_full.astype(np.int8),
        "Y_gmv_full": Y_gmv_full.astype(np.float32),
        "tau_cvr": tau_cvr.astype(np.float32),
        "tau_gmv": tau_gmv.astype(np.float32),
        "pi_true": pi_true,
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarize(bundle: Dict[str, np.ndarray], cfg: GenConfig) -> str:
    """Generate a per-arm summary + CVR/GMV TopK conflict diagnostic."""
    T = bundle["T"]
    Y_cvr = bundle["Y_cvr"]
    Y_gmv = bundle["Y_gmv"]
    tau_cvr = bundle["tau_cvr"]
    tau_gmv = bundle["tau_gmv"]

    lines: list[str] = []
    lines.append(
        f"N = {T.shape[0]:,}, d = {bundle['X'].shape[1]}, "
        f"K = {NUM_ARMS} arms, conflict_anticorr = {cfg.conflict_anticorr:.2f}"
    )
    lines.append(
        "Treatment marginal: "
        + ", ".join(f"{(T == t).mean():.4f}" for t in range(NUM_ARMS))
    )
    lines.append("")
    lines.append("Per-arm observed marginals + ground-truth ATE:")
    lines.append(
        f"{'arm':>4s} {'pct':>5s} {'n':>10s} {'CVR_obs':>9s} "
        f"{'GMV_obs':>10s} {'τCVR_gt':>9s} {'τGMV_gt':>10s}"
    )
    for t in range(NUM_ARMS):
        mask = T == t
        n = int(mask.sum())
        cvr_obs = float(Y_cvr[mask].mean()) if n > 0 else 0.0
        gmv_obs = float(Y_gmv[mask].mean()) if n > 0 else 0.0
        tau_c_gt = float(tau_cvr[:, t].mean())
        tau_g_gt = float(tau_gmv[:, t].mean())
        lines.append(
            f"{t:>4d} {DISCOUNT_PCT[t]:>4.0f}% {n:>10,} "
            f"{cvr_obs:>9.4f} {gmv_obs:>10.2f} "
            f"{tau_c_gt:>+9.4f} {tau_g_gt:>+10.3f}"
        )

    # CVR/GMV TopK conflict at the maximum dose (t=7, 14%)
    tau_c_at_max = tau_cvr[:, -1]
    tau_g_at_max = tau_gmv[:, -1]
    rho_pearson = float(np.corrcoef(tau_c_at_max, tau_g_at_max)[0, 1])

    # Spearman via ranks (avoid scipy dependency)
    rank_c = np.argsort(np.argsort(tau_c_at_max))
    rank_g = np.argsort(np.argsort(tau_g_at_max))
    rho_spearman = float(np.corrcoef(rank_c, rank_g)[0, 1])

    # TopK overlap (Innovation II's "conflict user" diagnostic)
    N = tau_c_at_max.shape[0]
    k = max(1, N // 10)
    top_c = np.argsort(-tau_c_at_max)[:k]
    top_g = np.argsort(-tau_g_at_max)[:k]
    jaccard = len(set(top_c.tolist()) & set(top_g.tolist())) / max(
        1, len(set(top_c.tolist()) | set(top_g.tolist()))
    )

    lines.append("")
    lines.append("Conflict diagnostic (at max dose, t=7 / 14%):")
    lines.append(f"  Pearson(τ_CVR,  τ_GMV) = {rho_pearson:+.4f}")
    lines.append(f"  Spearman(τ_CVR, τ_GMV) = {rho_spearman:+.4f}")
    lines.append(f"  TopK Jaccard (K=N/10)  = {jaccard:.4f}")
    lines.append(
        "  (lower correlation / Jaccard = stronger CVR/GMV TopK conflict, "
        "validating Innovation II motivation)"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_cfg(args: argparse.Namespace) -> GenConfig:
    cfg = GenConfig(n_samples=args.N, seed=args.seed)
    if args.preset is not None:
        for k, v in PRESETS[args.preset].items():
            setattr(cfg, k, v)
    if args.conflict_anticorr is not None:
        cfg.conflict_anticorr = float(args.conflict_anticorr)
    if args.cvr_dose_max is not None:
        cfg.cvr_dose_max = float(args.cvr_dose_max)
    if args.ov_dose_max is not None:
        cfg.ov_dose_max = float(args.ov_dose_max)
    return cfg


def _resolve_out_dir(args: argparse.Namespace, cfg: GenConfig) -> Path:
    if args.out is not None:
        return Path(args.out)
    preset_tag = args.preset or "custom"
    anticorr_tag = f"anticorr{cfg.conflict_anticorr:.2f}"
    return DEFAULT_OUT_ROOT / f"{preset_tag}_{anticorr_tag}"


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate the Criteo-MT7 semi-synthetic 8-arm uplift dataset."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help="Path to criteo-uplift-v2.1.csv(.gz)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory; defaults to data/semisynth/criteo-mt7/<preset>_anticorr<v>/")
    parser.add_argument("--N", type=int, default=500_000, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="realistic",
                        help="Dose-magnitude preset (Q-G2=c)")
    parser.add_argument("--conflict-anticorr", type=float, default=None,
                        help="Override δ↔β anti-correlation in [0,1] (default uses preset)")
    parser.add_argument("--cvr-dose-max", type=float, default=None,
                        help="Override CVR dose-max (advanced)")
    parser.add_argument("--ov-dose-max", type=float, default=None,
                        help="Override OV dose-max (advanced)")
    args = parser.parse_args(argv)

    cfg = _build_cfg(args)
    out_dir = _resolve_out_dir(args, cfg)

    print(f"[mt7] preset = {args.preset}, "
          f"conflict_anticorr = {cfg.conflict_anticorr:.2f}, "
          f"cvr_dose_max = {cfg.cvr_dose_max:.2f}, "
          f"ov_dose_max = {cfg.ov_dose_max:.3f}")
    print(f"[mt7] loading X from {args.csv} (target N = {args.N:,})")
    X = _load_criteo_X(args.csv, args.N, args.seed)
    print(f"[mt7] X shape = {X.shape}")

    bundle = generate(cfg, X)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "data.npz",
        X=bundle["X"],
        T=bundle["T"],
        Y_cvr=bundle["Y_cvr"],
        Y_gmv=bundle["Y_gmv"],
        mu_cvr_full=bundle["mu_cvr_full"],
        mu_gmv_full=bundle["mu_gmv_full"],
        Y_cvr_full=bundle["Y_cvr_full"],
        Y_gmv_full=bundle["Y_gmv_full"],
        tau_cvr=bundle["tau_cvr"],
        tau_gmv=bundle["tau_gmv"],
        pi_true=bundle["pi_true"],
    )

    meta = {
        "dataset": "criteo-mt7",
        "preset": args.preset,
        "source_csv": args.csv.name,
        "N": int(bundle["T"].shape[0]),
        "d": int(bundle["X"].shape[1]),
        "K": NUM_ARMS,
        "discount_pct": DISCOUNT_PCT.tolist(),
        "config": cfg.to_dict(),
    }
    with open(out_dir / "meta.json", "w") as fp:
        json.dump(meta, fp, indent=2)

    summary = summarize(bundle, cfg)
    with open(out_dir / "summary.txt", "w") as fp:
        fp.write(summary)

    print(f"[mt7] wrote bundle to {out_dir}")
    print("---- summary ----")
    print(summary)


if __name__ == "__main__":
    main()
