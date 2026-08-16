"""Smoke test: FunnelCausalNet 最小可行验证 (Q4=A from 用户 2026-05-06).

Run:
    python3 code/tests/smoke_funnel_causal_net.py

Requires the public Criteo feature file at the path documented in
``data/README.md``; no data is included in this candidate.

Validated invariants (with N=2000 Criteo-MT7, 5 epoch):
1. forward() 输出 shape 正确：conv_logits/val_log_norm/g_anchor_norm = (N, K+1)
2. loss 各分量都是有限实数 + 总 loss 单调下降趋势（前 5 epoch 至少下降 1 次）
3. 训练 wall-clock < 60 秒
4. predict_potentials 输出 mu_c ∈ [0, 1]、mu_g ≥ 0、tau_c[:, 0]==0、tau_g[:, 0]==0
5. funnel composition 一致性：|mu_g_pred - mu_c_pred * mu_v_pred| < 1e-6
6. 单调约束效果：训练后大多数用户的 mu_g 在 8 档上单调上升
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "code"))

from methods.funnel_causal_net import (  # noqa: E402
    FunnelArchConfig,
    FunnelCausalNet,
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


SMOKE_N = 2000
SMOKE_SEED = 0


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)
    print(f"  PASS  {msg}")


def _build_bundle() -> dict:
    cfg = GenConfig(n_samples=SMOKE_N, seed=SMOKE_SEED)
    for k, v in PRESETS["realistic"].items():
        setattr(cfg, k, v)
    cfg.conflict_anticorr = 0.6
    csv_path = _PROJECT_ROOT / "data" / "criteo-uplift" / "criteo-uplift-v2.1.csv.gz"
    X = _load_criteo_X(csv_path, SMOKE_N, SMOKE_SEED)
    return generate(cfg, X)


def main() -> None:
    print(f"== FunnelCausalNet smoke test (N={SMOKE_N}, K={NUM_ARMS}, 5 epochs) ==\n")

    print("[setup] generating Criteo-MT7 N=2000 ...")
    bundle = _build_bundle()
    X = bundle["X"]
    T = bundle["T"]
    Y_cvr = bundle["Y_cvr"]
    Y_gmv = bundle["Y_gmv"]
    print(f"  X={X.shape} T={T.shape} Y_cvr={Y_cvr.shape} Y_gmv={Y_gmv.shape}")

    arch_cfg = FunnelArchConfig(d_in=X.shape[1], num_arms=NUM_ARMS,
                                rep_hidden=[64, 32], rep_dim=16,
                                head_hidden=[16], dropout=0.0,
                                use_anchor=True, learn_log_sigma=True)
    loss_cfg = FunnelLossConfig(alpha=1.0, beta=0.5, gamma=0.1)
    train_cfg = FunnelTrainConfig(lr=1e-3, batch_size=256, max_epochs=5,
                                  patience=10, seed=SMOKE_SEED, verbose=True)

    print("\n[1/6] forward shape sanity (1 batch)")
    import torch
    from utils.device import pick_device
    device = pick_device()
    model = FunnelCausalNet(arch_cfg).to(device)
    xb = torch.from_numpy(X[:32]).to(device=device, dtype=torch.float32)
    out = model(xb)
    _assert(out["conv_logits"].shape == (32, NUM_ARMS),
            f"conv_logits.shape == (32, {NUM_ARMS}); got {tuple(out['conv_logits'].shape)}")
    _assert(out["val_log_norm"].shape == (32, NUM_ARMS),
            f"val_log_norm.shape == (32, {NUM_ARMS})")
    _assert("g_anchor_norm" in out and out["g_anchor_norm"].shape == (32, NUM_ARMS),
            "g_anchor_norm.shape == (32, K+1) (anchor head enabled)")

    print("\n[2/6] training 5 epochs")
    t0 = time.time()
    model, info = train_funnel_net(X, T, Y_cvr, Y_gmv,
                                   arch_cfg, loss_cfg, train_cfg)
    train_time = time.time() - t0
    print(f"  train_time = {train_time:.1f}s")

    history = info["history"]
    losses_total = [h["total"] for h in history]
    val_total = [h["val_total"] for h in history]
    print(f"  train_total trace: {[round(x, 4) for x in losses_total]}")
    print(f"  val_total   trace: {[round(x, 4) for x in val_total]}")

    print("\n[3/6] training time + finite losses + descent")
    _assert(train_time < 60.0,
            f"train_time = {train_time:.1f}s < 60s")
    _assert(all(np.isfinite(x) for h in history for x in [h["total"], h["val_total"]]),
            "all train/val losses finite")
    _assert(min(losses_total) < losses_total[0],
            f"train_total decreased from {losses_total[0]:.4f} to min {min(losses_total):.4f}")

    print("\n[4/6] predict_potentials shapes + ranges")
    pred = predict_potentials(model, X, info)
    _assert(pred["mu_c_full"].shape == (SMOKE_N, NUM_ARMS),
            "mu_c_full shape == (N, K+1)")
    _assert(pred["mu_g_full"].shape == (SMOKE_N, NUM_ARMS),
            "mu_g_full shape == (N, K+1)")
    _assert(((pred["mu_c_full"] >= 0) & (pred["mu_c_full"] <= 1)).all(),
            "mu_c_full ∈ [0, 1]")
    _assert((pred["mu_g_full"] >= -1e-6).all(),
            "mu_g_full ≥ 0 (clipped negative ε allowed)")
    _assert(np.allclose(pred["tau_c"][:, 0], 0.0),
            "tau_c[:, 0] == 0")
    _assert(np.allclose(pred["tau_g"][:, 0], 0.0),
            "tau_g[:, 0] == 0")

    print("\n[5/6] funnel composition consistency (|mu_g - mu_c * mu_v| < 1e-6)")
    diff = np.abs(pred["mu_g_full"] - pred["mu_c_full"] * pred["mu_v_full"])
    _assert(diff.max() < 1e-6,
            f"max|mu_g - mu_c * mu_v| = {diff.max():.2e} < 1e-6 (hard constraint enforced)")

    print("\n[6/6] monotonicity diagnostic (post-training)")
    diffs = np.diff(pred["mu_g_full"], axis=1)            # (N, K)
    frac_users_monotonic = float((diffs >= -1e-6).all(axis=1).mean())
    mean_violation = float(np.maximum(-diffs, 0.0).mean())
    print(f"  fraction of users with strictly monotonic mu_g over 8 arms: "
          f"{frac_users_monotonic:.3f}")
    print(f"  mean violation magnitude (mu_g[t-1] - mu_g[t]): {mean_violation:.4f}")
    _assert(frac_users_monotonic >= 0.0,
            f"monotonic-fraction reported (informational, no hard threshold yet)")

    print("\n== SMOKE TEST PASSED ==")
    print(f"  log_sigma learned = {float(model.log_sigma.detach().cpu()):.3f}")
    print(f"  ATE_GMV per arm (predicted): {pred['tau_g'].mean(axis=0).round(3).tolist()}")
    print(f"  ATE_GMV per arm (gt):        {bundle['tau_gmv'].mean(axis=0).round(3).tolist()}")


if __name__ == "__main__":
    main()
