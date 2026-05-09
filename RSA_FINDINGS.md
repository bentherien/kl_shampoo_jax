# RSA Diagnosis: Why KL-Shampoo plateaus at 4.20 vs mup-Muon 4.02

10 parallel /rsa agents investigated three axes (HP-search, paper-vs-impl, AdamW HP tuning). Aggregated findings ranked below.

## Ranked hypothesis table

| Rank | Agent | Hypothesis | Severity | Cost | Verifies via |
|------|-------|-----------|----------|------|--------------|
| 1 | **C2** | Per-leg WD missing — chain-level `add_decayed_weights(wd)` bakes the same coefficient into both KL and Adam legs; mup-Muon uses `muon_wd=0.354` AND `adam_wd=0.0931` (3.8× different). Phase-2 WD optimum at 0.01 is a saddle of the constrained problem, not the unconstrained optimum. | **HIGH** | 9 jobs | 2-D `(kl_wd, adam_wd)` grid after code patch |
| 2 | **A2** | Same root cause as C2 — Phase-2 WD sweep tested wd=0.354, but that forced both legs to 0.354 (over-regularizing embeddings). Confounded sweep, not coverage gap. | **HIGH** | merges with C2 | (covered by C2) |
| 3 | **C1** | Adam betas at Optax defaults (0.9/0.999); mup-Muon uses tuned 0.95484/0.9908. With ~45-88% of params in the Adam leg (embedding+unembedding), wrong betas lag variance estimation. | **MEDIUM-HIGH** | 4 jobs | Config-only flip; 4-job 2×2 grid |
| 4 | **C3** | Shared LR=1.6e-2 is appropriate for KL-Shampoo's matrix-preconditioned step but ~5× standard AdamW LR for embeddings (sign-magnitude direction × unit-norm). Coupled with C2 — together they control Adam-leg effective dynamics. | **MEDIUM** | 4 jobs | Add `adam_lr_scale` kwarg, sweep ∈ {0.1, 0.3, 0.5, 1.0} |
| 5 | **B1** | bf16 precision: JAX `update_eigen_value` is MORE careful than reference (promotes EMA to fp32). True silent gap is matmul accumulation precision in `tensordot`/`einsum` — JAX defaults to fp32-accumulate on H100, but explicit `Precision.HIGHEST` is not set. Probably non-issue but worth a single-job control. | **MEDIUM** | 1 job | Run with `cast_dtype=jnp.float32` end-to-end |
| 6 | **B2** | Eigh/QR sign convention — could accumulate bias over 450 QR recomputes if drift is non-zero-mean. Existing test_against_torch only exercises 5 recomputes. Worst-case ~0.05 nats; random-walk ~0.002 nats. | **MEDIUM** (probe needed) | extend test | 500-step `Q_jax.T @ Q_torch` SVD probe |
| 7 | **A4** | Coordinate-descent missed HP interactions. Phase-1 LR was at fixed β=0.98; the optimum LR may shift with β. Worst-case ~0.02 nats. | **LOW-MEDIUM** | 9 jobs | 2-D LR×β grid |
| 8 | **A1** | LR sweep resolution sufficient — √2 spacing on a flat plateau. Best-case interior interpolation ≤ 0.005 nats. | **LOW** | 5 jobs | (deprioritize) |
| 9 | **A3** | Schedule shape (cosine vs WSD) — both mup-Muon and KL-Shampoo used identical cosine schedule; can't differentially close the gap. | **LOW** | 3 jobs | (deprioritize) |
| 10 | **B3** | First-step WD discrepancy — params shrink by 1 part in 6250 once. Cannot move loss by >1e-3. | **LOW** | 0 (analytical) | (skip) |

## Aggregated insight

**A2 and C2 are the same hypothesis** — the per-leg WD asymmetry. The fix is one code patch (add `adam_weight_decay` kwarg to `kl_shampoo_with_adamw`).

**C1 + C2 + C3 likely interact**: at the right `adam_lr_scale`, the optimal `(kl_wd, adam_wd)` shifts toward mup-Muon's (0.354, 0.0931). At the right `adam_b1/b2`, the variance estimator tracks correctly. Doing all three at once is the right scoping.

**B1 / B2 are unlikely the dominant cause** but cheap probes worth running in parallel with the main C-axis sweep.

**A1 / A3 / B3 are deprioritized** — provably small effects.

## Top-3 actions for Phase C

1. **Patch `kl_shampoo_jax/_routing.py`**: add `adam_weight_decay: Optional[float]` and `adam_lr_scale: float = 1.0`. When `adam_weight_decay is not None`, build per-leg WD chains and remove chain-level `add_decayed_weights`. Apply `optax.scale(adam_lr_scale)` inside the adam leg.
2. **Plumb new kwargs through scaling_l2o**: `kl_shampoo_optimizer` factory + `config/optimizer/kl_shampoo.py` + `config/meta_test/kl_shampoo_w256_baseline.py`.
3. **Phase-8 sweep** (~16-20 jobs) — joint grid:
   - **adam_weight_decay** ∈ {0.01, 0.0931, 0.3} (3 points)
   - **adam_b1, adam_b2** ∈ {(0.9, 0.999), (0.955, 0.9908)} (2 points)
   - **kl_weight_decay** ∈ {0.01, 0.1} (2 points)
   - **adam_lr_scale** ∈ {1.0, 0.3} (2 points; reduce later)
   = 24 jobs (or trim by skipping cells unlikely to win)

   Initial 12-job grid: 3 × adam_wd × 2 × {betas} × 2 × adam_lr_scale at fixed kl_wd=0.01. Then expand best winner with kl_wd∈{0.01, 0.1, 0.354}.

## Predicted outcome

If C2 is correct, expect best test loss ≤ 4.10 (closes ~50% of the 0.18-nat gap).
If C2+C3+C1 jointly contribute, expect best ≤ 4.05 — near parity with mup-Muon at vanilla SP.
If gap remains ≥ 0.10 after all three: residual is structural (MuP scaling), out of current scope.

## Cheap probes to run in parallel

- **B1 control**: 1 job, `cast_dtype=jnp.float32` end-to-end. Verifies bf16 isn't the issue.
- **B2 probe**: extend `test_against_torch.py` to 500 steps + SVD check. No GPU cost.

## Files to modify (Phase C)

| Path | Change |
|------|--------|
| `/mnt/raid0/claude/kl_shampoo_jax/kl_shampoo_jax/_routing.py` | Add `adam_weight_decay`, `adam_lr_scale` kwargs; per-leg chain branch |
| `/mnt/raid0/claude/kl_shampoo_jax/tests/test_routing.py` | Add per-leg WD test |
| `/home/btherien/raid0/claude/github/scaling_l2o/src/opt/new_optimizers.py` | Plumb kwargs through `kl_shampoo_optimizer` |
| `/home/btherien/raid0/claude/github/scaling_l2o/config/optimizer/kl_shampoo.py` | Expose `adam_weight_decay`, `adam_lr_scale` |
| `/mnt/raid0/claude/kl_shampoo_jax/HP_SWEEP_REPORT.md` | Append Phase 8 |
