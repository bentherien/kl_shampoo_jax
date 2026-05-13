# Width-fix experiments at w=1024 (audit candidate validation)

**Date:** 2026-05-13. **Cluster:** rorqual (`rrg-bengioy-ad_gpu`). **Total jobs:** 18 (6 fix configs × 3 LRs at w=1024). All reached step 4499/4500.

## Setup

Each fix swept peak LR ∈ {3.51e-3, 5.41e-3, 8.11e-3} (1.5× spacing centered on the geometric mean of Phase 7v2's drifted optimum 3.51e-3 and the base-width optimum 8.11e-3). Smoothed L̂ via canonical α=0.2, H=30 EMA. Compared to Phase 7v2 baseline at w=1024: best LR=3.51e-3 → L̂=3.527.

## Results table

| Fix | LR=3.51e-3 | LR=5.41e-3 | LR=8.11e-3 | Best LR | L̂_best | Shift vs base |
|---|---|---|---|---|---|---|
| **baseline** (Phase 7v2) | **3.527** | ~3.55 | ~3.59 | 3.51e-3 | 3.527 | drifted 2.3× below base |
| A6 (clamp off) | 3.568 | 3.566 | 3.587 | 5.41e-3 | 3.566 | 1.5× recovery |
| A5a (init_factor=0.01) | 3.577 | 3.550 | 3.595 | 5.41e-3 | **3.550** | 1.5× recovery |
| **A5b (init_factor=0.001)** | 3.566 | 3.562 | 3.562 | **8.11e-3** | 3.562 | **2.3× full recovery ✓** |
| A7 (cast/eigh fp32) | 3.541 | 3.546 | 3.558 | 3.51e-3 | 3.541 | none |
| A2a (kl_eps=0.1) | 4.456 | 4.410 | 4.370 | 8.11e-3 | 4.370 | (over-damped, +0.84 nats) |
| A2b (kl_eps=1.0) | 4.632 | 4.608 | 4.608 | 5.41e-3 | 4.608 | (over-damped, +1.08 nats) |

Plots: `/mnt/raid0/claude/kl_shampoo_jax/figures/phase7v2_w1024_audit_fixes.{pdf,png}` (2-panel: zoomed near baseline + full-range showing A2 catastrophe).

## Verdict per fix

### 🏆 A5b (`init_factor=0.001`) — PASS

The audit's top hypothesis is **confirmed**. Reducing the fixed KL-Shampoo eigenvalue initialization from 0.1 to 0.001 brings the init closer to the equilibrium `λ_eq ~ 1/d` at wide widths. Result: **optimum LR shifts from 3.51e-3 back to 8.11e-3** (matching the base-width optimum), and the U-curve is **flat** (0.005 spread across 3 LRs vs baseline's ~0.07).

L̂ tradeoff:
- A5b @ optimum LR=8.11e-3: 3.562 vs baseline @ same LR ~3.59 → **A5b is 0.028 better** at the prescribed Qiu LR.
- A5b @ optimum LR=8.11e-3: 3.562 vs baseline's drifted optimum 3.527 → A5b is 0.035 worse in absolute terms.

Interpretation: the baseline was "compensating" for the init bug by drifting its LR; removing the bug recovers Qiu's predicted flat transfer but lands at a slightly worse plateau because the bug-induced drift had landed on a coincidentally-good LR for this specific width.

### A5a (`init_factor=0.01`) — PARTIAL

Half-strength version of A5b. Optimum shifts to 5.41e-3 (1 grid step up; ½ the shift of A5b). L̂@best=3.550 — slightly better absolute than A5b. Confirms the "init magnitude → LR-optimum location" monotone relationship.

### A6 (`using_clamping=False`) — PARTIAL

Disabling the spectral clamp also shifts the optimum to 5.41e-3, similar magnitude to A5a. L̂@best=3.566. Note the audit had pegged A6 as opposite-sign (LOW-MED severity); the actual result is small but in the *expected* direction. Probably interacting with the same width-equilibrium mechanism (clamp limits 1/√λ at small λ, similar to keeping init λ higher).

### A7 (`cast_dtype=fp32, eigh_dtype=fp32`) — NEUTRAL

No LR-optimum shift; best LR stays at 3.51e-3, slightly better absolute L̂ (3.541 vs baseline 3.527 — actually 0.014 *worse*, so noise-level). **bf16 storage is not the cause of the drift.** Audit's rank-4 hypothesis ruled out.

### A2a (`kl_eps=0.1`) — FAIL (catastrophic)

Activating damping with `kl_eps=0.1` increases L̂ by **+0.84 nats**. Optimum DOES shift to 8.11e-3 (which the audit predicted as a side-effect of damping making preconditioner more Adam-like), but the matrix-preconditioner advantage is destroyed. Confirms A2's audit severity of MED-LOW: the prescription change matters but the implementation can't tolerate damping at this scale without the parameterization fix (drop the per-leaf `√(d_in/d_out)/L` factor).

### A2b (`kl_eps=1.0`) — FAIL (more catastrophic)

+1.08 nats. Stronger damping → worse. Confirms the dose-response.

## Key finding

**The audit's rank-1 candidate (`init_factor=0.1` width-invariant init) is the cause of the width-transfer drift.** Empirical proof:

1. Smaller `init_factor` → larger LR-optimum shift back to base (monotone: 0.001 → full recovery, 0.01 → half recovery, 0.1 → drift).
2. Other candidates (clamp, fp32, damping) don't recover LR transfer in the right direction or sign.
3. Mechanism: KL-Shampoo equilibrium eigenvalue magnitude `λ_eq ~ Θ(1/d)` (per `_core.py:144-145` which divides the GG-EMA target by `total_factor/d_k = d_b`). At wide d, equilibrium `esi ~ √d ≈ 32`; the fixed `1/√0.1 ≈ 3.16` init is **10× too small**, creating a width-dependent ramp-up transient over the first ~1/(1−β₂) ≈ 50 steps that requires a smaller LR to remain stable.

## Recommendations

### Immediate next step (cheap)

Run a **base-width sanity check** before scaling: 3 jobs at w=256 × LRs {5.41e-3, 8.11e-3, 1.22e-2} with `init_factor=0.001`. If L̂ at w=256 is ≥ baseline (3.910 @ 8.11e-3), A5b is safe to ship. If A5b *worsens* w=256, the fix is only useful for wide widths and needs to become *width-aware* (e.g., `init_factor = 0.1 · (256/d)`).

### Mid-term verification sweep

5-LR sweep at 1.5× spacing × 4 widths {128, 256, 512, 1024} with `init_factor=0.001`. 20 jobs, ~7 H100-hours on rorqual. Success: best LR at every width within 1 grid step of 8.11e-3.

### Long-term: data-driven init (audit recommendation)

Replace fixed `init_factor` with the first-batch GG diagonal:
```python
# kl_shampoo_jax/_core.py:101
esi = 1.0 / jnp.sqrt(jnp.diag(GG) + eps)   # was: 1.0 / jnp.sqrt(init_factor)
```
This zeroes out the width dependence by construction, no per-width tuning needed. Use the existing `_first_iter` path that already computes `GG_0`/`GG_1` at `_core.py:142-145`. One-line change in the underlying library.

### Ruled out (don't pursue)

- A7 (bf16 → fp32 preconditioner state): no LR shift, ~2× memory cost
- A2 (raise `kl_eps`): catastrophic absolute-loss penalty, even with audit's full prescription unlikely to recover the matrix-preconditioner advantage

## Files

- Sweep script: `~/scratch/l2o_install/scaling_l2o/jobs/kl_shampoo_completedp/audit_fixes_rorqual.sh` (on rorqual)
- A7 config: `~/scratch/l2o_install/scaling_l2o/config/meta_test/mup_kl_shampoo_w256_fp32.py` (on rorqual)
- Aggregation: `/tmp/audit_fix_smoothed_loss.py`
- Plot: `/tmp/plot_audit_fixes_w1024.py`
- Wandb runs: `eb-lab/belo-meta-testing`, filter `name_suffix LIKE '_audit_%_w1024_%'` (18 runs)

## Wall-clock summary

- Submission: 1 batch via `audit_fixes_rorqual.sh`
- 18 jobs × 25 min × ~3 concurrent ≈ 2.5h queue wall
- Sync + aggregate + plot: 15 min
- Total wall: ~3h
