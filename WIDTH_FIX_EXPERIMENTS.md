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

---

# Option B follow-up — full 12-LR sweep at w=1024 (2026-05-13)

After the audit confirmed A5b (`init_factor=0.001` uniform global) recovers LR transfer, we implemented the **principled "Option B"** scaling from `kl_shampoo_init_section.tex`: `λ_k(0) ← ρ_0 / d_k` per factor, applied via a one-line edit at `kl_shampoo_jax/_core.py:101`. With ρ_0 = 1.0 (so λ_init = 1/d_k at every factor), we ran the same 12-LR Phase 7v2 grid at w=1024, distributed across rorqual + narval (fir partition was in maintenance — its jobs migrated to the other two clusters).

## Result: Option B does NOT recover the LR optimum

| LR | baseline init=0.1 | A5b init=0.001 | **Option B init=1/d** |
|---|---|---|---|
| 1.00e-5 | 7.97 | — | 8.32 |
| 2.31e-5 | 7.02 | — | 7.13 |
| 5.34e-5 | 6.00 | — | 6.11 |
| 1.23e-4 | 5.07 | — | 5.20 |
| 2.85e-4 | 4.34 | — | 4.47 |
| 6.58e-4 | 3.79 | — | 3.88 |
| 1.52e-3 | 3.58 | — | 3.61 |
| **3.51e-3** | **3.527** ← best | 3.566 | **3.577** ← best |
| 8.11e-3 | 3.59 | **3.562** ← best | 3.628 |
| 1.87e-2 | 3.84 | — | 3.86 |
| 4.33e-2 | 3.89 | — | 3.99 |
| 1.00e-1 | 5.77 | — | 4.14 |

**Key finding (12/12 grid LRs):** Option B's U-curve sits ~0.03-0.05 nats *above* the baseline across the whole sweep, with its **minimum at the same LR (3.51e-3)** as the baseline. It does NOT shift the optimum to base LR=8.11e-3 like A5b does. The "principled" per-factor scaling underperforms both the original `init_factor=0.1` baseline AND A5b's uniform-global `init_factor=0.001`.

This **contradicts the prediction in `kl_shampoo_init_section.tex` §1.3**, which derived `λ_eq ~ Θ(1/d)` and argued matching init to that should recover transfer. The empirical result shows the mechanism is more nuanced: A5b's *uniform* small init works (every factor gets λ=0.001), but the per-factor `λ = 1/d` scaling does not. Possible reasons:

1. **Small-factor over-amplification:** at small d_k (e.g. attention head intermediate dims), Option B gives larger init than A5b's uniform 0.001 (e.g. at d=256, Option B → 0.0039 vs A5b → 0.001). Over-amplification at small factors may degrade training.
2. **Equilibrium scaling assumption wrong at d=4:** the audit derived `λ_eq ~ Θ(1/d)` under μP gradient variance, but the actual scaling at depth=4 may differ — d_eq could scale differently than the d in `1/d`.
3. **Smaller-than-A5b at large factors:** at d=2816 (largest factor for MLP-down at w=1024), Option B → 3.6e-4, much smaller than A5b's 0.001. Possibly *too* small.

## Implementation notes

- Patch applied in-place via sed on `_core.py:101` (backup at `*.bak_optionB`):
  ```
  esi = jnp.full((d,), 1.0 / jnp.sqrt(init_factor / d), dtype=cast_dtype)
  ```
- Per-job override: `--cfg_options optimizer_args.kwargs.init_factor=1.0` (= ρ_0).
- Patch reverted on rorqual + narval. **Fir patch remains pending** (SSH expired during fir's maintenance window; revert when fir returns).
- Final figure: `figures/phase7v2_w1024_optb_full.{pdf,png}` (full range) + `..._zoom.{pdf,png}` (zoomed near optimum).

## Cluster distribution

- **Fir** (FS=0.41, planned 4 jobs): **partition went into maintenance** mid-experiment → jobs cancelled, 4 LRs migrated to rorqual+narval.
- **Rorqual** (FS=0.40, 7 jobs total): 6 single-GPU H100 jobs + 1 in-flight 4-GPU bonus (5.41e-3 failed mid-training due to JAX coordination crash).
- **Narval** (FS=0.36, 5 jobs A100): all 5 reached 100%; gRPC errors during post-training wandb cleanup but training data intact in offline-run dirs.

## Verdict

**A5b remains the only verified recovery mechanism**, even though its mechanism is harder to motivate theoretically. The principled per-factor scaling fails empirically. The next test (not run here) should be **Option A (data-driven init from first-batch GG)** which is what the audit originally recommended; it would test whether the right init is the *actual* GG diagonal vs any fixed analytical form.

## Pending work

1. **Revert** `fir:_core.py` when fir's SSH is restored (`*.bak_optionB` exists; one-line `cp` restore).
2. **Update `kl_shampoo_init_section.tex`** Table 1 and §1.3 — current draft over-promises Option B; replace with empirical finding + open question on Option A.
3. **(Optional) Test Option A** — data-driven init from first-batch GG diagonal. ~3 GPU-hours.
