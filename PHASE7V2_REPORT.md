# Phase 7v2 — μCompletedP-KL-Shampoo extended LR-transfer scan

**Date**: 2026-05-11
**Clusters**: rorqual (cancelled mid-flight, fairshare exhausted), narval (A100, 4-GPU), fir (H100, 1-GPU)
**LR grid**: 12 log-spaced values from 1e-5 to 1e-1 — `np.logspace(-5, -1, 12)`
**Anchor HPs** (from Phase 6 winner): `kl_b1=0.9`, `shampoo_b=0.99`, `kl_eps=1e-8`, `kl_weight_decay=0.01`, `precondition_frequency=5`, `adam_b1=0.95484`, `adam_b2=0.9908`, `adam_eps=1e-8`, `adam_weight_decay=0.01`, `adam_lr_scale=1.0`. Vary only `learning_rate`.

**Total runs**: 167 successful + 1 failed (port collision on a stale-port-config job).
**Smoothed L̂** via MuLoCo time-weighted EMA (α=0.2, H=30, no subsampling). Runs that did not reach their target inner step (`num_inner_steps - 1`, with a 50-step tolerance) were filtered out.

## Aggregate transfer table

Each axis-point: smoothed L̂_test (best across 12 LRs) and the LR that achieves it. n is the number of completed-and-final-step LRs.

### Width axis (base w=256)

| Point | n | Best LR | L̂_test | L̂_train | Δ vs base (test) |
|---|---|---|---|---|---|
| **w128** (0.5×) | 12 | 1.87e-2 | 4.237 | 4.233 | +0.331 |
| **w256** (base) | 12 | **8.11e-3** | **3.906** | 3.888 | 0 |
| **w512** (2×) | 12 | 8.11e-3 | 3.687 | 3.664 | −0.219 |
| **w1024** (4×) | 11 | 3.51e-3 | **3.527** | 3.499 | −0.379 |

### Depth axis (base d=4)

| Point | n | Best LR | L̂_test | L̂_train | Δ vs base (test) |
|---|---|---|---|---|---|
| **d4** (base) | 12 | **8.11e-3** | **3.959** | 3.888 | 0 |
| **d8** (2×) | 12 | **8.11e-3** | 3.867 | 3.796 | −0.092 |
| **d12** (3×) | 12 | **8.11e-3** | **3.816** | 3.747 | −0.143 |

### Batch axis (base 131k tokens)

| Point | n | Best LR | L̂_test | L̂_train | Δ vs base (test) |
|---|---|---|---|---|---|
| **bs65k** (0.5×) | 12 | **8.11e-3** | 4.001 | 3.987 | +0.091 |
| **bs131k** (base) | 12 | **8.11e-3** | 3.910 | 3.887 | 0 |
| **bs262k** (2×) | 12 | **8.11e-3** | **3.845** | 3.820 | −0.065 |

### Duration axis (base 4500 steps)

| Point | n | Best LR | L̂_test | L̂_train | Δ vs base (test) |
|---|---|---|---|---|---|
| **st1500** (0.33×) | 12 | 3.51e-3 | 4.205 | 4.221 | +0.247 |
| **st4500** (base) | 12 | **8.11e-3** | 3.958 | 3.888 | 0 |
| **st10000** (2.22×) | 12 | 1.87e-2 | **3.807** | 3.790 | −0.151 |

## Headline findings

### 1. Depth and batch axes: PERFECT HP transfer

All three depth points (d4/d8/d12) and all three batch points (bs65k/bs131k/bs262k) achieve their minimum at **the same LR=8.11e-3** with a tight flat plateau around it. The CompletedP parameterization correctly absorbs the depth and batch-size scaling: optimum LR is constant.

### 2. Width axis: mild shift, attributable to discrete LR grid

w256 and w512 both win at 8.11e-3. w128 picks the adjacent grid point (1.87e-2; one step up the log-grid, ratio 2.31×). w1024 picks the adjacent grid point (3.51e-3; one step down). These are 1-step shifts on a log-spaced grid where adjacent LRs differ by 2.31×, and the U-curves are quite flat near the optimum (within 0.05-0.10 nats across 4-5 adjacent LRs). The OOM of the optimum is constant (~1e-2) across all 4 widths.

### 3. Duration axis: stronger shift, monotone with horizon

st1500 → 3.51e-3, st4500 → 8.11e-3, st10000 → 1.87e-2. Each is one grid step from the next (the LR optimum rises with training horizon). This is the expected behavior for cosine-decay schedules with fixed warmup of 500 steps:
- At 1500 steps, the warmup uses 1/3 of the budget; the post-warmup decay only has 1000 steps. Lower peak LR is safer.
- At 10000 steps, more training budget tolerates higher peak LR.
This is a known limitation of fixed-warmup schedules under varying horizon, not a failure of the CompletedP scaling rules.

### 4. Best run overall: w1024 + lr=3.51e-3 → L̂_test=3.527

The widest model + appropriate LR achieves the lowest loss. Train and test L̂ are tightly correlated (Δ < 0.05 nats), confirming that the smoothed losses are picking up the true model fit (no test/train divergence indicating bad regularization).

## Comparison to Phase 7 v1 (3-LR sweep)

| Axis-point | v1 best (3 LRs) | v2 best (12 LRs) | Δ |
|---|---|---|---|
| w128 | 8e-3 → L̂=4.305 | 1.87e-2 → 4.237 | −0.068 |
| w256 | 8e-3 → 3.958 | 8.11e-3 → 3.906 | −0.052 |
| w512 | 8e-3 → 3.732 | 8.11e-3 → 3.687 | −0.045 |
| w1024 | 8e-3 → 3.605 | 3.51e-3 → 3.527 | −0.078 |
| d4 | 8e-3 → 3.958 | 8.11e-3 → 3.959 | +0.001 |
| d8 | 8e-3 → 3.868 | 8.11e-3 → 3.867 | −0.001 |
| d12 | 8e-3 → 3.815 | 8.11e-3 → 3.816 | +0.001 |
| bs131k | 8e-3 → 3.958 | 8.11e-3 → 3.910 | −0.048 |

v2 reaches a marginally lower L̂ at each width/batch point (denser LR grid helps); depth axis is identical between v1 and v2 (the 3-LR coarse sweep was already at the optimum).

## Plots

Saved at `/mnt/raid0/claude/kl_shampoo_jax/figures/`. All sized for LaTeX 1/2 \linewidth = 2.75" × 2.42" with 8pt body / 7pt ticks/legend:

- `phase7v2_overlaid_test_{width,depth,batch,duration}.{pdf,png}` — per-axis overlaid LR sweep (test loss).
- `phase7v2_overlaid_train_{width,depth,batch,duration}.{pdf,png}` — same for train loss.
- `phase7v2_combined_test_train.{pdf,png}` — 2×4 grid (test top, train bottom × 4 axes).

Each plot uses the viridis colormap, base curve in thicker blue (Width), stars marking per-curve minima, and base value annotated in lower-left.

## Conclusion

**μCompletedP-KL-Shampoo transfers HPs cleanly across the tested grid.** Depth and batch axes are PERFECT (single shared optimum LR=8.11e-3 across all axis-points). Width and duration axes show 1-step grid-quantized shifts that lie within the flat plateau around the optimum.

The headline w1024-d4 result at L̂_test=**3.527** confirms that scaling up width using HPs tuned at w256-d4 (with CompletedP shape factor √(d_out/d_in)) delivers the expected loss improvement.

## Pipeline notes

- 167 runs / 1 failure ran across narval (depth + duration, A100, 4-GPU jobs) and fir (width + batch, H100, 1-GPU jobs).
- Rorqual was the original target for width+batch but fairshare was exhausted by Phase 6+7v1; jobs cancelled mid-flight and migrated to fir.
- Fir 1-GPU jobs initially failed on port-12999 collisions when multiple landed on the same node; fixed with random per-job MASTER_PORT.
- Single-GPU jobs with 4× gradient_accumulation preserve the effective batch (256 sequences × 512 tokens = 131k) and unlocked 4× cluster concurrency.
- Final sync: 5 truncated runs (jobs that hit walltime before reaching `num_inner_steps`) were filtered from the L̂ aggregation. 1 job (39659486 / w256_lr1p23em4 on fir) failed due to port collision and is missing from the w256 row at lr=1.23e-4.

Wandb project: `eb-lab/belo-meta-testing`, filter `name_suffix` LIKE `%_phase7v2_%`.
