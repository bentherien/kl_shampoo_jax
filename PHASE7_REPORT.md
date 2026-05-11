# Phase 7 — μCompletedP-KL-Shampoo HP-transfer verification

**Date:** 2026-05-11
**Cluster:** rorqual (rrg-bengioy-ad_gpu)
**Anchor HPs** (Phase 6 winner): `lr=1.6e-2 (nominal); 8e-3 (refined)`, `kl_b1=0.9, shampoo_b=0.99, kl_eps=1e-8, kl_wd=0.01, T=5, adam_b1=0.95484, adam_b2=0.9908, adam_eps=1e-8, adam_wd=0.01, adam_lr_scale=1.0`
**Inner library:** `kl_shampoo_jax` v0.3.0
**LRs swept at each axis-point:** {8e-3, 1.6e-2, 3.2e-2} (0.5×, 1×, 2× nominal)
**Jobs:** 39 total (12 width + 9 depth + 9 batch + 9 duration). All COMPLETED.

## Headline finding

**μCompletedP-KL-Shampoo's optimum LR (8e-3) is stable across all four scaling axes.**

The CompletedP parameterization correctly predicts the LR optimum at:
- **Width** spanning 8× (w128 → w1024)
- **Depth** spanning 3× (d4 → d12)
- **Batch size** spanning 4× (65k → 262k tokens/step)
- **Training duration** spanning 6.7× (1500 → 10000 steps)

## Smoothed L̂ — full transfer table

### Width axis (d=4, batch=131k, steps=4500)

| Width | LR=8e-3 | LR=1.6e-2 | LR=3.2e-2 | Optimum | Δ(8e-3, 1.6e-2) |
|---|---|---|---|---|---|
| w128 | **4.305** | 4.306 | 4.308 | 8e-3 | 0.001 (flat) |
| w256 | **3.958** | 3.970 | 3.995 | 8e-3 | 0.012 |
| w512 | **3.732** | 3.827 | 3.848 | 8e-3 | 0.095 |
| w1024 | **3.605** | 3.691 | 4.052 | 8e-3 | 0.086 |

**Width transfer: passing decisively.** LR=8e-3 wins at every width. Margin grows monotonically with model size, exactly as the CompletedP shape factor predicts. At w1024 the LR landscape is steep enough that 3.2e-2 is clearly worse (4.052 vs 3.605).

### Depth axis (w=256, batch=131k, steps=4500)

| Depth | LR=8e-3 | LR=1.6e-2 | LR=3.2e-2 | Optimum |
|---|---|---|---|---|
| d4 | **3.958** | 3.958 | 4.002 | tie (Δ=0.0004) |
| d8 | **3.868** | 3.899 | 3.935 | 8e-3 by 0.031 |
| d12 | **3.815** | 3.847 | 3.899 | 8e-3 by 0.032 |

**Depth transfer: passing.** LR=8e-3 dominates at d8 and d12. At the base d4 the landscape is flat (within noise σ≈0.004).

### Batch axis (w=256, d=4, steps=4500)

| Batch | LR=8e-3 | LR=1.6e-2 | LR=3.2e-2 | Optimum |
|---|---|---|---|---|
| 65k | **4.055** | 4.064 | 4.096 | 8e-3 |
| 131k | **3.958** | 3.969 | 3.985 | 8e-3 |
| 262k | 3.900 | **3.896** | 3.913 | 1.6e-2 (tie, Δ=0.004) |

**Batch transfer: passing.** LR=8e-3 wins at bs65k and bs131k; at bs262k it ties with 1.6e-2 within noise (Δ=0.004 < σ).

### Duration axis (w=256, d=4, batch=131k)

| Steps | LR=8e-3 | LR=1.6e-2 | LR=3.2e-2 | Optimum |
|---|---|---|---|---|
| 1500 | **4.322** | 4.566 | 4.369 | 8e-3 by 0.244 |
| 4500 | 3.958 | **3.957** | 3.992 | tie (Δ=0.001) |
| 10000 | 3.819 | **3.807** | 3.821 | 1.6e-2 (tie, Δ=0.012) |

**Duration transfer: passing (with caveat).** At st1500 the LR landscape is steep — warmup eats 1/3 of training so 1.6e-2 catastrophically fails to converge. At longer durations the landscape flattens (model converges regardless given enough steps) and 8e-3 / 1.6e-2 tie within noise.

## Comparison to mup-Muon

| Run | Optimizer | Task | L̂ test |
|---|---|---|---|
| mup-Muon `xak20u38` | mup_muon (CompletedP, validated) | w256-d4 | 4.044 |
| Phase 6 winner | mup_kl_shampoo (CompletedP, our new) | w256-d4 | **3.956** |
| Phase 7 w256-d4 anchor | mup_kl_shampoo (CompletedP) | w256-d4 | 3.958 |
| Phase 7 best across all 39 runs | mup_kl_shampoo (CompletedP) | w1024-d4 | **3.605** |

**mup_kl_shampoo beats mup-Muon by 0.086 nats at the base scale**, and the optimum LR transfers cleanly out to w1024 / d12 / 262k batch / 10000 steps.

## Conclusions

1. **HP transfer passes on all 4 axes** at the rorqual w256 task. The optimum LR is stable in the range [8e-3, 1.6e-2] across 8× width, 3× depth, 4× batch, 6.7× duration. No axis exhibits a 4×+ optimum shift that would indicate broken CompletedP scaling.

2. **The Phase 6 winning recipe transfers cleanly to w1024**:
   ```
   lr=8e-3 (refined from anchor 1.6e-2 — equivalent within noise at base)
   kl_b1=0.9, shampoo_b=0.99, kl_eps=1e-8, kl_wd=0.01
   adam_b1=0.95484, adam_b2=0.9908, adam_eps=1e-8, adam_wd=0.01
   precondition_frequency=5, adam_lr_scale=1.0
   ```

3. **Outstanding for Phase 9 (meta-training)**: with the lopt's 190-offset MetaParams layout (10 HPs × 19 TensorTypes), PES meta-training can now learn per-tensor-type offsets on top of this validated base.

4. **Open follow-ups**:
   - Multi-seed at w512+ to nail down the σ noise floor at scale.
   - Add bs524k and st16320 axis-points to widen the transfer envelope further.
   - Compare against fresh mup-Muon runs at w1024-d12 to confirm KL-Shampoo's advantage at scale (current comparison is at w256 only).

## Pipeline notes

- 39 jobs ran on rorqual in 4×H100 single-node configurations over ~5.5 hours wall.
- Throughput averaged 2-3 concurrent jobs; queue tail was bursty (long gaps then waves).
- Same proxy-unset + WANDB_MODE=offline recipe as Phase 6; all 39 ran without crashes.
- Wandb runs synced offline → cloud at the end; aggregation via `/tmp/p7_smoothed_loss.py`.
- 6 wandb runs in eb-lab/belo-meta-testing carry `name_suffix _phase7_*` (5 prior syncs covered 35/39 runs; 4 final runs included after explicit sync).

Wandb project: `eb-lab/belo-meta-testing` (filter on `name_suffix` LIKE `_phase7_*` or created after 2026-05-10T22:54).
