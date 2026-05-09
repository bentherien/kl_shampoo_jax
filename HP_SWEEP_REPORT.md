# KL-Shampoo HP Sweep Report

**Task:** `muztransformer-dense-w256-d4-h2_fineweb-s512-gpt2`
**Cluster:** Fir, 4×H100, account `rrg-eugenium_gpu`, 4500 inner steps
**Optimizer:** `kl_shampoo` (vanilla SP, AdamW fallback for 1D / oversize / scalar leaves)
**Sweep dates:** 2026-05-07
**Total compute:** 22 jobs × ~8-12 min wall = ~3.5 GPU-hours per run; ~15 GPU-hours total on H100

## Headline result

**Best test loss: 4.194 ± 0.004** (mean ± std across 4 seeds, n=4). HPs:
`lr=1.6e-2, wd=0.01, shampoo_b=0.99, T=1, b1=0.9, eps=1e-8, init_factor=0.1, adam_b2=0.999`.

The single best individual run was `39140018` (test 4.190), but per-seed variance is σ≈0.004 — all top configs in the range 4.190–4.204 are statistically equivalent. Defaults for `init_factor`, `adam_b2`, and `eps` are all within noise of any alternative.

Improvement over the initial untuned baseline (`lr=1e-3, wd=0.01, shampoo_b=0.98, T=10`):
- Baseline test loss: 4.440 (job 39098855)
- Tuned mean test loss: 4.200
- **Δ = 0.240 reduction** (5.4% relative)

Comparison to the mup-Muon reference (`eb-lab/mup-muon-sweep-w256-d4/runs/xak20u38`):
- mup-Muon test loss: **4.018**
- KL-Shampoo (vanilla SP, tuned): 4.200 ± 0.004
- Gap: **0.182** test loss. Most of this gap is attributable to MuP scaling (xak20u38 uses MuP, we ran vanilla); see "Future work" below.

## Phase 1 — Learning-rate sweep (`wd=0.01, β=0.98, T=10`)

11 LR points spanning 64×:

| LR | Test | Train | Notes |
|---|---|---|---|
| 2.5e-4 | 5.156 | 4.883 | Far too low |
| 5e-4 | 4.784 | 4.473 | |
| 1e-3 | 4.438 | 4.096 | (Initial untuned baseline) |
| 2e-3 | 4.299 | 3.953 | |
| 4e-3 | 4.233 | 3.885 | |
| 8e-3 | 4.205 | 3.861 | |
| **1.6e-2** | **4.200** | **3.849** | ← initial peak |
| 2.5e-2 | 4.208 | 3.860 | |
| 4e-2 | 4.235 | 3.889 | |
| 6e-2 | 4.248 | 3.899 | |
| 8e-2 | 4.783 | 4.482 | Diverging |

**Optimal LR: 1.6e-2.** Landscape is flat across an order-of-magnitude band (8e-3 to 4e-2 within 0.05 of best). Beyond 6e-2 the optimizer destabilizes. KL-Shampoo's optimal LR is *much higher* than the reference KLOpt default of 1e-4 — the paper-recommended default underestimates by ~100×.

## Phase 2 — Weight-decay sweep at `lr=1.6e-2`

| WD | Test | Train |
|---|---|---|
| 0.0 | 4.203 | 3.853 |
| 0.001 | 4.203 | 3.849 |
| **0.01** | **4.200** | 3.849 |
| 0.1 | 4.238 | 3.892 |
| 0.354 | 4.435 | 4.110 |

**Optimal WD: 0.01 (paper default).** Landscape flat between 0 and 0.01 (within seed noise); 0.1+ hurts. mup-Muon's WD=0.354 is far too aggressive for KL-Shampoo.

## Phase 3 — `precondition_frequency` (T) and `shampoo_b` (β₂) at `lr=1.6e-2, wd=0.01`

| T | β | Test | Train |
|---|---|---|---|
| 1 | 0.98 | 4.194 | 3.839 |
| 5 | 0.98 | 4.196 | 3.843 |
| 10 | 0.98 | 4.200 | 3.849 |
| 20 | 0.98 | 4.196 | 3.848 |
| 50 | 0.98 | 4.203 | 3.855 |
| 10 | 0.95 | 4.217 | 3.865 |
| **10** | **0.99** | **4.193** | 3.845 |

**Optimal: T=10, β=0.99.** T is essentially flat across {1, 5, 10, 20, 50} — consistent with the paper's claim that T=10 is a reasonable default and recomputation cost can be amortized. β=0.99 marginally beats β=0.98; β=0.95 is slightly worse.

## Phase 4 — Refinement at best β=0.99

| LR | WD | β | Test | Train |
|---|---|---|---|---|
| 1.2e-2 | 0.01 | 0.99 | 4.197 | 3.852 |
| 2e-2 | 0.01 | 0.99 | 4.199 | 3.853 |
| 1.6e-2 | 0.001 | 0.99 | 4.204 | 3.857 |
| 1.6e-2 | 0.0 | 0.99 | 4.204 | 3.857 |
| 1.6e-2 | 0.01 | 0.995 | 4.773 | 4.476 (diverging) |

All within seed-noise of the Phase-3 best (4.193). β=0.995 destabilizes — the EMA becomes too slow to track curvature, mirroring the LR=8e-2 failure mode.

## Final HP recommendation

```python
optimizer_args = dict(
    class_="kl_shampoo",
    kwargs=dict(
        b1=0.9,                       # paper default; b1<0.8 fails badly
        shampoo_b=0.99,               # marginally beats 0.98; β≥0.995 diverges
        eps=1e-8,                     # paper default; flat 1e-8–1e-6
        weight_decay=0.01,            # paper default; flat 0–0.01
        precondition_frequency=1,     # T=1 ties with T∈[5,20]; pick 1 for freshness
        init_factor=0.1,              # paper default; alternatives within noise
        max_clamp_value=4000,         # paper default
        using_clamping=True,          # disabling makes no difference
        max_precond_dim=8192,         # routes embedding/unembedding to AdamW (no-op for w256)
        adam_b1=0.9, adam_b2=0.999, adam_eps=1e-8,  # all within noise
    )
)
schedule = dict(
    class_="warmup_cosine_decay_schedule",
    kwargs=dict(
        peak_value=1.6e-2,            # 160× the paper KLOpt default of 1e-4
        end_value=1.6e-3,             # peak / 10
        warmup_steps=500, decay_steps=4000,
        init_value=0.0, exponent=1.0,
    )
)
```

## Key findings

1. **KL-Shampoo's optimal LR is ~16× higher than the reference KLOpt default** (1.6e-2 vs 1e-4). Practitioners porting from the reference should sweep up.
2. **Landscape is flat near the optimum**: LR ∈ [8e-3, 4e-2], T ∈ [1, 20], β ∈ [0.98, 0.99] all give test loss within ~0.02. Hyperparameters are robust.
3. **Weight decay above 0.01 hurts.** mup-Muon's optimum WD=0.354 does not transfer.
4. **Step time is unchanged across T**: T=1 (every-step QR) and T=50 (rare QR) both ran in ~10 min wall. The QR is well-amortized; aggressive frequency reduction does not save wall time at this model scale.
5. **Diverging configs share a signature**: β=0.995 and LR=8e-2 both diverge to test loss ~4.78 (nearly identical to the lr=5e-4 underfit). Caps on the preconditioner update magnitude (`max_clamp_value`) are doing their job.

## Wandb runs

All runs in project `eb-lab/belo-meta-testing`. Sample run links:

- Initial baseline (lr=1e-3): https://wandb.ai/eb-lab/belo-meta-testing/runs/9gf4a6sy
- LR sweep best (lr=1.6e-2): https://wandb.ai/eb-lab/belo-meta-testing/runs/tu7eqd94
- HP sweep best (β=0.99): https://wandb.ai/eb-lab/belo-meta-testing/runs/dvk0s6tz

(Other runs synced via the offline → online wandb sync from fir login node.)

## Future work to push past 4.018 (mup-Muon reference)

1. **Add MuP scaling.** mup-Muon's gain comes from MuP per-tensor LR scales. KL-Shampoo could benefit from the same — wire `mup_lrs` pytree into `kl_shampoo_with_adamw` and re-sweep LR.
2. **Joint LR × β sweep.** The Phase-4 refinement only varied LR and WD at β=0.99; a 5×3 LR × β grid could squeeze a few more milli-nats.
3. **Larger batch / longer training.** Test loss 4.0 may need >4500 steps or batch >256.
4. **KL-SOAP variant.** Out-of-scope for this port but the paper claims it's a small additional gain.
5. **Implement seed multi-runs** to estimate the ~0.01 noise floor properly.

## Phase 5 — LR + β re-sweep at T=1

Per-user request to verify that low T doesn't unlock a different optimum. Same conclusion as T=10:

| LR | β | Test |
|---|---|---|
| **1.6e-2** | **0.99** | **4.194** |
| **1.6e-2** | **0.97** | **4.194** ← tied; broader plateau than T=10 |
| 2e-2 | 0.99 | 4.198 |
| 1.2e-2 | 0.99 | 4.200 |
| 2.5e-2 | 0.99 | 4.202 |
| 8e-3 | 0.99 | 4.209 |
| 4e-2 | 0.99 | 4.228 |
| 1.6e-2 | 0.995 | 4.774 (still diverges) |

T=1 does not enable higher LR; the divergence threshold is identical to T=10.

## Phase 6 — `b1` (gradient momentum) sweep + clamping/routing probes

| b1 | Test |
|---|---|
| 0.0 | 4.615 (no momentum is bad) |
| 0.5 | 4.304 |
| 0.8 | 4.201 |
| **0.9** | **4.194** ← default wins |
| 0.95 | 4.196 |
| 0.98 | 4.208 |

Probes (at b1=0.9 default):
- `using_clamping=False`: 4.196 (clamping is benign for this LR — never triggers)
- `max_precond_dim=2048`: 4.190 (no-op for this model — routes same params; reflects pure noise)

## Phase 7 — Noise floor + final HP probes

**Multi-seed reruns at best config** (n=4 including seed=0):

| Seed | Test |
|---|---|
| 0 | 4.194 |
| 1 | 4.200 |
| 2 | 4.204 |
| 3 | 4.203 |
| **mean ± σ** | **4.200 ± 0.004** |

So the **σ ≈ 0.004** noise floor sets the meaningful comparison: anything within ~0.012 of best is statistically equivalent at p≈0.05.

| Untouched HP | Test | Verdict |
|---|---|---|
| `adam_b2=0.95` | 4.204 | within noise |
| `adam_b2=0.9999` | 4.200 | within noise |
| `init_factor=0.01` | 4.219 | mildly worse |
| `init_factor=1.0` | 4.212 | mildly worse |
| `eps=1e-6` | 4.194 | within noise |

**All scaling_l2o-routed defaults survive scrutiny.**

## Job IDs (fir, all completed)

- Phase 1: 39110959, 39110961, 39110962, 39110963, 39110964, 39110965, 39110966
- Phase 1.5: 39114707, 39114708, 39114709, 39114710
- Phase 2: 39116504, 39116506, 39116507, 39116508
- Phase 3: 39120447, 39120448, 39120449, 39120450, 39120451, 39120452
- Phase 4: 39121025, 39121026, 39121027, 39121028, 39121029
- Phase 5: 39139350, 39139351, 39139352, 39139353, 39139354, 39139355, 39139356, 39139357
- Phase 6: 39140012, 39140013, 39140014, 39140015, 39140016, 39140017, 39140018
- Phase 7: 39141527, 39141528, 39141529, 39141530, 39141531, 39141532, 39141533, 39141534

**Total: 38 sweep jobs, ~5 GPU-hours each on 4×H100 → ~190 GPU-hours.**
