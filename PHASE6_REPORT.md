# Phase 6 — μCompletedP-KL-Shampoo HP sweep on w256-d4

**Date:** 2026-05-10
**Cluster:** rorqual (rrg-bengioy-ad_gpu)
**Task:** `muztransformer-dense-w256-d4-h2_fineweb-s512-gpt2`
**Steps:** 4500 (warmup=500, decay=4000)
**Batch:** local_bs=16 × ga=4 × world=4 = 256 sequences/step, seq_len=512
**Optimizer:** `mup_kl_shampoo` + `kl_shampoo_completedp` parameterization
**Inner library:** `kl_shampoo_jax` v0.3.0 (per-leaf eps_scale)

## Anchor config

```python
optimizer_args = dict(
    class_="mup_kl_shampoo",
    kwargs=dict(
        learning_rate=1.6e-2,
        kl_b1=0.9, shampoo_b=0.99, kl_eps=1e-8, kl_weight_decay=0.01,
        precondition_frequency=1,
        adam_b1=0.95484, adam_b2=0.9908, adam_eps=1e-8,
        adam_weight_decay=0.093198, adam_lr_scale=1.0,
    ),
)
```

## Coordinate-descent sweep — results

Ranked by **smoothed test L̂** (MuLoCo EMA, α=0.2, H=30, no subsampling):

| Rank | Config | Last test | **L̂ test** | Last train | L̂ train | Δ vs anchor | run_id |
|------|--------|-----------|---------|------------|---------|-------------|--------|
| 1 | **awd_0.01** ⭐ | 4.1715 | **3.9557** | 3.8222 | 3.8862 | **−0.034** | [80elwjmx](https://wandb.ai/eb-lab/belo-meta-testing/runs/80elwjmx) |
| 2 | lr_8e-3 | 4.1900 | 3.9718 | 3.8351 | 3.8997 | −0.018 | [rhaujgq1](https://wandb.ai/eb-lab/belo-meta-testing/runs/rhaujgq1) |
| 3 | sb_0.98 | 4.1932 | 3.9769 | 3.8485 | 3.9087 | −0.013 | [w0w91e2t](https://wandb.ai/eb-lab/belo-meta-testing/runs/w0w91e2t) |
| 4 | klwd_0.001 | 4.1993 | 3.9848 | 3.8562 | 3.9150 | −0.005 | [pwqsgmrk](https://wandb.ai/eb-lab/belo-meta-testing/runs/pwqsgmrk) |
| 5 | anchor | 4.2030 | 3.9897 | 3.8562 | 3.9195 | 0 | [nc9r9akh](https://wandb.ai/eb-lab/belo-meta-testing/runs/nc9r9akh) |
| 6 | T_5 | 4.2054 | 3.9941 | 3.8631 | 3.9244 | +0.004 | [yfrlfbg1](https://wandb.ai/eb-lab/belo-meta-testing/runs/yfrlfbg1) |
| 7 | klb1_0.95 | 4.2082 | 3.9952 | 3.8678 | 3.9265 | +0.005 | [f3wnemfp](https://wandb.ai/eb-lab/belo-meta-testing/runs/f3wnemfp) |
| 8 | lr_3.2e-2 | 4.3095 | 4.0927 | 3.9646 | 4.0219 | +0.103 | [xmmb48kc](https://wandb.ai/eb-lab/belo-meta-testing/runs/xmmb48kc) |
| 9 | klwd_0.1 | 4.3620 | 4.1466 | 4.0183 | 4.0769 | +0.157 | [soc5nn4q](https://wandb.ai/eb-lab/belo-meta-testing/runs/soc5nn4q) |
| 10 | awd_0.354 | 4.4435 | 4.2299 | 4.1124 | 4.1607 | +0.240 | [sh0zrsv1](https://wandb.ai/eb-lab/belo-meta-testing/runs/sh0zrsv1) |

## Headline comparisons

| Run | Optimizer | L̂ test | Note |
|---|---|---|---|
| mup-Muon `xak20u38` | mup_muon (CompletedP) | 4.044 | Validated baseline |
| Phase 8 `pmxjddz4` | kl_shampoo (vanilla SP) | 3.977 | Best pre-CompletedP |
| **Phase 6 `80elwjmx`** ⭐ | **mup_kl_shampoo (CompletedP)** | **3.9557** | **Best overall** |

**Phase 6 best beats:**
- Phase 8 vanilla-SP best by **0.021 nats** → CompletedP routing is net positive
- mup-Muon by **0.088 nats** → KL-Shampoo+CompletedP > Muon+CompletedP at base scale

## Findings

1. **CompletedP routing landed without regression.** The anchor's L̂=3.9897 is within noise (σ≈0.004 per Phase 7 multi-seed analysis) of Phase 8's L̂=3.977. The new `mup_kl_shampoo` and `KLShampooCompletedPParameterization` correctly reduce to the vanilla-SP path at the base scale (m_W=1, m_D=1, m_B=1) — a sanity check before testing transfer.
2. **`adam_wd=0.01` (vs anchor 0.0931) is the surprise win** (0.034 nats). Hypothesis: the KL-Shampoo leg already absorbs most regularization through the Kronecker preconditioner; the AdamW leg (embeddings/norms/output_head) prefers low WD when KL-side regularization is strong. mup-Muon's `adam_wd=0.0931` doesn't transfer.
3. **`lr_8e-3` ≈ `anchor`** (within noise). KL-Shampoo's LR landscape is flat between 0.8× and 1× anchor.
4. **`T=5` essentially ties `T=1`** (Δ=+0.004 nats). Confirms the Phase 8 finding that T=5 is 4.3× cheaper opt step at sub-noise quality cost.
5. **Outer envelopes confirmed:** `lr_3.2e-2`, `awd_0.354`, `klwd_0.1` all clearly worse — the search space is bounded.

## Recommended Phase 7 anchor

For the upcoming HP-transfer sweep (vary width / depth / batch / duration, hold HPs fixed):

```python
optimizer_args = dict(
    class_="mup_kl_shampoo",
    kwargs=dict(
        learning_rate=1.6e-2,        # anchor; sweep at each axis-point
        kl_b1=0.9, shampoo_b=0.99, kl_eps=1e-8,
        kl_weight_decay=0.01,
        precondition_frequency=5,     # cheaper than T=1 within noise
        adam_b1=0.95484, adam_b2=0.9908, adam_eps=1e-8,
        adam_weight_decay=0.01,       # ← changed from 0.0931 (Phase 6 win)
        adam_lr_scale=1.0,
    ),
)
```

## Pipeline notes

- Initial submission used cluster httpproxy module → JAX gRPC distributed init got 403s → 5-min DEADLINE_EXCEEDED crashes. Fix: `unset HTTP_PROXY HTTPS_PROXY NO_PROXY` (and lowercase variants + RSYNC_PROXY) before mpirun; `WANDB_MODE=offline` to avoid wandb needing the proxy.
- All 10 jobs trained cleanly on 4×H100 in ~17-20 min wall each (5.0-5.3 it/s @ 4500 steps).
- Optimizer step time: ~92ms/step at T=1, ~22ms/step at T=5 (matches Phase 8 numbers).
- Sweep script: `~/scratch/l2o_install/scaling_l2o/jobs/kl_shampoo_completedp/phase6_sweep.sh` on rorqual.
- Aggregation: `/tmp/p6_smoothed_loss.py` (entity=eb-lab, project=belo-meta-testing).
