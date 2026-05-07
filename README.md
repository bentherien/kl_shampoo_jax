# kl_shampoo_jax

A JAX/optax port of **KL-Shampoo** (Lin, Lowe, Dangel, Eschenhagen, Xu & Grosse,
*Understanding and Improving Shampoo and SOAP via Kullback-Leibler
Minimization*, [arXiv:2509.03378](https://arxiv.org/abs/2509.03378), ICLR 2026).

KL-Shampoo recasts the Shampoo Kronecker-factor estimation as a covariance
estimation under KL divergence. Compared to Shampoo and SOAP, it removes the
need for an Adam co-optimizer on the preconditionable tensors, eliminating the
associated memory overhead while matching SOAP's per-iteration runtime.

This package implements the **KL-Shampoo** branch only (KL-SOAP is out of
scope). It is a faithful port of `optim/kl_opt.py` from
[yorkerlin/KL-Methods](https://github.com/yorkerlin/KL-Methods) with
`using_klsoap=False`.

## Install

```bash
pip install git+https://github.com/bentherien/kl_shampoo_jax.git
```

## Quickstart

```python
import jax, jax.numpy as jnp, optax
from kl_shampoo_jax import kl_shampoo_with_adamw

# Toy 2-layer MLP
params = {
    "W1": jnp.zeros((128, 256)),   # → KL-Shampoo
    "b1": jnp.zeros((256,)),       # → AdamW (1D)
    "W2": jnp.zeros((256, 10)),    # → KL-Shampoo
    "b2": jnp.zeros((10,)),        # → AdamW
}

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3, warmup_steps=500,
    decay_steps=4500, end_value=1e-4,
)
optimizer = kl_shampoo_with_adamw(
    learning_rate=schedule,
    kl_kwargs=dict(b1=0.9, shampoo_b=0.98, eps=1e-8, precondition_frequency=10),
    adamw_kwargs=dict(b1=0.9, b2=0.999, eps=1e-8),
    weight_decay=0.01,
    max_precond_dim=8192,        # 2D tensors with max(shape) > this go to AdamW
)
opt_state = optimizer.init(params)

@jax.jit
def step(params, opt_state, grads):
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state
```

## API

```python
kl_shampoo(
    learning_rate=1.0,           # accepted for parity; unused (compose externally)
    b1=0.9,                      # momentum on grad
    shampoo_b=0.98,              # preconditioner EMA β₂
    eps=1e-8,                    # damping in inv-sqrt division
    precondition_frequency=10,   # T: Q-recompute period
    init_factor=0.1,             # initial λ_k constant (1/√init_factor for esi)
    max_clamp_value=4000,        # cap on |eigen_sqrt_inv|
    using_clamping=True,
    cast_dtype=jnp.bfloat16,     # storage / matmul
    eigh_dtype=jnp.float32,      # eigh / qr always run in this dtype
) -> optax.GradientTransformation
```

The transform returns the **positive** preconditioned gradient. Wrap with
`optax.add_decayed_weights(wd)` and `optax.scale_by_learning_rate(lr)` to get
the full update. Or use `kl_shampoo_with_adamw`, which composes the chain for
you and routes 1D / oversize tensors to AdamW.

```python
kl_shampoo_with_adamw(
    learning_rate,                  # scalar or optax.Schedule
    *,
    kl_kwargs=None,                 # forwarded to kl_shampoo
    adamw_kwargs=None,              # forwarded to optax.scale_by_adam
    weight_decay=0.01,              # decoupled WD applied to all params
    max_precond_dim=8192,           # KL leg threshold
) -> optax.GradientTransformation
```

## Hyperparameter defaults

| | reference (`KLOpt`) | this port (`kl_shampoo`) |
|---|---|---|
| `lr` | 1e-4 | (chain externally) |
| `betas[0]` (β₁ for grad EMA) | 0.9 | `b1=0.9` |
| `betas[1]` (β₂ for precond EMA) | 0.98 | `shampoo_b=0.98` |
| `eps` (damping) | 1e-8 | `eps=1e-8` |
| `weight_decay` | 0.01 | (chain externally) |
| `precondition_frequency` | 10 | 10 |
| `init_factor` | 0.1 | 0.1 |
| `using_clamping` | `True` | `True` |
| `max_clamp_value` | 4000 | 4000 |
| `cast_dtype` | `torch.bfloat16` | `jnp.bfloat16` |

## Limitations

- **KL-Shampoo only.** The KL-SOAP branch (`using_klsoap=True` in the reference)
  is not implemented.
- **2D and 3D tensors only.** 1D and 4D+ leaves must be routed to a different
  optimizer; `kl_shampoo_with_adamw` does this automatically.
- **No MuP scaling.** This package is plain SP. For MuP transfer, scale `lr`
  per-tensor in your training loop.
- **Single-host, no sharding.** State is replicated across DP ranks.

## Tests

```bash
pip install -e ".[test]"
pytest tests/ -v
```

The test suite includes:
- 2D and 3D synthetic streams: shape, finiteness, lr=0 invariance, first-step
  no-update.
- Numerical agreement vs the PyTorch reference (`kl_opt.py`) within 1e-5 (fp32)
  and 5e-3 (bf16). Skipped automatically if `torch` is not installed.
- Routing correctness on a heterogeneous param tree.

## Reference

```bibtex
@misc{lin2025klshampoo,
  title         = {Understanding and Improving Shampoo and SOAP via Kullback-Leibler Minimization},
  author        = {Wu Lin and Scott C. Lowe and Felix Dangel and Runa Eschenhagen and Zikun Xu and Roger B. Grosse},
  year          = {2025},
  eprint        = {2509.03378},
  archivePrefix = {arXiv},
}
```

## License

Apache-2.0. The reference implementation it is ported from is at
[github.com/yorkerlin/KL-Methods](https://github.com/yorkerlin/KL-Methods).
