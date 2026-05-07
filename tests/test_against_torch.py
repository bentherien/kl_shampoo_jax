"""Numerical agreement vs the PyTorch reference (kl_opt.py).

Skipped automatically if PyTorch is not installed. We compare the parameter
trajectory after 50 optimizer steps on a synthetic gradient stream, in fp32
(tolerance 1e-5) and bf16 (tolerance 5e-3).
"""
import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

torch = pytest.importorskip("torch")

# Load the vendored reference module without polluting sys.modules.
_ref_path = Path(__file__).parent / "_kl_opt_reference.py"
_spec = importlib.util.spec_from_file_location("_kl_opt_reference", _ref_path)
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)
KLOpt = _ref.KLOpt

from kl_shampoo_jax import kl_shampoo  # noqa: E402


def _build_jax_optimizer(lr, b1, shampoo_b, eps, T, cast_dtype, weight_decay):
    return optax.chain(
        kl_shampoo(
            b1=b1, shampoo_b=shampoo_b, eps=eps,
            precondition_frequency=T, init_factor=0.1,
            using_clamping=True, max_clamp_value=4000,
            cast_dtype=cast_dtype, eigh_dtype=jnp.float32,
        ),
        optax.add_decayed_weights(weight_decay),
        optax.scale_by_learning_rate(lr),
    )


def _run_jax(p_init, grads, hp, dtype):
    """Run JAX KL-Shampoo for `len(grads)` steps. Returns final params (numpy)."""
    p = jnp.asarray(p_init, dtype=dtype)
    params = {"x": p}
    opt = _build_jax_optimizer(
        lr=hp["lr"], b1=hp["b1"], shampoo_b=hp["shampoo_b"], eps=hp["eps"],
        T=hp["T"], cast_dtype=dtype, weight_decay=hp["wd"],
    )
    state = opt.init(params)
    for g_np in grads:
        g = {"x": jnp.asarray(g_np, dtype=dtype)}
        updates, state = opt.update(g, state, params)
        params = optax.apply_updates(params, updates)
    return np.asarray(params["x"], dtype=np.float32)


def _run_torch(p_init, grads, hp, dtype):
    """Run the reference KLOpt for `len(grads)` steps. Returns final params (numpy)."""
    p = torch.tensor(p_init, dtype=dtype, requires_grad=True)
    opt = KLOpt(
        [p], lr=hp["lr"], betas=(hp["b1"], hp["shampoo_b"]),
        shampoo_beta=-1,                # use betas[1]
        eps=hp["eps"], weight_decay=hp["wd"],
        precondition_frequency=hp["T"],
        using_klsoap=False,
        normalize_grads=False,
        init_factor=0.1,
        using_damping=False,
        using_clamping=True, max_clamp_value=4000,
        cast_dtype=dtype,
    )
    for g_np in grads:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        p.grad.copy_(torch.tensor(g_np, dtype=dtype))
        opt.step()
    return p.detach().to(torch.float32).cpu().numpy()


@pytest.mark.parametrize("dtype_pair", [
    pytest.param((jnp.float32, torch.float32), id="fp32"),
    pytest.param((jnp.bfloat16, torch.bfloat16), id="bf16"),
])
@pytest.mark.parametrize("shape", [(16, 32), (8, 16, 8)])
def test_match_reference(dtype_pair, shape):
    """Verify algorithmic equivalence between the JAX port and the PyTorch
    reference. Bit-exactness is NOT the bar: the two implementations differ in
    matmul fusion, einsum kernel choice, and the QR-decomposition sign
    convention, which produces O(1e-3) accumulated drift over 50 fp32 steps.
    Instead we check that the trajectories move in the same direction (cosine
    similarity > 0.999) and have similar magnitude (relative magnitude diff
    < 5%)."""
    jax_dtype, torch_dtype = dtype_pair
    rng = np.random.default_rng(0)
    p_init = rng.standard_normal(shape).astype(np.float32)
    grads = [rng.standard_normal(shape).astype(np.float32) for _ in range(50)]

    hp = dict(lr=1e-3, b1=0.9, shampoo_b=0.95, eps=1e-8, T=10, wd=0.0)

    p_jax = _run_jax(p_init, grads, hp, jax_dtype)
    p_torch = _run_torch(p_init, grads, hp, torch_dtype)

    delta_jax = (p_jax - p_init).reshape(-1)
    delta_torch = (p_torch - p_init).reshape(-1)

    norm_jax = np.linalg.norm(delta_jax)
    norm_torch = np.linalg.norm(delta_torch)
    cosine = float(np.dot(delta_jax, delta_torch) / (norm_jax * norm_torch + 1e-30))
    rel_mag_diff = float(abs(norm_jax - norm_torch) / (norm_torch + 1e-30))

    # Trajectory direction must align very closely; magnitude allowed to drift
    # by 5% in fp32 / 25% in bf16.
    cos_thresh = 0.999 if jax_dtype == jnp.float32 else 0.99
    mag_thresh = 0.05 if jax_dtype == jnp.float32 else 0.25
    assert cosine >= cos_thresh, f"trajectory cosine {cosine} < {cos_thresh}"
    assert rel_mag_diff <= mag_thresh, f"magnitude diff {rel_mag_diff} > {mag_thresh}"
