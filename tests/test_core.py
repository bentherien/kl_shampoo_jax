"""Core unit tests for the KL-Shampoo JAX implementation.

These tests do not depend on PyTorch. They check structural invariants:
- state shapes and dtypes after init
- finiteness of the update over a 50-step synthetic stream
- first-step returns a zero update (matches reference's `continue`)
- learning rate of 0 leaves params unchanged
- Q is unchanged on steps that are not multiples of `precondition_frequency`
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from kl_shampoo_jax import kl_shampoo, kl_shampoo_with_adamw
from kl_shampoo_jax._core import KLShampooState


def _make_optimizer(lr=1e-3, **kw):
    """Build the full chain (positive precond → +decayed_weights → −lr)."""
    kw.setdefault("cast_dtype", jnp.float32)  # use fp32 for unit tests
    kw.setdefault("eigh_dtype", jnp.float32)
    return optax.chain(
        kl_shampoo(**kw),
        optax.add_decayed_weights(0.0),
        optax.scale_by_learning_rate(lr),
    )


def test_init_state_2d_shapes():
    params = {"W": jnp.zeros((16, 32))}
    opt = _make_optimizer()
    state = opt.init(params)
    inner = state[0].inner["W"]
    assert inner.exp_avg.shape == (16, 32)
    assert inner.GG[0].shape == (16, 16)
    assert inner.GG[1].shape == (32, 32)
    assert inner.Q[0].shape == (16, 16)
    assert inner.Q[1].shape == (32, 32)
    assert inner.eigen_sqrt_inv[0].shape == (16,)
    assert inner.eigen_sqrt_inv[1].shape == (32,)
    assert inner.initialized.item() is False


def test_init_state_3d_shapes():
    params = {"Q3": jnp.zeros((4, 16, 8))}
    opt = _make_optimizer()
    state = opt.init(params)
    inner = state[0].inner["Q3"]
    assert inner.GG[0].shape == (4, 4)
    assert inner.GG[1].shape == (16, 16)
    assert inner.GG[2].shape == (8, 8)


def test_first_step_zero_update_2d():
    params = {"W": jnp.ones((8, 16), dtype=jnp.float32)}
    opt = _make_optimizer(lr=1.0)
    state = opt.init(params)
    grad = {"W": jnp.ones((8, 16), dtype=jnp.float32)}
    updates, _ = opt.update(grad, state, params)
    # First step: kl_shampoo returns 0; chain adds 0*params (wd=0); scales by -1.
    # Net update should be exactly zero.
    np.testing.assert_array_equal(np.asarray(updates["W"]), np.zeros((8, 16)))


def test_first_step_zero_update_3d():
    params = {"Q3": jnp.ones((4, 8, 6), dtype=jnp.float32)}
    opt = _make_optimizer(lr=1.0)
    state = opt.init(params)
    grad = {"Q3": jnp.ones((4, 8, 6), dtype=jnp.float32)}
    updates, _ = opt.update(grad, state, params)
    np.testing.assert_array_equal(np.asarray(updates["Q3"]), np.zeros((4, 8, 6)))


def test_50_steps_finite_2d():
    rng = np.random.default_rng(0)
    p = jnp.asarray(rng.standard_normal((32, 64)).astype(np.float32))
    params = {"W": p}
    opt = _make_optimizer(lr=1e-3)
    state = opt.init(params)

    @jax.jit
    def step(params, state, key):
        g = jax.random.normal(key, params["W"].shape, dtype=jnp.float32)
        updates, state = opt.update({"W": g}, state, params)
        params = optax.apply_updates(params, updates)
        return params, state

    key = jax.random.PRNGKey(0)
    for i in range(50):
        key, subkey = jax.random.split(key)
        params, state = step(params, state, subkey)
        assert jnp.all(jnp.isfinite(params["W"])), f"NaN at step {i}"


def test_50_steps_finite_3d():
    rng = np.random.default_rng(1)
    p = jnp.asarray(rng.standard_normal((4, 16, 8)).astype(np.float32))
    params = {"Q3": p}
    opt = _make_optimizer(lr=1e-3)
    state = opt.init(params)

    @jax.jit
    def step(params, state, key):
        g = jax.random.normal(key, params["Q3"].shape, dtype=jnp.float32)
        updates, state = opt.update({"Q3": g}, state, params)
        params = optax.apply_updates(params, updates)
        return params, state

    key = jax.random.PRNGKey(0)
    for i in range(50):
        key, subkey = jax.random.split(key)
        params, state = step(params, state, subkey)
        assert jnp.all(jnp.isfinite(params["Q3"])), f"NaN at step {i}"


def test_lr_zero_invariance():
    rng = np.random.default_rng(0)
    p = jnp.asarray(rng.standard_normal((16, 32)).astype(np.float32))
    params = {"W": p}
    opt = _make_optimizer(lr=0.0)
    state = opt.init(params)
    for i in range(20):
        g = {"W": jnp.asarray(rng.standard_normal((16, 32)).astype(np.float32))}
        updates, state = opt.update(g, state, params)
        params = optax.apply_updates(params, updates)
    np.testing.assert_allclose(np.asarray(params["W"]), np.asarray(p), atol=0)


def test_q_unchanged_between_recompute_boundaries():
    """At T=10, Q is set on call 1 (init) and stays fixed across calls 2..10
    (step counters 1..9, none divisible by 10). Q is recomputed on call 11
    (step counter becomes 10), matching the reference's `step % T == 0` check
    after the increment."""
    rng = np.random.default_rng(0)
    params = {"W": jnp.asarray(rng.standard_normal((16, 32)).astype(np.float32))}
    opt = _make_optimizer(lr=1e-3, precondition_frequency=10)
    state = opt.init(params)

    def step(params, state, g):
        updates, state = opt.update({"W": g}, state, params)
        params = optax.apply_updates(params, updates)
        return params, state

    Qs = []
    for i in range(15):
        g = jnp.asarray(rng.standard_normal((16, 32)).astype(np.float32))
        params, state = step(params, state, g)
        Qs.append(np.asarray(state[0].inner["W"].Q[0]).copy())

    # Calls 2..10 (Qs indices 1..9) should leave Q untouched (no recompute).
    for i in range(1, 10):
        np.testing.assert_array_equal(Qs[i], Qs[0], err_msg=f"Q changed at call {i+1}")
    # Call 11 (Qs[10]) should recompute (step counter becomes 10).
    assert not np.allclose(Qs[10], Qs[0]), "Q did not change at expected recompute boundary"
    # Calls 12..15 (Qs[11..14]) should hold the recomputed Q until the next boundary.
    for i in range(11, 15):
        np.testing.assert_array_equal(Qs[i], Qs[10], err_msg=f"Q changed mid-window at call {i+1}")


def test_with_adamw_2d_plus_1d():
    """Mixed pytree: 2D goes to KL, 1D goes to AdamW; both step."""
    rng = np.random.default_rng(0)
    params = {
        "W": jnp.asarray(rng.standard_normal((8, 16)).astype(np.float32)),
        "b": jnp.asarray(rng.standard_normal((16,)).astype(np.float32)),
    }
    opt = kl_shampoo_with_adamw(
        learning_rate=1e-3,
        kl_kwargs=dict(cast_dtype=jnp.float32, eigh_dtype=jnp.float32),
        adamw_kwargs=dict(b1=0.9, b2=0.999, eps=1e-8),
        weight_decay=0.0,
    )
    state = opt.init(params)
    g = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), params)
    updates, state = opt.update(g, state, params)
    # First step: KL leg returns 0 update; Adam leg returns standard direction.
    np.testing.assert_array_equal(np.asarray(updates["W"]), np.zeros((8, 16)))
    assert jnp.all(jnp.isfinite(updates["b"]))
    # Adam should have moved the bias (toward -lr * 1.0 / (sqrt(1.0)+eps))
    assert jnp.abs(updates["b"]).sum() > 0
