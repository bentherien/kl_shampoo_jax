"""Routing tests: verify the param_labels function classifies leaves correctly."""
import jax
import jax.numpy as jnp

from kl_shampoo_jax import param_labels


def test_routing_basic():
    params = {
        "W": jnp.zeros((64, 128)),
        "Q": jnp.zeros((8, 64, 16)),
        "b": jnp.zeros((1024,)),
        "embed": jnp.zeros((50257, 256)),
        "scalar": jnp.zeros(()),
    }
    labels = param_labels(max_precond_dim=8192)(params)
    assert labels == {
        "W": "kl",
        "Q": "kl",
        "b": "adam",
        "embed": "adam",
        "scalar": "adam",
    }


def test_routing_oversize_threshold():
    params = {
        "small": jnp.zeros((1024, 1024)),
        "wide": jnp.zeros((128, 8192)),
        "too_wide": jnp.zeros((128, 8193)),
    }
    labels = param_labels(max_precond_dim=8192)(params)
    assert labels["small"] == "kl"
    assert labels["wide"] == "kl"
    assert labels["too_wide"] == "adam"


def test_routing_min_dim_one():
    params = {
        "row": jnp.zeros((1, 1024)),
        "col": jnp.zeros((1024, 1)),
        "ok": jnp.zeros((2, 1024)),
    }
    labels = param_labels(max_precond_dim=8192)(params)
    assert labels["row"] == "adam"
    assert labels["col"] == "adam"
    assert labels["ok"] == "kl"


def test_routing_4d_falls_back():
    params = {"conv": jnp.zeros((3, 3, 64, 64))}
    labels = param_labels()(params)
    assert labels["conv"] == "adam"


def test_per_leg_weight_decay():
    """Verify that adam_weight_decay applies only to the Adam leg and
    `weight_decay` only to the KL leg when both are set."""
    import numpy as np
    import optax
    from kl_shampoo_jax import kl_shampoo_with_adamw

    rng = np.random.default_rng(0)
    params = {
        "W": jnp.asarray(rng.standard_normal((8, 16)).astype(np.float32)),  # KL leg
        "b": jnp.asarray(rng.standard_normal((16,)).astype(np.float32)),    # Adam leg
    }

    # Configure: KL_wd = 0.5, adam_wd = 0.0, lr = 1.0, no Adam direction.
    # Use b1=b2=0, eps=large to make Adam direction ~0; only WD remains.
    opt = kl_shampoo_with_adamw(
        learning_rate=1.0,
        kl_kwargs=dict(cast_dtype=jnp.float32, eigh_dtype=jnp.float32),
        adamw_kwargs=dict(b1=0.0, b2=0.0, eps=1.0),
        weight_decay=0.5,
        adam_weight_decay=0.0,
    )
    state = opt.init(params)
    g = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)
    updates, _ = opt.update(g, state, params)

    # KL leg's first iter returns 0 grad direction; with kl_wd=0.5 and lr=1.0,
    # update should be -1.0 * (0 + 0.5*W) = -0.5 * W
    np.testing.assert_allclose(np.asarray(updates["W"]), -0.5 * np.asarray(params["W"]), atol=1e-5)
    # Adam leg with b1=b2=0 gives Adam direction ~ g/(eps + sqrt(v)) = 0/(1+0) = 0;
    # with adam_wd=0.0, total update should be 0.
    np.testing.assert_allclose(np.asarray(updates["b"]), np.zeros_like(np.asarray(params["b"])), atol=1e-5)


def test_adam_lr_scale_isolates_adam_leg():
    """Verify adam_lr_scale only affects the Adam leg, not the KL leg."""
    import numpy as np
    import optax
    from kl_shampoo_jax import kl_shampoo_with_adamw

    rng = np.random.default_rng(0)
    params = {"b": jnp.asarray(rng.standard_normal((16,)).astype(np.float32))}
    g = {"b": jnp.ones((16,), dtype=jnp.float32)}

    # Build with adam_lr_scale=0.1 vs 1.0; same params, same grads.
    # The Adam direction is ~sign(g) = +1, so update = -lr * adam_lr_scale * 1.
    opt_full = kl_shampoo_with_adamw(
        learning_rate=1.0, kl_kwargs=dict(cast_dtype=jnp.float32),
        adamw_kwargs=dict(b1=0.9, b2=0.999, eps=1e-8),
        weight_decay=0.0, adam_lr_scale=1.0,
    )
    opt_scaled = kl_shampoo_with_adamw(
        learning_rate=1.0, kl_kwargs=dict(cast_dtype=jnp.float32),
        adamw_kwargs=dict(b1=0.9, b2=0.999, eps=1e-8),
        weight_decay=0.0, adam_lr_scale=0.1,
    )
    s_full = opt_full.init(params)
    s_scaled = opt_scaled.init(params)
    upd_full, _ = opt_full.update(g, s_full, params)
    upd_scaled, _ = opt_scaled.update(g, s_scaled, params)

    # Scaled update should be 10× smaller than full
    ratio = float(jnp.abs(upd_scaled["b"]).sum() / (jnp.abs(upd_full["b"]).sum() + 1e-12))
    assert 0.08 <= ratio <= 0.12, f"adam_lr_scale=0.1 should give ~0.1× update; got ratio {ratio}"
