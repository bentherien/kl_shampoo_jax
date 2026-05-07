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
