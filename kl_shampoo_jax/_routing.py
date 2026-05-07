"""Helpers for routing parameters to KL-Shampoo vs AdamW.

KL-Shampoo only handles 2D and 3D tensors with reasonable dimensions; 1D
biases / norms, scalars, and oversized embeddings need a different optimizer.
We use `optax.multi_transform` with a label tree to split.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import jax
import jax.numpy as jnp
import optax

from kl_shampoo_jax._core import kl_shampoo


def _label_one(p, max_precond_dim: int) -> str:
    if not hasattr(p, "ndim"):
        return "adam"
    if p.ndim not in (2, 3):
        return "adam"
    if max(p.shape) > max_precond_dim:
        return "adam"
    if min(p.shape) < 2:
        return "adam"
    return "kl"


def param_labels(max_precond_dim: int = 8192):
    """Return a label_fn for `optax.multi_transform`.

    Each leaf is labeled "kl" if it is a 2D/3D tensor with all dims in
    [2, max_precond_dim], else "adam".
    """
    def label_fn(params):
        return jax.tree_util.tree_map(
            lambda p: _label_one(p, max_precond_dim),
            params,
        )
    return label_fn


def kl_shampoo_with_adamw(
    learning_rate,
    *,
    kl_kwargs: Optional[Mapping[str, Any]] = None,
    adamw_kwargs: Optional[Mapping[str, Any]] = None,
    weight_decay: float = 0.01,
    max_precond_dim: int = 8192,
) -> optax.GradientTransformation:
    """KL-Shampoo on 2D/3D in-range matrices, AdamW on the rest.

    Both legs share the same `learning_rate` schedule. Decoupled weight decay
    is applied once at the chain level so it covers both groups uniformly.

    The KL leg returns the *positive* preconditioned grad and the AdamW leg
    returns the standard Adam direction; the outer chain applies
    `add_decayed_weights(weight_decay)` and `scale_by_learning_rate(lr)`,
    which inserts the minus sign and the schedule.

    Args:
      learning_rate: scalar or `optax.Schedule`. Shared across both legs.
      kl_kwargs: kwargs forwarded to `kl_shampoo` (b1, shampoo_b, eps, ...).
      adamw_kwargs: kwargs forwarded to `optax.scale_by_adam` (b1, b2, eps, ...).
      weight_decay: decoupled-WD coefficient applied to all params.
      max_precond_dim: KL leg routes only tensors with `max(shape) <= this`.
    """
    kl_kwargs = dict(kl_kwargs or {})
    adamw_kwargs = dict(adamw_kwargs or {})
    # kl_shampoo accepts a learning_rate kwarg for parity with optax.adamw, but
    # it is unused there — the schedule is applied at the chain level.
    kl_kwargs.pop("learning_rate", None)
    adamw_kwargs.pop("learning_rate", None)
    adamw_kwargs.pop("weight_decay", None)  # WD applied at chain level

    kl_leg = kl_shampoo(**kl_kwargs)
    adam_leg = optax.scale_by_adam(**adamw_kwargs)

    inner = optax.multi_transform(
        {"kl": kl_leg, "adam": adam_leg},
        param_labels(max_precond_dim=max_precond_dim),
    )

    return optax.chain(
        inner,
        optax.add_decayed_weights(weight_decay),
        optax.scale_by_learning_rate(learning_rate),
    )
