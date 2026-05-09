"""KL-Shampoo optimizer in JAX/optax.

JAX port of `KLOpt` from yorkerlin/KL-Methods (Lin et al., arXiv:2509.03378),
restricted to the KL-Shampoo branch (`using_klsoap=False`). Supports 2D and 3D
parameter tensors. Other ranks must be routed elsewhere (see _routing.py).

The factory returns a vanilla `optax.GradientTransformation`. Decoupled weight
decay and the learning-rate sign are *not* applied here; compose them via
`optax.chain(kl_shampoo(...), optax.add_decayed_weights(wd),
optax.scale_by_learning_rate(lr))`.
"""
from __future__ import annotations

from typing import Any, NamedTuple, Optional, Tuple

import chex
import flax.struct as fstruct
import jax
import jax.numpy as jnp
import optax


@fstruct.dataclass
class _PerParamState:
    step: jnp.ndarray                     # int32 scalar
    exp_avg: jnp.ndarray                  # cast_dtype, same shape as param
    GG: Tuple[jnp.ndarray, ...]           # length ndim; (d_k, d_k) each
    Q: Tuple[jnp.ndarray, ...]            # length ndim; (d_k, d_k) each
    eigen_sqrt_inv: Tuple[jnp.ndarray, ...]  # length ndim; (d_k,) each
    initialized: jnp.ndarray              # bool scalar


@fstruct.dataclass
class _StepOut:
    state: _PerParamState
    update: jnp.ndarray


class KLShampooState(NamedTuple):
    inner: chex.ArrayTree                 # tree of _PerParamState matching params


def _is_per_param_state(x):
    return isinstance(x, _PerParamState)


def _is_step_out(x):
    return isinstance(x, _StepOut)


def _is_masked(x):
    """True for `optax.MaskedNode` placeholders, regardless of optax version.

    `optax.multi_transform` masks non-matching leaves with a MaskedNode (an
    empty NamedTuple) before calling each inner transform's `update_fn`. We
    treat it as a leaf so we can propagate it correctly through tree_map.
    """
    return type(x).__name__ == "MaskedNode"


def _supported_rank(p) -> bool:
    return hasattr(p, "ndim") and p.ndim in (2, 3) and min(p.shape) >= 1


def _make_init_state(p, cast_dtype) -> _PerParamState:
    """Allocate zero state for a leaf. For unsupported ranks we still allocate a
    tiny dummy state so the pytree typechecks; callers should route 1D/oversize
    leaves through a different transform via optax.multi_transform."""
    if not _supported_rank(p):
        return _PerParamState(
            step=jnp.int32(0),
            exp_avg=jnp.zeros_like(p, dtype=cast_dtype),
            GG=(jnp.zeros((1, 1), dtype=cast_dtype),),
            Q=(jnp.eye(1, dtype=cast_dtype),),
            eigen_sqrt_inv=(jnp.zeros((1,), dtype=cast_dtype),),
            initialized=jnp.bool_(True),  # never run on unsupported leaves
        )
    GG = tuple(jnp.zeros((d, d), dtype=cast_dtype) for d in p.shape)
    Q = tuple(jnp.eye(d, dtype=cast_dtype) for d in p.shape)
    esi = tuple(jnp.zeros((d,), dtype=cast_dtype) for d in p.shape)
    return _PerParamState(
        step=jnp.int32(0),
        exp_avg=jnp.zeros_like(p, dtype=cast_dtype),
        GG=GG, Q=Q, eigen_sqrt_inv=esi,
        initialized=jnp.bool_(False),
    )


def _eigh_init(GG: jnp.ndarray, init_factor: float, cast_dtype, eigh_dtype) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """eigh of GG, return (Q, eigen_sqrt_inv) where esi is constant 1/sqrt(init_factor).

    Mirrors `get_orthogonal_matrix` in the reference: we use `init_factor` as the
    initial eigenvalue rather than the actual eigenvalues of GG (the eigenbasis
    Q is what carries the information from one batch).
    """
    GGf = GG.astype(eigh_dtype)
    GGf = GGf + 1e-30 * jnp.eye(GGf.shape[0], dtype=eigh_dtype)
    _, Q = jnp.linalg.eigh(GGf)
    Q = jnp.flip(Q, axis=1)                    # descending eigenvalues
    d = GG.shape[0]
    esi = jnp.full((d,), 1.0 / jnp.sqrt(init_factor), dtype=cast_dtype)
    return Q.astype(cast_dtype), esi


def _update_eigen_value(esi, diag, beta, using_clamping, max_clamp_value, cast_dtype, eigh_dtype):
    """EMA on D = 1/esi^2, return new esi = clamp(1/sqrt(D))."""
    inv_d = esi.astype(eigh_dtype) ** 2
    D = jnp.where(inv_d > 0, 1.0 / inv_d, 0.0)
    D = jnp.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
    D = (1.0 - beta) * diag.astype(eigh_dtype) + beta * D
    sqrt_inv = jnp.where(D > 0, 1.0 / jnp.sqrt(D), 0.0)
    sqrt_inv = jnp.nan_to_num(sqrt_inv, nan=0.0, posinf=0.0, neginf=0.0)
    if using_clamping:
        cap = max(10, min(D.shape[0], int(max_clamp_value)))
        sqrt_inv = jnp.clip(sqrt_inv, max=cap)
    return sqrt_inv.astype(cast_dtype)


def _recompute_q(GG, Q, cast_dtype, eigh_dtype):
    """Q ← qr(GG @ Q) via one round of power iteration + QR. fp32 internally."""
    powered = GG.astype(eigh_dtype) @ Q.astype(eigh_dtype)
    Qn, _ = jnp.linalg.qr(powered)
    return Qn.astype(cast_dtype)


def _step_2d(grad, st, hp):
    """Run one optimizer step for a 2D parameter, returning (new_state, update).

    Mirrors klshampoo_update + update_preconditioner from the reference
    (using_klsoap=False branch).
    """
    cast_dtype = hp["cast_dtype"]
    eigh_dtype = hp["eigh_dtype"]
    grad = grad.astype(cast_dtype)
    d0, d1 = grad.shape
    total_factor = grad.size
    beta = hp["shampoo_b"]

    def _first_iter(_):
        # init_preconditioner + update_preconditioner (first invocation only).
        # Compute initial GG via tensordot (axes contract everything except idx).
        mat0 = jnp.tensordot(grad, grad, axes=[[1], [1]])     # (d0, d0)
        mat1 = jnp.tensordot(grad, grad, axes=[[0], [0]])     # (d1, d1)
        GG0 = beta * st.GG[0] + (1.0 - beta) / (total_factor / d0) * mat0
        GG1 = beta * st.GG[1] + (1.0 - beta) / (total_factor / d1) * mat1
        Q0, esi0 = _eigh_init(GG0, hp["init_factor"], cast_dtype, eigh_dtype)
        Q1, esi1 = _eigh_init(GG1, hp["init_factor"], cast_dtype, eigh_dtype)
        new_st = _PerParamState(
            step=st.step,                 # don't increment on first iter (matches reference)
            exp_avg=st.exp_avg,
            GG=(GG0, GG1), Q=(Q0, Q1),
            eigen_sqrt_inv=(esi0, esi1),
            initialized=jnp.bool_(True),
        )
        return new_st, jnp.zeros_like(grad)

    def _regular(_):
        # 1) momentum on grad in original space
        new_exp = (1.0 - hp["b1"]) * grad + hp["b1"] * st.exp_avg

        # 2) project: new_exp → Q0^T new_exp Q1 via two tensordots along axis 0
        proj = jnp.tensordot(new_exp, st.Q[0], axes=[[0], [0]])   # (d1, d0)
        proj = jnp.tensordot(proj,    st.Q[1], axes=[[0], [0]])   # (d0, d1)

        # 3) Kronecker scaling by esi_0 ⊗ esi_1, with damping
        scale = st.eigen_sqrt_inv[0][:, None] * st.eigen_sqrt_inv[1][None, :]
        scale = scale / (1.0 + scale * jnp.asarray(hp["eps"], scale.dtype))
        precond = proj * scale

        # 4) project back: Q0 precond Q1^T
        upd = jnp.tensordot(precond, st.Q[0], axes=[[0], [1]])    # (d1, d0)
        upd = jnp.tensordot(upd,     st.Q[1], axes=[[0], [1]])    # (d0, d1)

        # 5) preconditioner update (mirrors update_2d_preconditioner)
        # axis 0 (left): step0 = Q1^T grad^T; lhalf = step0 * esi[1]; mat = lhalf^T lhalf
        step0 = st.Q[1].T @ grad.T                                # (d1, d0)
        lhalf = step0 * st.eigen_sqrt_inv[1][:, None]
        mat0 = lhalf.T @ lhalf                                    # (d0, d0)
        new_GG0 = beta * st.GG[0] + (1.0 - beta) / (total_factor / d0) * mat0

        # axis 1 (right): step1 = Q0^T grad; rhalf = step1 * esi[0]; mat = rhalf^T rhalf
        step1 = st.Q[0].T @ grad                                  # (d0, d1)
        rhalf = step1 * st.eigen_sqrt_inv[0][:, None]
        mat1 = rhalf.T @ rhalf                                    # (d1, d1)
        new_GG1 = beta * st.GG[1] + (1.0 - beta) / (total_factor / d1) * mat1

        # diag-EMA update of eigenvalues; reuse step1
        diag_half = step1 @ st.Q[1]                               # (d0, d1)
        ldiag = jnp.mean((diag_half * st.eigen_sqrt_inv[1][None, :]) ** 2, axis=1)  # (d0,)
        rdiag = jnp.mean((diag_half * st.eigen_sqrt_inv[0][:, None]) ** 2, axis=0)  # (d1,)
        new_esi0 = _update_eigen_value(
            st.eigen_sqrt_inv[0], ldiag, beta,
            hp["using_clamping"], hp["max_clamp_value"], cast_dtype, eigh_dtype,
        )
        new_esi1 = _update_eigen_value(
            st.eigen_sqrt_inv[1], rdiag, beta,
            hp["using_clamping"], hp["max_clamp_value"], cast_dtype, eigh_dtype,
        )

        # 6) Q recompute every T steps (after step increment, matches reference)
        new_step = st.step + jnp.int32(1)
        do_recompute = jnp.logical_and(new_step > 0, jnp.mod(new_step, hp["precondition_frequency"]) == 0)
        new_Q0 = jax.lax.cond(
            do_recompute,
            lambda _: _recompute_q(new_GG0, st.Q[0], cast_dtype, eigh_dtype),
            lambda _: st.Q[0],
            operand=None,
        )
        new_Q1 = jax.lax.cond(
            do_recompute,
            lambda _: _recompute_q(new_GG1, st.Q[1], cast_dtype, eigh_dtype),
            lambda _: st.Q[1],
            operand=None,
        )

        new_st = _PerParamState(
            step=new_step,
            exp_avg=new_exp,
            GG=(new_GG0, new_GG1), Q=(new_Q0, new_Q1),
            eigen_sqrt_inv=(new_esi0, new_esi1),
            initialized=jnp.bool_(True),
        )
        return new_st, upd

    return jax.lax.cond(st.initialized, _regular, _first_iter, operand=None)


def _step_3d(grad, st, hp):
    """Run one optimizer step for a 3D parameter. Mirrors update_3d_preconditioner."""
    cast_dtype = hp["cast_dtype"]
    eigh_dtype = hp["eigh_dtype"]
    grad = grad.astype(cast_dtype)
    d0, d1, d2 = grad.shape
    total_factor = grad.size
    beta = hp["shampoo_b"]

    def _first_iter(_):
        # tensordot trick: contract all axes except idx
        mat0 = jnp.tensordot(grad, grad, axes=[[1, 2], [1, 2]])   # (d0, d0)
        mat1 = jnp.tensordot(grad, grad, axes=[[0, 2], [0, 2]])   # (d1, d1)
        mat2 = jnp.tensordot(grad, grad, axes=[[0, 1], [0, 1]])   # (d2, d2)
        GG0 = beta * st.GG[0] + (1.0 - beta) / (total_factor / d0) * mat0
        GG1 = beta * st.GG[1] + (1.0 - beta) / (total_factor / d1) * mat1
        GG2 = beta * st.GG[2] + (1.0 - beta) / (total_factor / d2) * mat2
        Q0, esi0 = _eigh_init(GG0, hp["init_factor"], cast_dtype, eigh_dtype)
        Q1, esi1 = _eigh_init(GG1, hp["init_factor"], cast_dtype, eigh_dtype)
        Q2, esi2 = _eigh_init(GG2, hp["init_factor"], cast_dtype, eigh_dtype)
        new_st = _PerParamState(
            step=st.step,
            exp_avg=st.exp_avg,
            GG=(GG0, GG1, GG2), Q=(Q0, Q1, Q2),
            eigen_sqrt_inv=(esi0, esi1, esi2),
            initialized=jnp.bool_(True),
        )
        return new_st, jnp.zeros_like(grad)

    def _regular(_):
        # 1) momentum
        new_exp = (1.0 - hp["b1"]) * grad + hp["b1"] * st.exp_avg

        # 2) project: tensordot along axis 0 three times in succession
        proj = jnp.tensordot(new_exp, st.Q[0], axes=[[0], [0]])   # (d1, d2, d0)
        proj = jnp.tensordot(proj,    st.Q[1], axes=[[0], [0]])   # (d2, d0, d1)
        proj = jnp.tensordot(proj,    st.Q[2], axes=[[0], [0]])   # (d0, d1, d2)

        # 3) Kronecker scaling
        scale = (st.eigen_sqrt_inv[0][:, None, None]
                 * st.eigen_sqrt_inv[1][None, :, None]
                 * st.eigen_sqrt_inv[2][None, None, :])
        scale = scale / (1.0 + scale * jnp.asarray(hp["eps"], scale.dtype))
        precond = proj * scale

        # 4) project back
        upd = jnp.tensordot(precond, st.Q[0], axes=[[0], [1]])    # (d1, d2, d0)
        upd = jnp.tensordot(upd,     st.Q[1], axes=[[0], [1]])    # (d2, d0, d1)
        upd = jnp.tensordot(upd,     st.Q[2], axes=[[0], [1]])    # (d0, d1, d2)

        # 5) preconditioner update — direct port of update_3d_preconditioner einsums.
        invS_h0 = st.Q[0] * st.eigen_sqrt_inv[0][None, :]         # Q[0] · diag(esi[0])
        invS_h1 = st.Q[1] * st.eigen_sqrt_inv[1][None, :]
        invS_h2 = st.Q[2] * st.eigen_sqrt_inv[2][None, :]

        # GinvS1h = G *axis0* invS_h0 → indices 'ija,ip->pja'
        GinvS1h = jnp.einsum("ija,ip->pja", grad, invS_h0)
        GinvS1Q2 = jnp.einsum("pja,jl->pla", GinvS1h, st.Q[1])
        GinvS1Q2Q3 = jnp.einsum("pqa,am->pqm", GinvS1Q2, st.Q[2])

        # update GG[2] (S3 in the reference)
        GinvS12h = GinvS1Q2 * st.eigen_sqrt_inv[1][None, :, None]
        GinvS12G_T = jnp.tensordot(GinvS12h, GinvS12h, axes=[[0, 1], [0, 1]])
        new_GG2 = beta * st.GG[2] + (1.0 - beta) / (total_factor / d2) * GinvS12G_T

        # update GG[1] (S2) using OLD esi[2]
        GinvS1Q3 = jnp.einsum("pqb,bm->pqm", GinvS1h, st.Q[2])
        GinvS13h = GinvS1Q3 * st.eigen_sqrt_inv[2][None, None, :]
        GinvS13G_T = jnp.tensordot(GinvS13h, GinvS13h, axes=[[0, 2], [0, 2]])
        new_GG1 = beta * st.GG[1] + (1.0 - beta) / (total_factor / d1) * GinvS13G_T

        # Update esi[2] (S3 diag) using OLD esi[1]; reference's diag3.
        diag_for_esi2 = jnp.mean(
            (GinvS1Q2Q3 * st.eigen_sqrt_inv[1][None, :, None]) ** 2, axis=(0, 1)
        )
        new_esi2 = _update_eigen_value(
            st.eigen_sqrt_inv[2], diag_for_esi2, beta,
            hp["using_clamping"], hp["max_clamp_value"], cast_dtype, eigh_dtype,
        )

        # Update esi[1] (S2 diag) using NEW esi[2] — the reference's update_eigen_value
        # writes back into state inplace before this line, so esi[2] is already new.
        diag_for_esi1 = jnp.mean(
            (GinvS1Q2Q3 * new_esi2[None, None, :]) ** 2, axis=(0, 2)
        )
        new_esi1 = _update_eigen_value(
            st.eigen_sqrt_inv[1], diag_for_esi1, beta,
            hp["using_clamping"], hp["max_clamp_value"], cast_dtype, eigh_dtype,
        )

        # update GG[0] (S1) using GinvS3 chain. The reference uses NEW esi[1] here
        # because update_eigen_value just rewrote state['eigen_sqrt_inv'][1].
        GinvS3h = jnp.einsum("ijb,bm->ijm", grad, invS_h2)
        GinvS3Q2 = jnp.einsum("ijm,jq->iqm", GinvS3h, st.Q[1])
        GinvS3Q2Q1 = jnp.einsum("iqm,ip->pqm", GinvS3Q2, st.Q[0])
        GinvS32h = GinvS3Q2 * new_esi1[None, :, None]
        GinvS32G_T = jnp.tensordot(GinvS32h, GinvS32h, axes=[[1, 2], [1, 2]])
        new_GG0 = beta * st.GG[0] + (1.0 - beta) / (total_factor / d0) * GinvS32G_T

        # Update esi[0] (S1 diag) using NEW esi[1].
        diag_for_esi0 = jnp.mean(
            (GinvS3Q2Q1 * new_esi1[None, :, None]) ** 2, axis=(1, 2)
        )
        new_esi0 = _update_eigen_value(
            st.eigen_sqrt_inv[0], diag_for_esi0, beta,
            hp["using_clamping"], hp["max_clamp_value"], cast_dtype, eigh_dtype,
        )

        # 6) Q recompute every T steps
        new_step = st.step + jnp.int32(1)
        do_recompute = jnp.logical_and(new_step > 0, jnp.mod(new_step, hp["precondition_frequency"]) == 0)
        new_Q0 = jax.lax.cond(do_recompute, lambda _: _recompute_q(new_GG0, st.Q[0], cast_dtype, eigh_dtype), lambda _: st.Q[0], operand=None)
        new_Q1 = jax.lax.cond(do_recompute, lambda _: _recompute_q(new_GG1, st.Q[1], cast_dtype, eigh_dtype), lambda _: st.Q[1], operand=None)
        new_Q2 = jax.lax.cond(do_recompute, lambda _: _recompute_q(new_GG2, st.Q[2], cast_dtype, eigh_dtype), lambda _: st.Q[2], operand=None)

        new_st = _PerParamState(
            step=new_step,
            exp_avg=new_exp,
            GG=(new_GG0, new_GG1, new_GG2),
            Q=(new_Q0, new_Q1, new_Q2),
            eigen_sqrt_inv=(new_esi0, new_esi1, new_esi2),
            initialized=jnp.bool_(True),
        )
        return new_st, upd

    return jax.lax.cond(st.initialized, _regular, _first_iter, operand=None)


def _step_passthrough(grad, st, hp):
    """No-op for unsupported ranks (1D / scalar). Should not be called if routing
    is correct, but provided for safety."""
    return st, grad.astype(hp["cast_dtype"])


def kl_shampoo(
    learning_rate=1.0,                      # ignored; chain with scale_by_learning_rate
    b1: float = 0.9,
    shampoo_b: float = 0.98,
    eps: float = 1e-8,
    eps_scale: Optional[chex.ArrayTree] = None,
    precondition_frequency: int = 10,
    init_factor: float = 0.1,
    max_clamp_value: int = 4000,
    using_clamping: bool = True,
    cast_dtype=jnp.bfloat16,
    eigh_dtype=jnp.float32,
) -> optax.GradientTransformation:
    """KL-Shampoo as an optax GradientTransformation.

    The transform returns the *positive* preconditioned gradient. The minus sign
    and the learning rate are applied by the caller via
    `optax.scale_by_learning_rate`. Decoupled weight decay is the caller's
    responsibility (`optax.add_decayed_weights`).

    Only 2D and 3D leaves are preconditioned. Other ranks are passed through
    unchanged (cast to `cast_dtype`); use `optax.multi_transform` to route them
    to AdamW or another optimizer.

    Args:
      eps_scale: optional pytree of per-leaf eps multipliers (same structure as
        the params). If provided, the effective per-leaf eps is `eps * eps_scale`.
        Useful for μCompletedP-style per-tensor scaling. If None (default),
        the scalar `eps` is broadcast uniformly across all leaves.
    """
    del learning_rate  # accepted for parity with optax.adamw signature; unused

    base_hp = dict(
        b1=b1, shampoo_b=shampoo_b,
        precondition_frequency=precondition_frequency,
        init_factor=init_factor,
        max_clamp_value=max_clamp_value, using_clamping=using_clamping,
        cast_dtype=cast_dtype, eigh_dtype=eigh_dtype,
    )

    def init_fn(params):
        inner = jax.tree_util.tree_map(lambda p: _make_init_state(p, cast_dtype), params)
        return KLShampooState(inner=inner)

    def update_fn(updates, state, params=None):
        del params  # we work from updates' shapes/ranks

        # Build a per-leaf effective eps tree whose treedef MATCHES `updates`.
        # When this transform is wrapped by `optax.multi_transform`, `updates`
        # contains MaskedNode placeholders at non-KL positions; `eps_scale`
        # (captured at factory time) does not. We zip them through tree_map
        # with `is_leaf=_is_masked` so MaskedNodes terminate traversal.
        if eps_scale is None:
            eps_tree = jax.tree_util.tree_map(
                lambda u: u if _is_masked(u) else jnp.asarray(eps, eigh_dtype),
                updates,
                is_leaf=_is_masked,
            )
        else:
            eps_tree = jax.tree_util.tree_map(
                lambda u, s: u if _is_masked(u)
                else jnp.asarray(eps, eigh_dtype) * jnp.asarray(s, eigh_dtype),
                updates, eps_scale,
                is_leaf=_is_masked,
            )

        def _per_leaf(g, st, leaf_eps):
            hp = {**base_hp, "eps": leaf_eps}
            if not _supported_rank(g):
                return _StepOut(state=st, update=_step_passthrough(g, st, hp)[1])
            if g.ndim == 2:
                new_st, upd = _step_2d(g, st, hp)
            elif g.ndim == 3:
                new_st, upd = _step_3d(g, st, hp)
            else:
                raise AssertionError(f"unreachable: ndim={g.ndim}")
            return _StepOut(state=new_st, update=upd.astype(g.dtype))

        pairs = jax.tree_util.tree_map(
            _per_leaf, updates, state.inner, eps_tree,
            is_leaf=_is_per_param_state,
        )
        new_inner = jax.tree_util.tree_map(
            lambda x: x.state, pairs, is_leaf=_is_step_out,
        )
        new_upd = jax.tree_util.tree_map(
            lambda x: x.update, pairs, is_leaf=_is_step_out,
        )
        return new_upd, KLShampooState(inner=new_inner)

    return optax.GradientTransformation(init_fn, update_fn)
