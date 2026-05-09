"""KL-Shampoo for JAX/optax.

A JAX port of the KL-Shampoo optimizer from Lin, Lowe, Dangel, Eschenhagen,
Xu & Grosse, "Understanding and Improving Shampoo and SOAP via
Kullback-Leibler Minimization" (arXiv:2509.03378). KL-SOAP is out of scope
for this package; only the KL-Shampoo branch is implemented.

Public API:
- `kl_shampoo`: an `optax.GradientTransformation` that returns the positive
  preconditioned gradient. Compose with `optax.add_decayed_weights` and
  `optax.scale_by_learning_rate` to form a full optimizer.
- `kl_shampoo_with_adamw`: convenience wrapper that routes 1D, scalar, and
  oversized matrices through AdamW while applying KL-Shampoo to 2D/3D
  preconditionable tensors.
- `param_labels`: the label function used by `kl_shampoo_with_adamw`,
  exposed in case you want to build the chain yourself.

The reference PyTorch implementation lives at github.com/yorkerlin/KL-Methods.
"""
from kl_shampoo_jax._core import KLShampooState, kl_shampoo
from kl_shampoo_jax._routing import kl_shampoo_with_adamw, param_labels

__all__ = ["kl_shampoo", "KLShampooState", "kl_shampoo_with_adamw", "param_labels"]
__version__ = "0.3.0"
