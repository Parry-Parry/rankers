import optax.losses as L
import jax
from jax import jit
import jax.numpy as jnp
from . import FlaxBaseLoss

class FlaxPointwiseMSELoss(FlaxBaseLoss):
    """Pointwise MSE loss"""
    @jit
    def forward(self, pred: jax.Array, labels: jax.Array) -> jax.Array:
        flattened_pred = jnp.reshape(pred, (-1,))
        flattened_labels = jnp.reshape(labels, (-1,))
        return self._reduce(L.squared_error(flattened_pred, flattened_labels))

POINTWISE_LOSSES = {
    'mse': FlaxPointwiseMSELoss,
}

__all__ = ['FlaxPointwiseMSELoss']