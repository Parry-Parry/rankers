from . import FlaxBaseLoss
from ..util import register_loss
import jax.numpy as jnp
import optax.losses as L


@register_loss("flax_flax_pointwise_mse")
class PointwiseMSELoss(FlaxBaseLoss):
    """Pointwise MSE loss"""

    name = "PointwiseMSE"

    def forward(self, pred: jnp.Array, labels: jnp.Array) -> jnp.Array:
        flattened_pred = pred.view(-1)
        flattened_labels = labels.view(-1)
        return self._reduce(L.squared_error(flattened_pred, flattened_labels))


__all__ = [
    "PointwiseMSELoss",
]
