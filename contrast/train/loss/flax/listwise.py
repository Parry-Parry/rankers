import optax.losses as L
import jax
import jax.numpy as jnp
import jax.nn as nn
from jax import jit
from . import FlaxBaseLoss

class FlaxKL_DivergenceLoss(FlaxBaseLoss):
    """KL Divergence loss"""

    def __init__(self, reduction='batchmean', temperature=1.):
        super().__init__(reduction)
        self.temperature = temperature
        self.kl_div = L.convex_kl_divergence
    @jit
    def forward(self, pred: jnp.array, labels: jax.Array) -> jax.Array:
        return self._reduce(self.kl_div(nn.log_softmax(pred / self.temperature, axis=1), nn.softmax(labels / self.temperature, axis=1)))


class FlaxRankNetLoss(FlaxBaseLoss):
    """RankNet loss
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf
    """

    def __init__(self, reduction='mean', temperature=1.):
        super().__init__(reduction)
        self.temperature = temperature
        self.bce = L.sigmoid_binary_cross_entropy
    @jit
    def forward(self, pred: jax.Array, labels: jax.Array = None) -> jax.Array:
        g = jnp.shape(pred, 1)
        i1, i2 = jnp.triu_indices(g, k=1)
        pred_diff = pred[:, i1] - pred[:, i2]
        if labels is None:
            targets = jnp.zeros_like(pred_diff)
            targets[:, 0] = 1.
        else:
            label_diff = labels[:, i1] - labels[:, i2]
            targets = (label_diff > 0)

        return self._reduce(self.bce(pred_diff, targets))


class FlaxListNetLoss(FlaxBaseLoss):
    """ListNet loss
    """

    def __init__(self, reduction='mean', temperature=1., epsilon=1e-8):
        super().__init__(reduction)
        self.temperature = temperature
        self.epsilon = epsilon
    @jit
    def forward(self, pred: jax.Array, labels: jax.Array) -> jax.Array:
        if not jnp.all((labels >= 0) & (labels <= 1)):
            labels = nn.softmax(labels / self.temperature, axis=1)
        return self._reduce(-jnp.sum(labels * nn.log_softmax(pred + self.epsilon  / self.temperature, axis=1), axis=-1))

class FlaxPoly1SoftmaxLoss(FlaxBaseLoss):
    """Poly1 softmax loss with automatic softmax handling and reduction."""

    def __init__(self, reduction='mean', epsilon : float = 1., temperature=1.):
        super().__init__(reduction)
        self.epsilon = epsilon
        self.temperature = temperature
        self.ce = L.softmax_cross_entropy
    @jit
    def forward(self, pred: jax.Array, labels: jax.Array) -> jax.Array:
        labels_for_softmax = jnp.divide(labels, labels.sum(axis=1))
        expansion = jnp.sum((labels_for_softmax * nn.softmax(pred / self.temperature, axis=1)), axis=-1)
        ce = self.ce(pred / self.temperature, labels_for_softmax)
        return self._reduce(ce + (1 - expansion) * self.epsilon)

__all__ = ['FlaxKL_DivergenceLoss', 'FlaxRankNetLoss', 'FlaxListNetLoss', 'FlaxPoly1SoftmaxLoss']