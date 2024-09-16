import jax 
import jax.numpy as jnp
import jax.nn as nn
import optax.losses as L
from jax import jit
from . import FlaxBaseLoss

residual = lambda x : jnp.expand_dims(x[:, 0], axis=1) - x[:, 1:]

class FlaxMarginMSELoss(FlaxBaseLoss):
    """Margin MSE loss with residual calculation."""

    @jit
    def forward(self, pred: jax.Array, labels: jax.Array) -> jax.Array:
        residual_pred = residual(pred)
        residual_label = residual(labels)
        return self._reduce(L.squared_error(residual_pred, residual_label))


class FlaxHingeLoss(FlaxBaseLoss):
    """Hinge loss with sigmoid activation and residual calculation."""

    def __init__(self, margin=1, reduction='mean'):
        super().__init__(reduction)
        self.margin = margin

    @jit
    def forward(self, pred: jax.Array, labels: jax.Array) -> jax.Array:
        pred_residuals = nn.relu(residual(nn.sigmoid(pred)))
        label_residuals = jnp.sign(residual(nn.sigmoid(labels)))
        return self._reduce(nn.relu(self.margin - (label_residuals * pred_residuals)))


class FlaxClearLoss(FlaxBaseLoss):
    """Clear loss with margin and residual calculation."""

    def __init__(self, margin=1, reduction='mean'):
        super().__init__(reduction)
        self.margin = margin

    @jit
    def forward(self, pred: jax.Array, labels: jax.Array) -> jax.Array:
        margin_b = self.margin - residual(labels)
        return self._reduce(nn.relu(margin_b - residual(pred)))
    
class FlaxLCELoss(FlaxBaseLoss):
    """LCE loss: Cross Entropy for NCE with localised examples."""

    @jit
    def forward(self, pred: jax.Array, labels: jax.Array=None) -> jax.Array:
        labels = jnp.argmax(labels, axis=1) if labels is not None else jnp.zeros(jnp.size(pred, 0))
        return self._reduce(L.softmax_cross_entropy(pred, labels))


class FlaxContrastiveLoss(FlaxBaseLoss):
    """Contrastive loss with log_softmax and negative log likelihood."""

    def __init__(self, reduction='sum', temperature=1.):
        super().__init__(reduction)
        self.temperature = temperature

    @jit
    def forward(self, pred: jax.Array, labels : jax.Array = None) -> jax.Array:
        softmax_scores = nn.log_softmax(pred / self.temperature, axis=1)
        labels = jnp.argmax(labels, axis=1) if labels is not None else jnp.zeros(jnp.size(pred, 0))

        label_probs = softmax_scores * labels + (1 - labels) * (1 - softmax_scores)

        return self._reduce(-jnp.log(label_probs))

__all__ = ['FlaxMarginMSELoss', 'FlaxHingeLoss', 'FlaxClearLoss', 'FlaxLCELoss', 'FlaxContrastiveLoss', 'FlaxPAIRWISE_LOSSES']