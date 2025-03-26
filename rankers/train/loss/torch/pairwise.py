import torch
from torch import Tensor
import torch.nn.functional as F
from ..util import register_loss
from . import BaseLoss


def residual(x):
    return x[:, 0].unsqueeze(1) - x[:, 1:]


@register_loss("margin_mse")
class MarginMSELoss(BaseLoss):
    """Margin MSE loss with residual calculation."""

    name = "MarginMSE"

    def forward(self, pred: Tensor, labels: Tensor, **kwargs) -> Tensor:
        residual_pred = residual(pred)
        residual_label = residual(labels)
        return F.mse_loss(residual_pred, residual_label, reduction=self.reduction)


@register_loss("hinge")
class HingeLoss(BaseLoss):
    """Hinge loss with sigmoid activation and residual calculation."""

    name = "Hinge"

    def __init__(self, margin=1, reduction="mean"):
        super().__init__(reduction)
        self.margin = margin

    def forward(self, pred: Tensor, labels: Tensor, **kwargs) -> Tensor:
        pred_residuals = F.relu(residual(F.sigmoid(pred)))
        label_residuals = torch.sign(residual(F.sigmoid(labels)))
        return self._reduce(F.relu(self.margin - (label_residuals * pred_residuals)))


@register_loss("clear")
class ClearLoss(BaseLoss):
    """Clear loss with margin and residual calculation."""

    name = "CLEAR"

    def __init__(self, margin=1, residual_lambda=0.1, reduction="mean"):
        super().__init__(reduction)
        self.margin = margin
        self.residual_lambda = residual_lambda

    def forward(self, pred: Tensor, labels: Tensor, **kwargs) -> Tensor:
        margin_b = self.margin - (self.residual_lambda * residual(labels))
        return self._reduce(F.relu(margin_b - residual(pred)))


@register_loss("lce")
class LCELoss(BaseLoss):
    """LCE loss: Cross Entropy for NCE with localised examples."""

    name = "LCE"

    def forward(self, pred: Tensor, labels: Tensor =  None, **kwargs) -> Tensor:
        if labels is not None:
            labels = labels.argmax(dim=1)
        else:
            labels = torch.zeros(pred.size(0), dtype=pred.dtype, device=pred.device)
        return F.cross_entropy(pred, labels, reduction=self.reduction)


@register_loss("contrastive")
class ContrastiveLoss(BaseLoss):
    """Contrastive loss with log_softmax and negative log likelihood."""

    name = "Contrastive"

    def __init__(self, reduction="mean", temperature=1.0):
        super().__init__(reduction)
        self.temperature = temperature

    def forward(self, pred: Tensor, labels: Tensor =  None, **kwargs) -> Tensor:
        softmax_scores = F.log_softmax(pred / self.temperature, dim=1)
        labels = (
            labels.argmax(dim=1)
            if labels is not None
            else torch.zeros(pred.size(0), dtype=pred.dtype, device=pred.device).view(
                -1, 1
            )
        )
        return F.nll_loss(softmax_scores, labels, reduction=self.reduction)


__all__ = [
    "MarginMSELoss",
    "HingeLoss",
    "ClearLoss",
    "LCELoss",
    "ContrastiveLoss",
]
