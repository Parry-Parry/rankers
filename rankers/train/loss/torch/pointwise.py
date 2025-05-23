import torch
from torch import Tensor
from torch.nn import functional as F
from ..util import register_loss
from . import BaseLoss


@register_loss("pointwise_mse")
class PointwiseMSELoss(BaseLoss):
    """PointwiseMSE loss"""

    name = "PointwiseMSE"

    def forward(self, pred: Tensor, labels: Tensor, **kwargs) -> Tensor:
        flattened_pred = pred.view(-1)
        flattened_labels = labels.view(-1)
        return F.mse_loss(flattened_pred, flattened_labels, reduction=self.reduction)


__all__ = [
    "PointwiseMSELoss",
]
