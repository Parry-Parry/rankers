from . import BaseLoss, register_loss

@register_loss('flax_pointwise_mse')
class PointwiseMSELoss(BaseLoss):
    """Pointwise MSE loss"""
    name = "PointwiseMSE"
    def forward(self, pred: Tensor, labels: Tensor) -> Tensor:
        flattened_pred = pred.view(-1)
        flattened_labels = labels.view(-1)
        return F.mse_loss(flattened_pred, flattened_labels, reduction=self.reduction)

__all__ = [
    'PointwiseMSELoss',
]   
