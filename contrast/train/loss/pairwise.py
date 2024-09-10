import torch
from torch import Tensor
import torch.nn.functional as F
from . import BaseLoss

residual = lambda x : x[:, 0].unsqueeze(1) - x[:, 1:]

class MarginMSELoss(BaseLoss):
    """Margin MSE loss with residual calculation."""

    def forward(self, pred: Tensor, labels: Tensor) -> Tensor:
        residual_pred = pred[:, 0].unsqueeze(1) - pred[:, 1:]
        residual_label = labels[:, 0].unsqueeze(1) - labels[:, 1:]
        return F.mse_loss(residual_pred, residual_label, reduction=self.reduction)


class HingeLoss(BaseLoss):
    """Hinge loss with sigmoid activation and residual calculation."""

    def __init__(self, margin=1, reduction='mean'):
        super().__init__(reduction)
        self.margin = margin

    def forward(self, pred: Tensor, labels: Tensor) -> Tensor:
        pred_residuals = F.relu(residual(F.sigmoid(pred)))
        label_residuals = torch.sign(residual(F.sigmoid(labels)))
        return self._reduce(F.relu(self.margin - (label_residuals * pred_residuals)))


class ClearLoss(BaseLoss):
    """Clear loss with margin and residual calculation."""

    def __init__(self, margin=1, reduction='mean'):
        super().__init__(reduction)
        self.margin = margin

    def forward(self, pred: Tensor, labels: Tensor) -> Tensor:
        margin_b = self.margin - residual(labels)
        return self._reduce(F.relu(margin_b - residual(pred)))
    
class LCELoss(BaseLoss):
    """LCE loss: Cross Entropy for NCE with localised examples."""
    def forward(self, pred: Tensor, labels: Tensor=None) -> Tensor:
        if labels is not None:
            labels = labels.argmax(dim=1)
        else:
            labels = torch.zeros(pred.size(0), dtype=torch.long, device=pred.device)
        return F.cross_entropy(pred, labels, reduction=self.reduction)


class ContrastiveLoss(BaseLoss):
    """Contrastive loss with log_softmax and negative log likelihood."""

    def __init__(self, reduction='mean', temperature=1.):
        super().__init__(reduction)
        self.temperature = temperature

    def forward(self, pred: Tensor, labels : Tensor = None) -> Tensor:
        softmax_scores = F.log_softmax(pred / self.temperature, dim=1)
        labels = labels.argmax(dim=1) if labels is not None else torch.zeros(pred.size(0), dtype=torch.long, device=pred.device).view(-1, 1)
        return F.nll_loss(softmax_scores, labels, reduction=self.reduction)

PAIRWISE_LOSSES = {
    'margin_mse': MarginMSELoss,
    'hinge': HingeLoss,
    'clear': ClearLoss,
    'lce': LCELoss,
    'contrastive': ContrastiveLoss,
}