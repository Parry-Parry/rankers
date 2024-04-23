import torch
from torch import Tensor
from torch.nn import functional as F
from . import BaseLoss

class KL_DivergenceLoss(BaseLoss):
    """KL Divergence loss with automatic softmax and reduction handling."""

    def __init__(self, reduction='batchmean', temperature=1.):
        super().__init__(reduction)
        self.temperature = temperature

    def forward(self, pred: Tensor, labels: Tensor) -> Tensor:
        return F.kl_div(F.log_softmax(pred / self.temperature, dim=1), F.softmax(labels / self.temperature, dim=1), reduction=self.reduction)


class RankNetLoss(BaseLoss):
    """RankNet loss with automatic softmax handling and reduction."""

    def __init__(self, reduction='mean', temperature=1.):
        super().__init__(reduction)
        self.temperature = temperature

    def forward(self, pred: Tensor, labels: Tensor) -> Tensor:
        if not torch.all((labels >= 0) & (labels <= 1)):
            labels = F.softmax(labels  / self.temperature, dim=1)
        if not torch.all((pred >= 0) & (pred <= 1)):
            pred = F.softmax(pred  / self.temperature, dim=1)
        return F.soft_margin_loss(pred, labels, reduction=self.reduction)


class ListNetLoss(BaseLoss):
    """ListNet loss with automatic softmax handling and reduction."""

    def __init__(self, reduction='mean', temperature=1.):
        super().__init__(reduction)
        self.temperature = temperature

    def forward(self, pred: Tensor, labels: Tensor) -> Tensor:
        if not torch.all((labels >= 0) & (labels <= 1)):
            labels = F.softmax(labels / self.temperature, dim=1)
        return self._reduce(-torch.sum(labels * F.log_softmax(pred  / self.temperature, dim=1), dim=-1))

class Poly1SoftmaxLoss(BaseLoss):
    """Poly1 softmax loss with automatic softmax handling and reduction."""

    def __init__(self, reduction='mean', epsilon : float = 1., temperature=1.):
        super().__init__(reduction)
        self.epsilon = epsilon
        self.temperature = temperature

    def forward(self, pred: Tensor, labels: Tensor) -> Tensor:
        labels_for_softmax = torch.divide(labels, labels.sum(dim=1))
        expansion = labels_for_softmax * F.softmax(pred / self.temperature, dim=1)
        ce = F.cross_entropy(pred / self.temperature, labels_for_softmax, reduction='none')
        return self._reduce(ce + (1 - expansion) * self.epsilon)