import torch
from torch import Tensor
from torch.nn import functional as F
from . import BaseLoss

class KL_DivergenceLoss(BaseLoss):
    """KL Divergence loss"""

    def __init__(self, reduction='batchmean', temperature=1.):
        super().__init__(reduction)
        self.temperature = temperature
        self.kl_div = torch.nn.KLDivLoss(reduction=self.reduction)

    def forward(self, pred: Tensor, labels: Tensor) -> Tensor:
        return self.kl_div(F.log_softmax(pred / self.temperature, dim=1), F.softmax(labels / self.temperature, dim=1))


class RankNetLoss(BaseLoss):
    """RankNet loss"""

    def __init__(self, reduction='mean', temperature=1.):
        super().__init__(reduction)
        self.temperature = temperature
        self.bce = torch.nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, pred: Tensor, labels: Tensor=None) -> Tensor:
        neg = pred[:, 1:]
        pos = pred[:, 0]
        pos = torch.repeat_interleave(pos, neg.shape[0] // pos.shape[0],dim=0)

        residual = pos - neg
        if labels is None: labels = torch.ones_like(residual)

        return self.bce(residual, labels)

class ListNetLoss(BaseLoss):
    """ListNet loss"""

    def __init__(self, reduction='mean', temperature=1., epsilon=1e-8):
        super().__init__(reduction)
        self.temperature = temperature
        self.epsilon = epsilon

    def forward(self, pred: Tensor, labels: Tensor) -> Tensor:
        if not torch.all((labels >= 0) & (labels <= 1)):
            labels = F.softmax(labels / self.temperature, dim=1)
        return self._reduce(-torch.sum(labels * F.log_softmax(pred + self.epsilon  / self.temperature, dim=1), dim=-1))

class Poly1SoftmaxLoss(BaseLoss):
    """Poly1 softmax loss with automatic softmax handling and reduction."""

    def __init__(self, reduction='mean', epsilon : float = 1., temperature=1.):
        super().__init__(reduction)
        self.epsilon = epsilon
        self.temperature = temperature
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred: Tensor, labels: Tensor) -> Tensor:
        labels_for_softmax = torch.divide(labels, labels.sum(dim=1))
        expansion = (labels_for_softmax * F.softmax(pred / self.temperature, dim=1)).sum(dim=-1)
        ce = self.ce(pred / self.temperature, labels_for_softmax)
        return self._reduce(ce + (1 - expansion) * self.epsilon)