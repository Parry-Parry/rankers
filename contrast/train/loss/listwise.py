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
    """RankNet loss
    Use with caution, this is a WIP
    """

    def __init__(self, reduction='mean', temperature=1.):
        super().__init__(reduction)
        self.temperature = temperature
        self.bce = torch.nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, pred: Tensor, labels: Tensor=None) -> Tensor:
        _, g = pred.shape
        i1, i2 = torch.triu_indices(g, g, offset=1)
        pred_diff = pred[:, i1] - pred[:, i2]
        label_diff = labels[:, i1] - labels[:, i2]

        targets = (label_diff > 0).float()

        return self.bce(pred_diff, targets)


class DistillRankNetLoss(BaseLoss):
    """DistillRankNet loss
    Very much a WIP from https://arxiv.org/pdf/2402.10769
    DO NOT USE
    """
    def __init__(self, reduction='mean', temperature=1., base_margin=300., increment_margin=100.):
        super().__init__(reduction)
        self.temperature = temperature
        self.base_margin = base_margin
        self.increment_margin = increment_margin
    
    def forward(self, pred: Tensor, labels: Tensor) -> Tensor:
        _, g = pred.shape
        i1, i2 = torch.triu_indices(g, g, offset=1)

        pred_diff = pred[:, i1] - pred[:, i2]

        label_diff = labels[:, i1] - labels[:, i2]
        label_margin = (label_diff -1) * self.increment_margin + self.base_margin

        final_margin = pred_diff + label_margin
        targets = (label_diff > 0).float()

        return self._reduce(final_margin[targets])

class ListNetLoss(BaseLoss):
    """ListNet loss
    """

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

LISTWISE_LOSSES = {
    'kl_div': KL_DivergenceLoss,
    'ranknet': RankNetLoss,
    'distill_ranknet': DistillRankNetLoss,
    'listnet': ListNetLoss,
    'poly1': Poly1SoftmaxLoss,
}