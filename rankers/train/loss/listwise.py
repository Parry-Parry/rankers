import torch
from torch import Tensor
from torch.nn import functional as F
from . import BaseLoss, register_loss

@register_loss('kl_div')
class KL_DivergenceLoss(BaseLoss):
    """KL Divergence loss"""

    def __init__(self, reduction='batchmean', temperature=1.):
        super().__init__(reduction)
        self.temperature = temperature
        self.kl_div = torch.nn.KLDivLoss(reduction=self.reduction)

    def forward(self, pred: Tensor, labels: Tensor) -> Tensor:
        return self.kl_div(F.log_softmax(pred / self.temperature, dim=1), F.softmax(labels / self.temperature, dim=1))

@register_loss('ranknet')
class RankNetLoss(BaseLoss):
    """RankNet loss
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf
    """

    def __init__(self, reduction='mean', temperature=1.):
        super().__init__(reduction)
        self.temperature = temperature
        self.bce = torch.nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, pred: Tensor, labels: Tensor=None) -> Tensor:
        _, g = pred.shape
        i1, i2 = torch.triu_indices(g, g, offset=1)
        pred_diff = pred[:, i1] - pred[:, i2]
        if labels is None:
            labels = torch.zeros_like(pred_diff)
            labels[:, 0] = 1.
        else:
            label_diff = labels[:, i1] - labels[:, i2]
            labels = (label_diff > 0).float()

        return self.bce(pred_diff, labels)

@register_loss('distill_ranknet')
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
        labels = (label_diff > 0).float()

        return self._reduce(final_margin[labels])

@register_loss('listnet')
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

@register_loss('poly1')
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

def get_approx_ranks(pred: torch.Tensor, temperature: float) -> torch.Tensor:
    score_diff = pred[:, None] - pred[..., None]
    normalized_score_diff = torch.sigmoid(score_diff / temperature)
    # set diagonal to 0
    normalized_score_diff = normalized_score_diff * (1 - torch.eye(pred.shape[1], device=pred.device))
    approx_ranks = normalized_score_diff.sum(-1) + 1
    return approx_ranks

# Taken from https://github.com/webis-de/lightning-ir/blob/main/lightning_ir/loss/loss.py

def get_dcg(
        ranks: torch.Tensor,
        labels: torch.Tensor,
        k: int | None = None,
        scale_gains: bool = True,
    ) -> torch.Tensor:
        log_ranks = torch.log2(1 + ranks)
        discounts = 1 / log_ranks
        if scale_gains:
            gains = 2**labels - 1
        else:
            gains = labels
        dcgs = gains * discounts
        if k is not None:
            dcgs = dcgs.masked_fill(ranks > k, 0)
        return dcgs.sum(dim=-1)

def get_ndcg(
        ranks: torch.Tensor,
        labels: torch.Tensor,
        k: int | None = None,
        scale_gains: bool = True,
        optimal_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        labels = labels.clamp(min=0)
        if optimal_labels is None:
            optimal_labels = labels
        optimal_ranks = torch.argsort(torch.argsort(optimal_labels, descending=True))
        optimal_ranks = optimal_ranks + 1
        dcg = get_dcg(ranks, labels, k, scale_gains)
        idcg = get_dcg(optimal_ranks, optimal_labels, k, scale_gains)
        ndcg = dcg / (idcg.clamp(min=1e-12))
        return ndcg

@register_loss('approx_ndcg')
class ApproxNDCGLoss(BaseLoss):
    def __init__(self, reduction: str = 'mean', temperature=1., scale_gains: bool = True) -> None:
        super().__init__(reduction)
        self.temperature = temperature
        self.scale_gains = scale_gains
    
    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = self.process_labels(pred, labels)
        approx_ranks = get_approx_ranks(pred, self.temperature)
        ndcg = get_ndcg(approx_ranks, labels, k=None, scale_gains=self.scale_gains)
        loss = 1 - ndcg
        return self._reduce(loss)

def get_mrr(ranks: torch.Tensor, labels: torch.Tensor, k: int | None = None) -> torch.Tensor:
        labels = labels.clamp(None, 1)
        reciprocal_ranks = 1 / ranks
        mrr = reciprocal_ranks * labels
        if k is not None:
            mrr = mrr.masked_fill(ranks > k, 0)
        mrr = mrr.max(dim=-1)[0]
        return mrr

@register_loss('approx_mrr')
class ApproxMRRLoss(BaseLoss):
    def __init__(self, reduction: str = 'mean', temperature=1.) -> None:
        super().__init__(reduction)
        self.temperature = temperature
    
    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        approx_ranks = get_approx_ranks(pred, self.temperature)
        mrr = get_mrr(approx_ranks, labels, k=None)
        loss = 1 - mrr
        return self._reduce(loss)


__all__ = [
    'KL_DivergenceLoss',
    'RankNetLoss',
    'DistillRankNetLoss',
    'ListNetLoss',
    'Poly1SoftmaxLoss',
    'ApproxNDCGLoss',
    'ApproxMRRLoss',
]