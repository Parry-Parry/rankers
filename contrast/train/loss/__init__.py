from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor

def reduce(a : torch.Tensor, reduction : str):
    """
    Reducing a tensor along a given dimension.
    Parameters
    ----------
    a: torch.Tensor
        the input tensor
    reduction: str
        the reduction type
    Returns
    -------
    torch.Tensor
        the reduced tensor
    """
    if reduction == 'none':
        return a
    if reduction == 'mean':
        return a.mean()
    if reduction == 'sum':
        return a.sum()
    if reduction == 'batchmean':
        return a.mean(dim=0).sum()
    raise ValueError(f"Unknown reduction type: {reduction}")

class BaseLoss(nn.Module):
    """
    Base class for Losses

    Parameters
    ----------
    reduction: str
        the reduction type
    """
    def __init__(self, reduction : str = 'mean') -> None:
        super(BaseLoss, self).__init__()
        self.reduction = reduction
    
    def _reduce(self, a : torch.Tensor):
        return reduce(a, self.reduction)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

def normalize(a: Tensor, dim: int = -1):
    """
    Normalizing a tensor along a given dimension.
    Parameters
    ----------
    a: torch.Tensor
        the input tensor
    dim: int
        the dimension to normalize along
    Returns
    -------
    torch.Tensor
        the normalized tensor
    """
    min_values = a.min(dim=dim, keepdim=True)[0]
    max_values = a.max(dim=dim, keepdim=True)[0]
    return (a - min_values) / (max_values - min_values + 1e-10)

def residual(a : Tensor):
    """
    Calculating the residual between a positive sample and multiple negatives.
    Parameters
    ----------
    a: torch.Tensor
        the input tensor
    Returns
    -------
    torch.Tensor
        the residuals
    """
    if a.size(1) == 1: return a
    if len(a.size()) == 3:
        assert a.size(2) == 1, "Expected scalar values for residuals."
        a = a.squeeze(2)

    positive = a[:, 0]
    negative = a[:, 1]

    return positive - negative

def dot_product(a: Tensor, b: Tensor):
    """
    Calculating row-wise dot product between two tensors a and b.
    a and b must have the same dimensionality.
    Parameters
    ----------
    a: torch.Tensor
        size: batch_size x vector_dim
    b: torch.Tensor
        size: batch_size x vector_dim
    Returns
    -------
    torch.Tensor: size of (batch_size x 1)
        dot product for each pair of vectors
    """
    return (a * b).sum(dim=-1)


def cross_dot_product(a: Tensor, b: Tensor):
    """
    Calculating the cross doc product between each row in a with every row in b. a and b must have the same number of columns, but can have varied nuber of rows.
    Parameters
    ----------
    a: torch.Tensor
        size: (batch_size_1,  vector_dim)
    b: torch.Tensor
        size: (batch_size_2, vector_dim)
    Returns
    -------
    torch.Tensor: of size (batch_size_1, batch_size_2) where the value at (i,j) is dot product of a[i] and b[j].
    """
    return torch.mm(a, b.transpose(0, 1))

def batched_dot_product(a: Tensor, b: Tensor):
    """
    Calculating the dot product between two tensors a and b.

    Parameters
    ----------
    a: torch.Tensor
        size: batch_size x 1 x vector_dim
    b: torch.Tensor
        size: batch_size x group_size x vector_dim
    Returns
    -------
    torch.Tensor: size of (batch_size x group_size)
        dot product for each group of vectors
    """
    if len(b.shape) == 2:
        return torch.matmul(a, b.transpose(0, 1))
    return torch.bmm(a,torch.permute(b,[0,2,1])).squeeze(1)

def num_non_zero(a: Tensor):
    """
    Calculating the average number of non-zero columns in each row.
    Parameters
    ----------
    a: torch.Tensor
        the input tensor
    """
    return (a > 0).float().sum(dim=1).mean()

class dotLoss(nn.Module):
    """
    Wrapper for Dot Model Losses

    Parameters
    ----------
    fn: callable
        the loss function
    group_size: int
        the number of samples
    """
    def __init__(self, fn : callable, group_size=2, **kwargs) -> None:
        super(dotLoss, self).__init__()
        self.group_size = group_size
        self.fn = fn
    
    def forward(self, q_reps, d_reps, labels=None):
        batch_size = q_reps.size(0)
        e_q = q_reps.reshape(batch_size, 1, -1)
        e_d = d_reps.reshape(batch_size, self.group_size, -1)
        pred = batched_dot_product(e_q, e_d)
        if labels is not None: labels = labels.reshape(batch_size, self.group_size)
        loss = self.fn(pred, labels)

        to_log = {
            "loss_no_reg": loss.detach(),
        }
        return (
            loss,
            pred,
            to_log,
        )

class catLoss(nn.Module):
    """
    Wrapper for Cat Model Losses

    Parameters
    ----------
    fn: callable
        the loss function
    group_size: int
        the number of samples
    """
    def __init__(self, fn : callable, group_size=2, **kwargs) -> None:
        super(catLoss, self).__init__()
        self.group_size = group_size
        self.fn = fn
    
    def forward(self, logits, labels=None):
        pred = logits.reshape(-1, self.group_size, 2)
        pred = pred[:, :, 1]
        
        if labels is not None: labels = labels.reshape(-1, self.group_size)
        loss = self.fn(pred, labels)

        to_log = {
            "loss_no_reg": loss.detach(),
        }
        return (
            loss,
            pred,
            to_log,
        )

class duoLoss(nn.Module):
    """
    Wrapper for Duo Model Losses

    Parameters
    ----------
    fn: callable
        the loss function
    group_size: int
        the number of samples
    """
    def __init__(self, fn : callable, group_size=2, **kwargs) -> None:
        super(duoLoss, self).__init__()
        self.group_size = group_size
        self.fn = fn

    def forward(self, logits, labels):
        loss = self.fn(logits, labels)
        to_log = {
            "loss_no_reg": loss.detach(),
        }
        return (
            loss,
            F.softmax(logits, dim=-1)[:, 0],
            to_log,
        )
    
class WeightedLoss(nn.Module):
    """
    Wrapper for combining multiple losses

    Parameters
    ----------
    losses: list
        the list of losses
    weights: list
        the weights for each loss
    """
    def __init__(self, losses : list, weights : list = None) -> None:
        super(WeightedLoss, self).__init__()
        self.losses = losses
        self.weights = [1.0 for _ in losses] if not weights else weights
        
    def forward(self, *args, **kwargs):
        loss = 0.
        scores = None
        to_log = {}
        for w, l in zip(self.weights, self.losses):
            curr_scores, l, curr_log = l(*args, **kwargs)
            if curr_scores is not None: scores = curr_scores
            loss += w * l
            to_log.update(curr_log)
        return (loss, scores, to_log)
    
CONSTRUCTORS = defaultdict(lambda: dotLoss)

CONSTRUCTORS.update({
    'dot': dotLoss,
    'cat': catLoss,
    'duo': duoLoss,
})
    
from .listwise import *
from .pointwise import *
from .pairwise import *

LOSSES = {
    **POINTWISE_LOSSES,
    **PAIRWISE_LOSSES,
    **LISTWISE_LOSSES,
}