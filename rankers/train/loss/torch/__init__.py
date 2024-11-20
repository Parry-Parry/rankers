import torch
from torch import nn, Tensor

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

class RegWrapper(object):
        def __init__(self, obj : BaseLoss, q_weight=0.08, d_weight=0.1, t=0, T=1000):
            self.obj = obj
            self.q_weight = q_weight
            self.d_weight = d_weight
            self.name = obj.name
            self.t = t
            self.T = T
        
        def step_q(self):
            if self.t >= self.T:
                pass
            else:
                self.t += 1
                self.q_weight = self.q_weight * (self.t / self.T) ** 2
        
        def step_d(self):
            if self.t >= self.T:
                pass
            else:
                self.t += 1
                self.d_weight = self.d_weight * (self.t / self.T) ** 2
        
        def step(self):
            if self.t >= self.T:
                pass
            else:
                self.t += 1
                self.q_weight = self.q_weight * (self.t / self.T) ** 2
                self.d_weight = self.d_weight * (self.t / self.T) ** 2

        @staticmethod
        def reg(reg, weight=0):
            raise NotImplementedError

        def __call__(self, pred, labels=None, queries=None, documents=None, **kwargs):
            q_reg = self.reg(queries, self.q_weight) if queries is not None else 0
            d_reg = self.reg(documents, self.d_weight) if documents is not None else 0
            outputs = self.obj(pred, labels, **kwargs)
            loss_val = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            loss_val += q_reg + d_reg
            if isinstance(outputs, dict):
                outputs["loss"] = loss_val
            else:
                outputs = (loss_val, *outputs[1:])
            return outputs

def FLOPS_regularization(object, q_weight=0.08, d_weight=0.1, t=0, T=1000):
    class FLOPSWrapper(RegWrapper):
        @staticmethod
        def reg(reps, weight=0):
            return (torch.abs(reps).mean(dim=0) ** 2).sum() * weight
    return FLOPSWrapper(object, q_weight, d_weight, t, T)

def L1_regularization(object, q_weight=0.08, d_weight=0.1, t=0, T=1000):
    class L1Wrapper(RegWrapper):
        @staticmethod
        def reg(reps, weight=0):
            return torch.abs(reps).sum(dim=1).mean() * weight  
    return L1Wrapper(object, q_weight, d_weight, t, T)


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
        size: batch_size x vector_dim
    b: torch.Tensor
        size: batch_size x group_size x vector_dim
    Returns
    -------
    torch.Tensor: size of (batch_size x group_size)
        dot product for each group of vectors
    """
    if len(b.shape) == 2:
        return dot_product(a, b)

    # Ensure `a` is of shape (batch_size, 1, vector_dim)
    if len(a.shape) == 2:
        a = a.unsqueeze(1)
    
    # Compute batched dot product, result shape: (batch_size, 1, group_size)
    return (a * b).sum(dim=-1)

def maxsim(a : Tensor, b : Tensor, a_mask : Tensor, temperature : float = 1.0):
    scores = torch.einsum('qin,pjn->qipj', a, b)
    scores, _ = scores.max(-1)
    scores = scores.sum(1) / a_mask[:, 1:].sum(-1, keepdim=True)
    scores = scores / temperature
    return scores

def num_non_zero(a: Tensor):
    """
    Calculating the average number of non-zero columns in each row.
    Parameters
    ----------
    a: torch.Tensor
        the input tensor
    """
    return (a > 0).float().sum(dim=1).mean()
    
from . import listwise as listwise
from . import pointwise as pointwise
from . import pairwise as pairwise

from .listwise import *
from .pointwise import *
from .pairwise import *

__all__ = [*listwise.__all__, *pointwise.__all__, *pairwise.__all__]