import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from .listwise import *
from .pointwise import *
from .pairwise import *

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
    def __init__(self, fn : callable, num_negatives=1, margin=1., **kwargs) -> None:
        super(dotLoss, self).__init__()
        self.num_negatives = num_negatives
        self.margin = margin
        self.fn = fn
    
    def forward(self, q_reps, d_reps, labels):
        batch_size = q_reps.size(0)
        e_q = q_reps.view(batch_size, 1, -1)
        e_d = d_reps.view(batch_size, self.num_negatives+1, -1)
        pred = batched_dot_product(e_q, e_d)
        labels = labels.view(batch_size, self.num_negatives+1)
        loss = self.fn(pred, labels, margin=self.margin)

        to_log = {
            "loss_no_reg": loss.detach(),
        }
        return (
            pred,
            loss,
            to_log,
        )

class catLoss(nn.Module):
    def __init__(self, fn : callable, num_negatives=1, margin=1., **kwargs) -> None:
        super(catLoss, self).__init__()
        self.num_negatives = num_negatives
        self.margin = margin
        self.fn = fn
    
    def forward(self, logits, labels):
        pred = F.softmax(logits, dim=-1)[:, 1]
        pred = pred.view(-1, self.num_negatives+1)
        labels = labels.view(-1, self.num_negatives+1)
        loss = self.fn(pred, labels, margin=self.margin)

        to_log = {
            "loss_no_reg": loss.detach(),
        }
        return (
            pred,
            loss,
            to_log,
        )

class duoLoss(nn.Module):
    def __init__(self, fn : callable, num_negatives=1, margin=1., **kwargs) -> None:
        super(duoLoss, self).__init__()
        self.num_negatives = num_negatives
        self.margin = margin
        self.fn = fn
    def forward(self, logits, labels):
        loss = self.fn(logits, labels, margin=self.margin)
        to_log = {
            "loss_no_reg": loss.detach(),
        }
        return (
            F.softmax(logits, dim=-1)[:, 0],
            loss,
            to_log,
        )
    
class CombiLoss(nn.Module):
    def __init__(self, losses : list, weights : list = None) -> None:
        super(CombiLoss, self).__init__()
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
        return (scores, loss, to_log)