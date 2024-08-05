import jax
from jax import numpy as jnp

def reduce(a : jnp.array, reduction : str):
    """
    Reducing a jnp.array along a given dimension.
    Parameters
    ----------
    a: jnp.array
        the input jnp.array
    reduction: str
        the reduction type
    Returns
    -------
    jnp.array
        the reduced jnp.array
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

class BaseLoss(object):
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
    
    def _reduce(self, a : jnp.array):
        return reduce(a, self.reduction)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

def normalize(a: jnp.array, dim: int = -1):
    """
    Normalizing a jnp.array along a given dimension.
    Parameters
    ----------
    a: jnp.array
        the input jnp.array
    dim: int
        the dimension to normalize along
    Returns
    -------
    jnp.array
        the normalized jnp.array
    """
    min_values = a.min(axis=dim, keepdim=True)[0]
    max_values = a.max(axis=dim, keepdim=True)[0]
    return (a - min_values) / (max_values - min_values + 1e-10)

def residual(a : jnp.array):
    """
    Calculating the residual between a positive sample and multiple negatives.
    Parameters
    ----------
    a: jnp.array
        the input jnp.array
    Returns
    -------
    jnp.array
        the residuals
    """
    if a.size(1) == 1: return a
    if len(a.size()) == 3:
        assert a.size(2) == 1, "Expected scalar values for residuals."
        a = a.squeeze(2)

    positive = a[:, 0]
    negative = a[:, 1]

    return positive - negative

def dot_product(a: jnp.array, b: jnp.array):
    """
    Calculating row-wise dot product between two jnp.arrays a and b.
    a and b must have the same dimensionality.
    Parameters
    ----------
    a: jnp.array
        size: batch_size x vector_dim
    b: jnp.array
        size: batch_size x vector_dim
    Returns
    -------
    jnp.array: size of (batch_size x 1)
        dot product for each pair of vectors
    """
    return (a * b).sum(dim=-1)


def cross_dot_product(a: jnp.array, b: jnp.array):
    """
    Calculating the cross doc product between each row in a with every row in b. a and b must have the same number of columns, but can have varied nuber of rows.
    Parameters
    ----------
    a: jnp.array
        size: (batch_size_1,  vector_dim)
    b: jnp.array
        size: (batch_size_2, vector_dim)
    Returns
    -------
    jnp.array: of size (batch_size_1, batch_size_2) where the value at (i,j) is dot product of a[i] and b[j].
    """
    return jnp.matmul(a, b.transpose(0, 1))

def batched_dot_product(a: jnp.array, b: jnp.array):
    """
    Calculating the dot product between two jnp.arrays a and b.

    Parameters
    ----------
    a: jnp.array
        size: batch_size x 1 x vector_dim
    b: jnp.array
        size: batch_size x group_size x vector_dim
    Returns
    -------
    jnp.array: size of (batch_size x group_size)
        dot product for each group of vectors
    """
    if len(b.shape) == 2:
        return jnp.matmul(a, b.transpose(0, 1))
    return jnp.matmul(a,torch.permute(b,[0,2,1])).squeeze(1)

def num_non_zero(a: jnp.array):
    """
    Calculating the average number of non-zero columns in each row.
    Parameters
    ----------
    a: jnp.array
        the input jnp.array
    """
    return (a > 0).float().sum(dim=1).mean()
    
from .listwise import *
from .pointwise import *
from .pairwise import *

FlaxLOSSES = {
    **POINTWISE_LOSSES,
    **PAIRWISE_LOSSES,
    **LISTWISE_LOSSES,
}