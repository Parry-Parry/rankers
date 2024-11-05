import jax
from jax import numpy as jnp

def reduce(a : jax.Array, reduction : str):
    """
    Reducing a jax.Array along a given dimension.
    Parameters
    ----------
    a: jax.Array
        the input jax.Array
    reduction: str
        the reduction type
    Returns
    -------
    jax.Array
        the reduced jax.Array
    """
    if reduction == 'none':
        return a
    if reduction == 'mean':
        return jnp.mean(a)
    if reduction == 'sum':
        return jnp.sum(a)
    if reduction == 'batchmean':
        return jnp.sum(jnp.mean(a, axis=0))
    raise ValueError(f"Unknown reduction type: {reduction}")

class FlaxBaseLoss(object):
    """
    Base class for Losses

    Parameters
    ----------
    reduction: str
        the reduction type
    """
    def __init__(self, reduction : str = 'mean') -> None:
        self.reduction = reduction
    
    def _reduce(self, a : jax.Array):
        return reduce(a, self.reduction)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

def normalize(a: jax.Array, axis: int = -1):
    """
    Normalizing a jax.Array along a given dimension.
    Parameters
    ----------
    a: jax.Array
        the input jax.Array
    axis: int
        the dimension to normalize along
    Returns
    -------
    jax.Array
        the normalized jax.Array
    """
    min_values = jnp.min(a, axis=axis, keepdims=True)
    max_values = jnp.max(a, axis=axis, keepdims=True)
    return (a - min_values) / (max_values - min_values + 1e-10)

def residual(a : jax.Array):
    """
    Calculating the residual between a positive sample and multiple negatives.
    Parameters
    ----------
    a: jax.Array
        the input jax.Array
    Returns
    -------
    jax.Array
        the residuals
    """
    if jnp.size(a, 1) == 1: return a
    if len(jnp.size(a)) == 3:
        assert jnp.size(a, 2) == 1, "Expected scalar values for residuals."
        a = jnp.squeeze(a, axis=2)

    positive = a[:, 0]
    negative = a[:, 1]

    return positive - negative

def dot_product(a: jax.Array, b: jax.Array):
    """
    Calculating row-wise dot product between two jax.Arrays a and b.
    a and b must have the same dimensionality.
    Parameters
    ----------
    a: jax.Array
        size: batch_size x vector_dim
    b: jax.Array
        size: batch_size x vector_dim
    Returns
    -------
    jax.Array: size of (batch_size x 1)
        dot product for each pair of vectors
    """
    return (a * b).sum(dim=-1)


def cross_dot_product(a: jax.Array, b: jax.Array):
    """
    Calculating the cross doc product between each row in a with every row in b. a and b must have the same number of columns, but can have varied nuber of rows.
    Parameters
    ----------
    a: jax.Array
        size: (batch_size_1,  vector_dim)
    b: jax.Array
        size: (batch_size_2, vector_dim)
    Returns
    -------
    jax.Array: of size (batch_size_1, batch_size_2) where the value at (i,j) is dot product of a[i] and b[j].
    """
    return jnp.matmul(a, b.transpose(0, 1))

def batched_dot_product(a: jax.Array, b: jax.Array):
    """
    Calculating the dot product between two jax.Arrays a and b.

    Parameters
    ----------
    a: jax.Array
        size: batch_size x 1 x vector_dim
    b: jax.Array
        size: batch_size x group_size x vector_dim
    Returns
    -------
    jax.Array: size of (batch_size x group_size)
        dot product for each group of vectors
    """
    if len(b.shape) == 2:
        return jnp.matmul(a, b.transpose(0, 1))
    return jnp.matmul(a,jnp.permute(b,[0,2,1])).squeeze(1)

def num_non_zero(a: jax.Array):
    """
    Calculating the average number of non-zero columns in each row.
    Parameters
    ----------
    a: jax.Array
        the input jax.Array
    """
    return jnp.mean(jnp.sum((a > 0), axis=1))
    
from . import listwise as listwise
from . import pointwise as pointwise
from . import pairwise as pairwise

from .listwise import *
from .pointwise import *
from .pairwise import *

__all__ = [*listwise.__all__, *pointwise.__all__, *pairwise.__all__]