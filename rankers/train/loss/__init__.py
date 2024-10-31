import torch.nn as nn
import torch
from torch import Tensor
import functools

class SingletonMeta(type):
    """
    Metaclass to implement the Singleton design pattern.
    Ensures only one instance of a class can be created.
    """
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        """
        Controlled object creation to ensure only one instance exists.
        """
        if cls not in cls._instances:
            # If no instance exists, create one
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class LossFunctionRegistry(metaclass=SingletonMeta):
    """
    A singleton registry for managing and retrieving loss functions by name.
    Supports both built-in PyTorch losses and custom loss functions.
    """
    
    def __init__(self):
        # Check if the registry has already been initialized
        if not hasattr(self, '_registry'):
            # Dictionary to store registered loss functions
            self._registry = {}
            
            # Automatically register built-in PyTorch loss functions
            self._register_builtin_losses()
    
    def _register_builtin_losses(self):
        """
        Automatically register common PyTorch loss functions.
        """
        builtin_losses = {
            # Basic losses
            'mse': nn.MSELoss,
            'l1': nn.L1Loss,
            'cross_entropy': nn.CrossEntropyLoss,
            'nll': nn.NLLLoss,
            'binary_cross_entropy': nn.BCELoss,
            'binary_cross_entropy_with_logits': nn.BCEWithLogitsLoss,
            
            # Reduction variant losses
            'mse_sum': functools.partial(nn.MSELoss, reduction='sum'),
            'mse_none': functools.partial(nn.MSELoss, reduction='none'),
        }
        
        for name, loss_fn in builtin_losses.items():
            self.register(name, loss_fn)
    
    def register(self, name, loss_fn):
        """
        Register a loss function with a given name.
        
        Args:
            name (str): Name to register the loss function under
            loss_fn (callable): Loss function to register
        """
        if not callable(loss_fn):
            raise TypeError(f"Loss function {name} must be callable")
        
        if name in self._registry:
            print(f"Warning: Overwriting existing loss function '{name}'")
        
        self._registry[name] = loss_fn
    
    def get(self, name, **kwargs):
        """
        Retrieve a loss function by name.
        
        Args:
            name (str): Name of the loss function
            **kwargs: Additional arguments to pass to the loss function constructor
        
        Returns:
            callable: Instantiated loss function
        
        Raises:
            KeyError: If the loss function is not found in the registry
        """
        if name not in self._registry:
            available_losses = ", ".join(sorted(self._registry.keys()))
            raise KeyError(f"Loss function '{name}' not found. Available losses: {available_losses}")
        
        return self._registry[name](**kwargs)
    
    @property
    def available(self):
        """
        List all available loss functions.
        
        Returns:
            list: Names of registered loss functions
        """
        return list(self._registry.keys())

# Global singleton instance
LOSS_REGISTRY = LossFunctionRegistry()

def register_loss(name):
    """
    Decorator to register a custom loss function.
    
    Args:
        name (str): Name to register the loss function under
    
    Returns:
        decorator function
    """
    def decorator(loss_fn):
        LOSS_REGISTRY.register(name, loss_fn)
        return loss_fn
    return decorator

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
        return torch.matmul(a, b.transpose(0, 1))

    # Ensure `a` is of shape (batch_size, 1, vector_dim)
    if len(a.shape) == 2:
        a = a.unsqueeze(1)
    
    # Compute batched dot product, result shape: (batch_size, 1, group_size)
    return torch.bmm(b, a.transpose(1, 2)).squeeze()

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