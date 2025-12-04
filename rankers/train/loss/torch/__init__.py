"""Base loss classes for neural ranking.

This module provides the foundational loss class hierarchy and common regularization
losses. All ranking losses inherit from BaseLoss and can be registered in the
global loss registry for easy access.
"""

from abc import abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

from ..util import register_loss


class BaseLoss(nn.Module):
    """Abstract base class for all ranking loss functions.

    Provides common functionality including reduction strategies and a standard
    interface for all loss implementations. Subclasses must implement the forward method.

    Args:
        reduction (str, optional): Reduction method for loss values. Options:
            "mean", "sum", "none". Defaults to "mean".

    Attributes:
        name (str): Loss function identifier.
        reduction (str): Reduction strategy.

    Examples:
        Creating a custom loss::

            from rankers.train.loss.torch import BaseLoss, register_loss

            @register_loss("my_custom_loss")
            class MyLoss(BaseLoss):
                def forward(self, scores, labels):
                    return torch.nn.functional.mse_loss(scores, labels)

    Note:
        Use the @register_loss decorator to make losses available via string
        keys in LOSS_REGISTRY.
    """

    name = "base"

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def _reduce(self, a: torch.Tensor):
        """Apply reduction to tensor.

        Args:
            a (torch.Tensor): Tensor to reduce.

        Returns:
            torch.Tensor: Reduced tensor.
        """
        return reduce(a, self.reduction)

    def forward(self, *args, **kwargs):
        """Compute the loss.

        Args:
            *args: Loss-specific arguments (typically scores and labels).
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Computed loss value.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def __repr__(self):
        return self.name


class RegularizationLoss(BaseLoss):
    """Base class for regularization losses with warmup scheduling.

    Implements regularization for query and document representations with
    quadratic warmup scheduling. Commonly used for sparse models (e.g., SPLADE)
    to control representation sparsity.

    Args:
        q_weight (float, optional): Initial query regularization weight. Defaults to 0.08.
        d_weight (float, optional): Initial document regularization weight. Defaults to 0.1.
        t (int, optional): Initial timestep. Defaults to 0.
        warmup_steps (int, optional): Warmup steps. Defaults to 1000.
        reduction (str, optional): Reduction method. Defaults to "mean".

    Attributes:
        q_weight (float): Current query regularization weight.
        d_weight (float): Current document regularization weight.
        t (int): Current training step.
        T (int): Total warmup steps.

    Examples:
        Using with sparse models::

            from rankers.train.loss.torch import FLOPSLoss

            reg_loss = FLOPSLoss(
                q_weight=0.001,
                d_weight=0.001,
                warmup_steps=5000  # Warmup over 5000 steps
            )

    Note:
        Weights increase quadratically: weight * (t/warmup_steps)^2 until t >= warmup_steps.
        Call step() after each training step to update weights.
    """

    def __init__(
        self, q_weight=0.08, d_weight=0.1, t=0, warmup_steps=1000, reduction: str = "mean"
    ) -> None:
        super().__init__(reduction)
        self.q_weight = q_weight
        self.d_weight = d_weight
        self.t = t
        self.T = warmup_steps

    def step_q(self):
        """Update query weight for one training step."""
        if self.t >= self.T:
            pass
        else:
            self.t += 1
            self.q_weight = self.q_weight * (self.t / self.T) ** 2

    def step_d(self):
        """Update document weight for one training step."""
        if self.t >= self.T:
            pass
        else:
            self.t += 1
            self.d_weight = self.d_weight * (self.t / self.T) ** 2

    def step(self):
        """Update both query and document weights for one training step."""
        if self.t >= self.T:
            pass
        else:
            self.t += 1
            self.q_weight = self.q_weight * (self.t / self.T) ** 2
            self.d_weight = self.d_weight * (self.t / self.T) ** 2

    @abstractmethod
    def reg(self, reps, weight=0):
        """Compute regularization for representations.

        Args:
            reps (torch.Tensor): Representations to regularize.
            weight (float, optional): Regularization weight. Defaults to 0.

        Returns:
            torch.Tensor: Regularization loss value.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def forward(self, query_hidden_states, text_hidden_states, **kwargs):
        """Compute combined regularization loss.

        Args:
            query_hidden_states (torch.Tensor): Query representations.
            text_hidden_states (torch.Tensor): Document representations.
            **kwargs: Additional arguments (unused).

        Returns:
            tuple: Contains:
                - loss (torch.Tensor): Total regularization loss.
                - to_log (dict): Metrics dictionary with q_reg, d_reg, and sparsity stats.
        """
        q_reg = self.reg(query_hidden_states, self.q_weight)
        d_reg = self.reg(text_hidden_states, self.d_weight)
        self.step()
        to_log = {"q_reg": q_reg, "d_reg": d_reg}
        to_log["q_num_non_zeo"] = num_non_zero(query_hidden_states)
        to_log["d_num_non_zeo"] = num_non_zero(text_hidden_states)
        return q_reg + d_reg, to_log


@register_loss("flops_reg")
class FLOPSLoss(RegularizationLoss):
    """FLOPS-based regularization loss for sparse models.

    Penalizes the computational cost (FLOPs) of sparse representations by
    encouraging smaller L1 norms. Commonly used in SPLADE models.

    The regularization is computed as: sum((abs(reps).mean(dim=0))^2) * weight

    Args:
        q_weight (float, optional): Query regularization weight. Defaults to 0.08.
        d_weight (float, optional): Document regularization weight. Defaults to 0.1.
        t (int, optional): Initial timestep. Defaults to 0.
        T (int, optional): Warmup steps. Defaults to 1000.
        reduction (str, optional): Reduction method. Defaults to "mean".

    Examples:
        Training a sparse model with FLOPS regularization::

            from rankers.train import RankerTrainer, RankerTrainingArguments

            args = RankerTrainingArguments(
                output_dir="./output",
                regularization="flops_reg",
                q_regularization_weight=0.001,
                d_regularization_weight=0.001,
                regularization_warmup_steps=5000
            )
    """

    def __init__(
        self, q_weight=0.08, d_weight=0.1, t=0, warmup_steps=1000, reduction: str = "mean"
    ) -> None:
        super().__init__(q_weight, d_weight, t, warmup_steps, reduction)

    def reg(self, reps, weight=0):
        """Compute FLOPS regularization.

        Args:
            reps (torch.Tensor): Sparse representations.
            weight (float, optional): Regularization weight. Defaults to 0.

        Returns:
            torch.Tensor: FLOPS regularization loss.
        """
        return (torch.abs(reps).mean(dim=0) ** 2).sum() * weight


@register_loss("l1_reg")
class L1Loss(BaseLoss):
    """L1 regularization loss.

    Applies L1 (absolute value) regularization to encourage sparsity in
    representations.

    Args:
        reduction (str, optional): Reduction method. Defaults to "mean".

    Examples:
        Using L1 regularization::

            reg_loss = L1Loss()
            loss_value = reg_loss.reg(representations, weight=0.01)
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(reduction)

    def reg(self, reps, weight=0):
        """Compute L1 regularization.

        Args:
            reps (torch.Tensor): Representations to regularize.
            weight (float, optional): Regularization weight. Defaults to 0.

        Returns:
            torch.Tensor: L1 regularization loss.
        """
        return torch.abs(reps).sum(dim=1).mean() * weight


class CompoundLoss(BaseLoss):
    def __init__(self, losses: list, alphas: list = None):
        super().__init__()
        self.losses = losses
        self.alphas = alphas if alphas is not None else [1] * len(losses)

    def forward(
        self,
        pred,
        labels=None,
        query_hidden_states=None,
        text_hidden_states=None,
        **kwargs,
    ):
        total = 0.0
        to_log = {}
        for loss, alpha in zip(self.losses, self.alphas):
            loss_val = loss(
                pred=pred,
                labels=labels,
                query_hidden_states=query_hidden_states,
                text_hidden_states=text_hidden_states,
                **kwargs,
            )
            if len(loss_val) == 2:
                loss_val, _to_log = loss_val
                for k, v in _to_log.items():
                    to_log[k] = v.detach().cpu().item()
            to_log[loss.name] = loss_val.detach().cpu().item()
            loss_val = loss_val * alpha
            total += loss_val
        return total, to_log


def reduce(a: torch.Tensor, reduction: str):
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
    if reduction == "none":
        return a
    if reduction == "mean":
        return a.mean()
    if reduction == "sum":
        return a.sum()
    if reduction == "batchmean":
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


def residual(a: Tensor):
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
    if a.size(1) == 1:
        return a
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
    return dot_product(a, b)


def maxsim(a: Tensor, b: Tensor, a_mask: Tensor, temperature: float = 1.0):
    scores = torch.einsum("qin,pjn->qipj", a, b)
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
from . import pairwise as pairwise
from . import pointwise as pointwise
from .listwise import *
from .pairwise import *
from .pointwise import *

__all__ = [*listwise.__all__, *pointwise.__all__, *pairwise.__all__]
