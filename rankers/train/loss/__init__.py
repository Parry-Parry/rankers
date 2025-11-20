"""Loss functions for training neural ranking models.

This module provides an extensible framework for loss functions used in training neural rankers.
It includes a registry system for dynamically registering and retrieving loss functions, along
with implementations of various pointwise, pairwise, and listwise loss functions.

Key Components:
    - **BaseLoss**: Abstract base class for all loss functions
    - **LOSS_REGISTRY**: Global registry for loss function lookup
    - **register_loss**: Decorator for registering custom loss functions
    - **Pointwise losses**: MSE, cross-entropy for point-based scoring
    - **Pairwise losses**: Margin-based losses for ranking pairs
    - **Listwise losses**: Losses operating on entire ranked lists

The registry pattern allows for flexible loss function selection via string names
in training configurations.

Examples:
    Using a registered loss function::

        from rankers.train.loss import LOSS_REGISTRY

        loss_fn = LOSS_REGISTRY.get("margeMSE")
        loss = loss_fn(scores, labels)

    Registering a custom loss::

        from rankers.train.loss import register_loss, BaseLoss

        @register_loss("my_loss")
        class MyCustomLoss(BaseLoss):
            def forward(self, scores, labels):
                return custom_computation(scores, labels)
"""

from typing import TYPE_CHECKING

from transformers.utils import _LazyModule

from ..._optional import is_flax_available, is_torch_available

_import_structure = {
    "util": ["LossFunctionRegistry", "register_loss", "LOSS_REGISTRY"],
}


if is_torch_available():
    from .torch import __all__ as _torch_all

    _import_structure["torch"] = ["BaseLoss", *_torch_all]

if TYPE_CHECKING:
    from .util import (
        LOSS_REGISTRY as LOSS_REGISTRY,
    )
    from .util import (
        LossFunctionRegistry as LossFunctionRegistry,
    )
    from .util import (
        register_loss as register_loss,
    )

    if is_torch_available():
        from . import torch as torch
        from .torch import BaseLoss as BaseLoss

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
