import functools
from ..._optional import is_torch_available, is_flax_available
from transformers.utils import _LazyModule, OptionalDependencyNotAvailable
from typing import TYPE_CHECKING


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
        if not hasattr(self, "_registry"):
            # Dictionary to store registered loss functions
            self._registry = {}

            # Automatically register built-in PyTorch loss functions
            self._register_builtin_losses()

    def _register_builtin_losses(self):
        """
        Automatically register common PyTorch loss functions.
        """
        import torch.nn as nn

        builtin_losses = {
            # Basic losses
            "mse": nn.MSELoss,
            "l1": nn.L1Loss,
            "cross_entropy": nn.CrossEntropyLoss,
            "nll": nn.NLLLoss,
            "binary_cross_entropy": nn.BCELoss,
            "binary_cross_entropy_with_logits": nn.BCEWithLogitsLoss,
            # Reduction variant losses
            "mse_sum": functools.partial(nn.MSELoss, reduction="sum"),
            "mse_none": functools.partial(nn.MSELoss, reduction="none"),
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
            raise KeyError(
                f"Loss function '{name}' not found. Available losses: {available_losses}"
            )

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


_import_structure = {}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    from .torch import __all__ as __all_torch

    _import_structure["loss"] = [
        "BaseLoss",
        "LossFunctionRegistry",
        "LOSS_REGISTRY",
        "register_loss",
        "FLOPS_regularization",
        "L1_regularization",
        "reduce",
        "normalize",
        "residual",
        "maxsim",
        "dot_product",
        "batched_dot_product",
        "cross_dot_product",
        *__all_torch,
    ]
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    from .flax import __all__ as __all_flax

    _import_structure["flax_loss"] = [
        "FlaxBaseLoss",
        *__all_flax,
    ]

if TYPE_CHECKING:
    if is_torch_available():
        from .torch import BaseLoss as BaseLoss
        from .torch import *
    if is_flax_available():
        from .flax import FlaxBaseLoss as FlaxBaseLoss
        from .flax import *
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
