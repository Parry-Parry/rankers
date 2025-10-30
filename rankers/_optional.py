"""Optional dependency checkers.

This module provides utility functions to check for the availability of optional dependencies.
The rankers package uses these functions to conditionally import and use features that depend
on external libraries, allowing for a minimal base installation.

The lazy import pattern ensures that users only need to install dependencies for features they
actually use (e.g., PyTerrier integration, Flax support).

Functions:
    is_torch_available: Check if PyTorch is installed
    is_pyterrier_available: Check if PyTerrier is installed
    is_ir_datasets_available: Check if ir_datasets is installed
    is_ir_measures_available: Check if ir_measures is installed
    is_tira_available: Check if TIRA is installed
    is_flax_available: Check if Flax (JAX) is installed
    is_optax_available: Check if Optax optimizer is installed
    is_orbax_checkpoint_available: Check if Orbax checkpointing is installed

Examples:
    Using dependency checks for conditional imports::

        from rankers._optional import is_torch_available

        if is_torch_available():
            from rankers.modelling import Dot
            model = Dot.from_pretrained("model-name")
        else:
            print("PyTorch not available")
"""


def is_torch_available():
    """Check if PyTorch is available.

    Returns:
        bool: True if torch can be imported, False otherwise.
    """
    try:
        import torch

        return True
    except ImportError:
        return False


def is_pyterrier_available():
    """Check if PyTerrier is available.

    Returns:
        bool: True if pyterrier can be imported, False otherwise.
    """
    try:
        import pyterrier as pt

        return True
    except ImportError:
        return False


def is_ir_datasets_available():
    """Check if ir_datasets is available.

    Returns:
        bool: True if ir_datasets can be imported, False otherwise.
    """
    try:
        import ir_datasets

        return True
    except ImportError:
        return False


def is_ir_measures_available():
    """Check if ir_measures is available.

    Returns:
        bool: True if ir_measures can be imported, False otherwise.
    """
    try:
        import ir_measures

        return True
    except ImportError:
        return False


def is_tira_available():
    """Check if TIRA is available.

    Returns:
        bool: True if tira can be imported, False otherwise.
    """
    try:
        import tira

        return True
    except ImportError:
        return False


def is_flax_available():
    """Check if Flax (JAX framework) is available.

    Returns:
        bool: True if flax can be imported, False otherwise.
    """
    try:
        import flax

        return True
    except ImportError:
        return False


def is_optax_available():
    """Check if Optax optimizer library is available.

    Returns:
        bool: True if optax can be imported, False otherwise.
    """
    try:
        import optax

        return True
    except ImportError:
        return False


def is_orbax_checkpoint_available():
    """Check if Orbax checkpointing library is available.

    Returns:
        bool: True if orbax.checkpoint can be imported, False otherwise.
    """
    try:
        import orbax.checkpoint as checkpoint

        return True
    except ImportError:
        return False
