def is_torch_available():
    try:
        import torch

        return True
    except ImportError:
        return False


def is_pyterrier_available():
    try:
        import pyterrier as pt

        return True
    except ImportError:
        return False

def is_ir_datasets_available():
    try:
        import ir_datasets

        return True
    except ImportError:
        return False

def is_tira_available():
    try:
        import tira

        return True
    except ImportError:
        return False


def is_flax_available():
    try:
        import flax

        return True
    except ImportError:
        return False


def is_optax_available():
    try:
        import optax

        return True
    except ImportError:
        return False


def is_orbax_checkpoint_available():
    try:
        import orbax.checkpoint as checkpoint

        return True
    except ImportError:
        return False
