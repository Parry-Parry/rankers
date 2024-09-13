__version__ = "0.0.2"

from .train import loss as loss
from .train.trainer import ContrastTrainer
from .train.arguments import ContrastArguments
from .datasets import *
from .modelling import *

def is_torch_available():
    try:
        import torch
        return True
    except ImportError:
        return False

def is_flax_available():
    try:
        import flax
        return True
    except ImportError:
        return False

def seed_everything(seed=42):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True