__version__ = "0.0.1"

from .train import loss as loss
from .train.trainer import ConstrastTrainer
from .train.arguments import ContrastArguments
from .train.callback import *
from . import datasets as datasets
from .inference import models as transformer

def seed_everything(seed=42):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True