__version__ = "0.0.5"

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

from .train import loss as loss
from .train.trainer import RankerTrainer
from .train.training_arguments import RankerTrainingArguments
from .train.data_arguments import RankerDataArguments
from .train.model_arguments import RankerDotArguments, RankerCatArguments, RankerModelArguments
from .datasets import *
from .modelling import *