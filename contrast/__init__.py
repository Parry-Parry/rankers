import random
import numpy as np
import torch

__version__ = "0.0.1"
SEED = 42

# seed everything
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
