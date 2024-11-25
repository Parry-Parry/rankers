__version__ = "0.0.6"


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
from .datasets import *

from .modelling.base import Ranker
from .modelling.dot import Dot, DotConfig
from .modelling.sparse import Sparse
from .modelling.cat import Cat
from .modelling.seq2seq import Seq2Seq
from .modelling.bge import BGE

from .pyterrier.dot import DotTransformer
from .pyterrier.sparse import SparseTransformer
from .pyterrier.cat import CatTransformer