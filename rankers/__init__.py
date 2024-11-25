from ._optional import is_torch_available, is_flax_availible, is_pyterrier_available
import functools
from transformers.utils import _LazyModule
from typing import TYPE_CHECKING

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


_import_structure = {
    "train.trainer": ["RankerTrainer"],
    "train.loss": ["LOSS_REGISTRY, register_loss"],
    "train.data_arguments": ["RankerDataArguments"],
    "train.model_arguments": [
        "RankerModelArguments",
        "RankerDotArguments",
        "RankerCatArguments",
    ],
    "train.training_arguments": ["RankerTrainingArguments"],
    "datasets": [""],
    "modelling.cat": [],
    "modelling.dot": [],
    "modelling.sparse": [],
    "modelling.bge": [],
    "modelling.base": [],
    "modelling.seq2seq": [],
}


if is_torch_available():
    _import_structure["train.loss.torch"] = ["BaseLoss"]
    _import_structure["modelling.base"].extend(["Ranker"])
    _import_structure["modelling.cat"].extend(["Cat"])
    _import_structure["modelling.dot"].extend(["Dot", "DotConfig"])
    _import_structure["modelling.sparse"].extend(["Sparse"])
    _import_structure["modelling.bge"].extend(["BGE"])
    _import_structure["modelling.seq2seq"].extend(["Seq2Seq"])

if is_flax_availible():
    _import_structure["modelling.flax"].extend(["FlaxBaseLoss"])
    _import_structure["modelling.cat"].extend(["FlaxCat"])
    _import_structure["modelling.dot"].extend(["FlaxDot", "FlaxDotConfig"])
    _import_structure["modelling.sparse"].extend(["FlaxSparse"])
    _import_structure["modelling.bge"].extend(["FlaxBGE"])
    _import_structure["modelling.seq2seq"].extend(["FlaxSeq2Seq"])

if is_pyterrier_available():
    _import_structure["pyterrier.dot"] = ["DotTransformer"]
    _import_structure["pyterrier.sparse"] = ["SparseTransformer"]
    _import_structure["pyterrier.cat"] = ["CatTransformer"]

if TYPE_CHECKING:
    if is_torch_available():
        from .train.loss.torch import BaseLoss as BaseLoss
        from .train.loss.torch import *

        from .modelling.base import Ranker as Ranker
        from .modelling.cat import Cat as Cat
        from .modelling.dot import Dot as Dot
        from .modelling.dot import DotConfig as DotConfig
        from .modelling.sparse import Sparse as Sparse
        from .modelling.bge import BGE as BGE
        from .modelling.seq2seq import Seq2Seq as Seq2Seq

    if is_flax_available():
        from .train.loss.flax import FlaxBaseLoss as FlaxBaseLoss
        from .train.loss.flax import *

        from .modelling.cat import FlaxCat as FlaxCat
        from .modelling.dot import FlaxDot as FlaxDot

    if is_pyterrier_available():
        from .pyterrier.dot import DotTransformer as DotTransformer
        from .pyterrier.sparse import SparseTransformer as SparseTransformer
        from .pyterrier.cat import CatTransformer as CatTransformer

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
