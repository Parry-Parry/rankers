"""Rankers: A neural information retrieval framework.

This package provides modular implementations of various neural ranking models including
dot-product (bi-encoder), concatenation (cross-encoder), sparse, and sequence-to-sequence
architectures. It integrates with HuggingFace Transformers and PyTerrier for flexible
information retrieval pipelines.

The package uses lazy loading for optional dependencies (torch, pyterrier, flax) to ensure
minimal installation requirements while supporting advanced features when available.

Examples:
    Basic usage with a dot-product model::

        from rankers.modelling import Dot

        model = Dot.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        scores = model.score(["query"], ["document"])

    Training a custom ranker::

        from rankers.modelling import Dot
        from rankers.train import RankerTrainer, RankerTrainingArguments
        from rankers.datasets import TrainingDataset

        model = Dot.from_pretrained("bert-base-uncased")
        dataset = TrainingDataset.from_json("train.json")
        args = RankerTrainingArguments(output_dir="./output")
        trainer = RankerTrainer(model=model, args=args, train_dataset=dataset)
        trainer.train()
"""

import os
from typing import TYPE_CHECKING

from transformers.utils import _LazyModule


from ._optional import is_torch_available, is_pyterrier_available

__version__ = "0.0.6"

if os.getenv("RANKERS_EAGER_IMPORTS") == "1":
    # Import the subpackages you normally expose lazily
    from . import (
        modelling,
        datasets,
        train,
        pyterrier,
        _util,
    )  # adjust to your actual public surface

    # Advertise a concrete __all__ (must be a list of strings)
    __all__ = ["modelling", "datasets", "train", "pyterrier", "_util"]

_import_structure = {
    "train.trainer": ["RankerTrainer"],
    "train.loss": ["LOSS_REGISTRY, register_loss"],
    "train.data_arguments": ["RankerDataArguments"],
    "train.model_arguments": [
        "RankerModelArguments",
        "RankerDotArguments",
        "RankerCatArguments",
    ],
    "_util": [
        "seed_everything",
        "not_tested",
        "load_json",
        "save_json",
        "read_trec",
        "write_trec",
    ],
    "train.training_arguments": ["RankerTrainingArguments"],
    "datasets": [
        "Corpus",
        "TrainingDataset",
        "EvaluationDataset",
        "DotDataCollator",
        "CatDataCollator",
    ],
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

if is_pyterrier_available():
    _import_structure["pyterrier.dot"] = ["DotTransformer"]
    _import_structure["pyterrier.sparse"] = ["SparseTransformer"]
    _import_structure["pyterrier.cat"] = ["CatTransformer"]

if TYPE_CHECKING:
    from ._util import (
        seed_everything,
        not_tested,
        load_json,
        save_json,
        read_trec,
        write_trec,
    )
    from .datasets import (
        Corpus,
        TrainingDataset,
        EvaluationDataset,
        DotDataCollator,
        CatDataCollator,
    )

    if is_torch_available():
        from .train.loss import (
            LOSS_REGISTRY as LOSS_REGISTRY,
            register_loss as register_loss,
        )
        from .train.loss.torch import BaseLoss as BaseLoss
        from .train.loss.torch import *

        from .modelling.base import Ranker as Ranker
        from .modelling.cat import Cat as Cat
        from .modelling.dot import Dot as Dot
        from .modelling.dot import DotConfig as DotConfig
        from .modelling.sparse import Sparse as Sparse
        from .modelling.bge import BGE as BGE
        from .modelling.seq2seq import Seq2Seq as Seq2Seq

    if is_pyterrier_available():
        from .pyterrier.dot import DotTransformer as DotTransformer
        from .pyterrier.sparse import SparseTransformer as SparseTransformer
        from .pyterrier.cat import CatTransformer as CatTransformer

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
