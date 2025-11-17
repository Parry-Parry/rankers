"""Dataset classes and data collators for ranking tasks.

This module provides dataset classes for training, validation, and testing of neural rankers,
along with specialized data collators for different model architectures.

Key Components:
    - **Corpus**: Text corpus management with lazy loading capabilities
    - **TrainingDataset**: Dataset for training ranking models with query-document pairs
    - **TestDataset**: Dataset for evaluation with TREC-style ranking data
    - **Data Collators**: Specialized collators for different architectures (Dot, Cat, Sparse)

The module supports both PyTorch and Flax frameworks through conditional imports.

Examples:
    Loading a training dataset::

        from rankers.datasets import TrainingDataset

        dataset = TrainingDataset.from_json("train.json")

    Using with a data collator::

        from rankers.datasets import DotDataCollator

        collator = DotDataCollator(tokenizer=tokenizer)
        batch = collator(dataset[:4])
"""

from .._optional import is_torch_available, is_flax_available
from transformers.utils import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "corpus": ["Corpus"],
    "dataset": ["TrainingDataset", "TestDataset", "ValidationDataset"],
}

if is_torch_available():
    _import_structure["loader"] = [
        "DotDataCollator",
        "CatDataCollator",
        "PairDataCollator",
        "PromptDataCollator",
        "ListwisePromptDataCollator",
    ]

if is_flax_available():
    _import_structure["flaxloader"] = [
        "FlaxDotDataCollator",
        "FlaxCatDataCollator",
        "FlaxPairDataCollator",
    ]

if TYPE_CHECKING:
    if is_torch_available():
        from .loader import (
            DotDataCollator as DotDataCollator,
            CatDataCollator as CatDataCollator,
            PairDataCollator as PairDataCollator,
            PromptDataCollator as PromptDataCollator,
            ListWisePromptDataCollator as ListWisePromptDataCollator,
        )
    if is_flax_available():
        from .flaxloader import (
            FlaxDotDataCollator as FlaxDotDataCollator,
            FlaxCatDataCollator as FlaxCatDataCollator,
            FlaxPairDataCollator as FlaxPairDataCollator,
            FlaxPromptDataCollator as FlaxPromptDataCollator,
        )

    from .corpus import Corpus as Corpus
    from .dataset import (
        TrainingDataset as TrainingDataset,
        TestDataset as TestDataset,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
