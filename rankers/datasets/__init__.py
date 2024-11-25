from .._optional import is_torch_available, is_flax_available
from transformers.utils import _LazyModule
from typing import TYPE_CHECKING

_import_structure = {
    "corpus": ["Corpus"],
    "dataset": ["TrainingDataset", "EvaluationDataset"],
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
        EvaluationDataset as EvaluationDataset,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
