
from ... import is_torch_available, is_flax_available
from transformers.utils import _LazyModule, OptionalDependencyNotAvailable
from typing import TYPE_CHECKING

_import_structure = {
    'dataset' : [
        'TrainingDataset',
        'EvaluationDataset'
    ],
    'corpus' : [
        'Corpus'
    ],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure['loader'] = [
        'DotDataCollator',
        'CatDataCollator',
        'PairDataCollator',
        'PromptDataCollator',
        'PairPromptDataCollator',
    ]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure['flaxloader'] = [
        'FlaxDotDataCollator',
        'FlaxCatDataCollator',
        'FlaxPairDataCollator',
        'FlaxPromptDataCollator',
        'FlaxPairPromptDataCollator',
    ]

from .dataset import TrainingDataset, EvaluationDataset
from .corpus import Corpus

if TYPE_CHECKING:
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .loader import DotDataCollator, CatDataCollator, PairDataCollator, PromptDataCollator, PairPromptDataCollator
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .flaxloader import FlaxDotDataCollator, FlaxCatDataCollator, FlaxPairDataCollator, FlaxPromptDataCollator, FlaxPairPromptDataCollator
else:
    import sys 
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)