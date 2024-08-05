from ... import is_torch_available, is_flax_available
from transformers.utils import _LazyModule, OptionalDependencyNotAvailable
from typing import TYPE_CHECKING

_import_structure = {}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure['cat'] = [
        'Cat',
        'CatTransformer',
    ]
    _import_structure['dot'] = [
        'Dot',
        'DotTransformer',
        'DotConfig',
    ]
    _import_structure['seq2seq'] = [
        'Seq2Seq',
        'Seq2SeqTransformer',
    ]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure['flaxcat'] = [
        'FlaxCat',
        'FlaxCatTransformer',
    ]
    _import_structure['flaxdot'] = [
        'FlaxDot',
        'FlaxDotTransformer',
        'FlaxDotConfig',
    ]
    _import_structure['flaxseq2seq'] = [
        'FlaxSeq2Seq',
        'FlaxSeq2SeqTransformer',
    ]

if TYPE_CHECKING:
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .cat import Cat, CatTransformer
        from .dot import Dot, DotTransformer, DotConfig
        from .seq2seq import Seq2Seq, Seq2SeqTransformer
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .flaxcat import FlaxCat, FlaxCatTransformer
        from .flaxdot import FlaxDot, FlaxDotTransformer, FlaxDotConfig
        from .flaxseq2seq import FlaxSeq2Seq, FlaxSeq2SeqTransformer
else:
    import sys 
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
    