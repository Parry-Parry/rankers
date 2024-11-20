from .._optional import is_torch_available, is_flax_available
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
    ]
    _import_structure['dot'] = [
        'Dot',
        'DotConfig',
    ]
    _import_structure['seq2seq'] = [
        'Seq2Seq',
    ]
    _import_structure['bge'] = [
        'BGE',
    ]
    _import_structure['sparse'] = [
        'Sparse',
    ]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure['flaxcat'] = [
        'FlaxCat',
    ]
    _import_structure['flaxdot'] = [
        'FlaxDot',
    ]
    _import_structure['flaxseq2seq'] = [
        'FlaxSeq2Seq',
    ]

if TYPE_CHECKING:
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .cat.cat import Cat
        from .dot.dot import Dot, DotConfig
        from .seq2seq.seq2seq import Seq2Seq
        from .bge.bge import BGE
        from sparse.sparse import Sparse
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .cat.flaxcat import FlaxCat
        from .dot.flaxdot import FlaxDot
        from .seq2seq.flaxseq2seq import FlaxSeq2Seq
else:
    import sys 
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
    