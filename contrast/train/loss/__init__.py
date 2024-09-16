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
    from .torch.listwise import __all__ as listwise_all
    from .torch.pointwise import __all__ as pointwise_all
    from .torch.pairwise import __all__ as pairwise_all
    _import_structure['loss'] = [
        'torchBaseLoss',
        *listwise_all,
        *pairwise_all,
        *pointwise_all
    ]
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    from .flax.listwise import __all__ as listwise_all
    from .flax.pointwise import __all__ as pointwise_all
    from .flax.pairwise import __all__ as pairwise_all
    _import_structure['Flaxloss'] = [
        'FlaxBaseLoss',
        *listwise_all,
        *pairwise_all,
        *pointwise_all
    ]

if TYPE_CHECKING:
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .torch.listwise import *
        from .torch.pointwise import *
        from .torch.pairwise import *
    
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .flax import FlaxBaseLoss
        from .flax.listwise import *
        from .flax.pointwise import *
        from .flax.pairwise import *
else:
    import sys 
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
    