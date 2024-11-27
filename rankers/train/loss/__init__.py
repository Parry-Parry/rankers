from typing import TYPE_CHECKING
from ..._optional import is_torch_available, is_flax_available
from transformers.utils import _LazyModule

_import_structure = {
    "util": ["LossFunctionRegistry", "register_loss", "LOSS_REGISTRY"],
}


if is_torch_available():
    from .torch import __all__ as _torch_all

    _import_structure["torch"] = ["BaseLoss", *_torch_all]
if is_flax_available():
    from .flax import __all__ as _flax_all

    _import_structure["flax"] = ["FlaxBaseLoss", *_flax_all]

if TYPE_CHECKING:
    from .util import LossFunctionRegistry as LossFunctionRegistry, register_loss as register_loss, LOSS_REGISTRY as LOSS_REGISTRY
    if is_torch_available():
        from .torch import BaseLoss as BaseLoss
        from . import torch as torch
    if is_flax_available():
        from .flax import FlaxBaseLoss as FlaxBaseLoss
        from . import flax as flax
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
