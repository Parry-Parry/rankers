from typing import TYPE_CHECKING
from ..._optional import is_pyarrow_available
from transformers.utils import _LazyModule

_import_structure = {
    "jsonl": ["JSONLTrainingData"],
}


if is_pyarrow_available():
    _import_structure["parquet"] = ["ParquetTrainingData"]

if TYPE_CHECKING:
    from . import jsonl as jsonl
    from .jsonl import JSONLTrainingData as JSONLTrainingData

    if is_pyarrow_available():
        from . import parquet as parquet
        from .parquet import ParquetTrainingData as ParquetTrainingData
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
