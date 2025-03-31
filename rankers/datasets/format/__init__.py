from typing import TYPE_CHECKING
from ..._optional import is_pyarrow_available, is_lmdb_available, is_lz4_available
from transformers.utils import _LazyModule

_import_structure = {
    "jsonl": ["JSONLTrainingData"],
}


if is_pyarrow_available():
    _import_structure["parquet"] = ["ParquetTrainingData"]

if is_lmdb_available():
    _import_structure["lmdb"] = ["LMDBTrainingData"]

if is_lz4_available():
    _import_structure["lz4"] = ["LZ4TrainingData"]

if TYPE_CHECKING:
    from . import jsonl as jsonl
    from .jsonl import JSONLTrainingData as JSONLTrainingData

    if is_pyarrow_available():
        from . import parquet as parquet
        from .parquet import ParquetTrainingData as ParquetTrainingData

    if is_lmdb_available():
        from . import lmdb as lmdb
        from .lmdb import LMDBTrainingData as LMDBTrainingData
    
    if is_lz4_available():
        from . import lz4 as lz4
        from .lz4 import LZ4TrainingData as LZ4TrainingData
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
