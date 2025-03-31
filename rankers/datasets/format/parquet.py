import pyarrow.parquet as pq
from functools import cached_property
from typing import List


class ParquetTrainingData:
    def __init__(self, file: str):
        self.file = file

        self.__post_init__()

    def __post_init__(self):
        self.parquet_file = pq.ParquetFile(self.file)
        self.num_rows = self.parquet_file.num_rows

    def __len__(self):
        return self.num_rows

    def validate_schema(self, schema: List[str]):
        _schema = self.table.schema
        for key in schema:
            if key not in _schema.names:
                raise ValueError(f"Key {key} not found in schema")

    def _get_line_by_index(self, idx):
        """Efficient row retrieval using PyArrow without full memory load."""
        batch = self.table.read_row_group(idx // 1000)  # Read in small chunks
        row = {col: batch[col].to_pylist()[idx % 1000] for col in batch.column_names}
        return row

    @cached_property
    def first_entry(self):
        return self._get_line_by_index(0)

    def __getitem__(self, idx):
        return self._get_line_by_index(idx)


__all__ = ["ParquetTrainingData"]
