import lmdb
import json
from functools import cached_property
from typing import List


class LMDBTrainingData:
    def __init__(self, file: str):
        self.file = file

        self.__post_init__()

    def __post_init__(self):
        self.env = lmdb.open(self.file, readonly=True, lock=False)
        with self.env.begin() as txn:
            self.num_rows = sum(1 for _ in txn.cursor())

    def __len__(self):
        return self.num_rows

    def validate_schema(self, schema: List[str]):
        first_entry = self.first_entry
        for key in schema:
            if key not in first_entry:
                raise ValueError(f"Key {key} not found in schema")

    def _get_line_by_index(self, idx):
        e_idx = str(idx).encode()
        with self.env.begin() as txn:
            entry = txn.get(e_idx)
        if entry is None:
            raise IndexError(f"Index {idx} out of range")
        record = json.loads(entry)
        return record

    @cached_property
    def first_entry(self):
        return self._get_line_by_index(0)

    def __getitem__(self, idx):
        return self._get_line_by_index(idx)


__all__ = ["LMDBTrainingData"]
