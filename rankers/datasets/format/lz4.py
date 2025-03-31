from functools import cached_property
from typing import List
from pickle import loads
from lz4.frame import decompress


def load_record(row):
    d = decompress(row).decode()

    return loads(d)


class LZ4TrainingData:
    def __init__(self, file):
        self.file = file

        self.__post_init__()

    def __post_init__(self):
        self.line_offsets = self._get_line_offsets()

    def validate_schema(self, schema: List[str]):
        first_entry = self.first_entry
        for key in schema:
            if key not in first_entry:
                raise ValueError(f"Key {key} not found in schema")

    def __len__(self):
        return len(self.line_offsets)

    def _get_line_offsets(self):
        """Store byte offsets for each line"""
        offsets = []
        with open(self.training_dataset_file, "r", encoding="utf-8") as f:
            while True:
                offset = f.tell()
                line = f.readline().strip()
                if not line:
                    if f.tell() == f.seek(
                        0, 2
                    ):  # Check if we've reached the end of the file
                        break
                    continue
                offsets.append(offset)
        return offsets

    def _get_line_by_index(self, idx):
        """Retrieve a line by index, using offsets for uncompressed files."""
        with open(self.training_dataset_file, "r", encoding="utf-8") as f:
            f.seek(self.line_offsets[idx])
            return load_record(f.readline())

    @cached_property
    def first_entry(self):
        return self._get_line_by_index(0)

    def __getitem__(self, idx):
        return self._get_line_by_index(idx)


__all__ = ["LZ4TrainingData"]
