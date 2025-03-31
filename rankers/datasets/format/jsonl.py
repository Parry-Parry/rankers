import json
from functools import cached_property
from typing import List


class JSONLTrainingData:
    def __init__(self, file):
        self.file = file

        self.__post_init__()

    def __post_init__(self):
        self.line_offsets = self._get_line_offsets()

    def validate_schema(self, schema: List[str]):
        first_entry = self.get_first_entry()
        for key in schema:
            if key not in first_entry:
                raise ValueError(f"Key {key} not found in schema")

    def __len__(self):
        return len(self.line_offsets)

    def _get_line_offsets(self):
        """Store byte offsets for each line in an uncompressed JSONL file, skipping blank lines."""
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
            return json.loads(f.readline())

    @cached_property
    def first_entry(self):
        return self._get_line_by_index(0)

    def __get__(self, idx):
        return self._get_line_by_index(idx)


__all__ = ["JSONLTrainingData"]
