import json
from rankers._optional import is_lz4_available
from fire import Fire
import pickle

if not is_lz4_available():
    raise ImportError("lz4 is not available. Please install it via `pip install lz4`.")

import lz4


def compress_record(record):
    """Compress a single JSON record using LZ4 and pickle."""
    pickled_data = pickle.dumps(record)
    compressed_data = lz4.frame.compress(pickled_data)
    return compressed_data


def jsonl_to_lz4(jsonl_file, output_file):
    """Convert a JSONL file to an LZ4-compressed pickle format line by line."""
    with open(jsonl_file, "r", encoding="utf-8") as infile, open(output_file, "wb") as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)  # Parse JSON
                compressed_record = compress_record(record)
                outfile.write(compressed_record + b'\n')  # Store each compressed line
            except json.JSONDecodeError as e:
                print(f"Skipping malformed line: {e}")


if __name__ == "__main__":
    Fire(jsonl_to_lz4)
