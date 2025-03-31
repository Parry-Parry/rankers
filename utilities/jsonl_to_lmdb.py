import json
from fire import Fire
from rankers._optional import is_lmdb_available

if not is_lmdb_available():
    raise ImportError("lmdb is not available. Please install it via `pip install lmdb`.")

import lmdb


def jsonl_to_lmdb(jsonl_path, lmdb_path):
    """Convert JSONL triplets to LMDB"""
    triplet_env = lmdb.open(lmdb_path, map_size=10**9)  # Triplets

    with triplet_env.begin(write=True) as txn:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                idx = str(i).encode()
                try:
                    _ = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Skipping line {i} due to JSON decoding error")
                    continue
                txn.put(idx, line.encode())

    return f"Converted {jsonl_path} to {lmdb_path} with {i+1} records."


if __name__ == "__main__":
    Fire(jsonl_to_lmdb)
