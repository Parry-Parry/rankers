import json
from fire import Fire
from rankers._optional import is_pyarrow_available

if not is_pyarrow_available():
    raise ImportError("PyArrow is not available. Please install it via `pip install pyarrow`.")

import pyarrow as pa
import pyarrow.parquet as pq


def jsonl_to_parquet(jsonl_path: str, parquet_path: str, compression: str = "snappy"):
    """Convert JSON Lines file to Parquet format using PyArrow."""
    keys = set()
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                keys.update(record.keys())
                records.append(record)

    table = pa.Table.from_pydict({key: [rec.get(key, None) for rec in records] for key in keys})
    pq.write_table(table, parquet_path, compression=compression)
    print(f"Converted {jsonl_path} to {parquet_path} with {len(records)} records.")


if __name__ == "__main__":
    Fire(jsonl_to_parquet)
