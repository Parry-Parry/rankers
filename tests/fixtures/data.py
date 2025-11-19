"""Fixtures for generating synthetic test data."""

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


def create_synthetic_jsonl(
    num_queries: int = 10,
    num_docs_per_query: int = 5,
    query_prefix: str = "q",
    doc_prefix: str = "d",
) -> Tuple[str, List[Dict]]:
    """Create a synthetic JSONL training file for testing.

    Args:
        num_queries: Number of queries to generate.
        num_docs_per_query: Number of documents per query.
        query_prefix: Prefix for query IDs.
        doc_prefix: Prefix for document IDs.

    Returns:
        Tuple of (filepath, list of records).
    """
    records = []
    for qid in range(num_queries):
        query_id = f"{query_prefix}{qid}"
        positive_id = f"{doc_prefix}{qid * num_docs_per_query}"
        negative_ids = [
            f"{doc_prefix}{qid * num_docs_per_query + i + 1}"
            for i in range(num_docs_per_query - 1)
        ]

        record = {
            "query_id": query_id,
            "doc_id_a": positive_id,
            "doc_id_b": negative_ids,
        }
        records.append(record)

    # Write to temporary file
    tmp_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    )
    for record in records:
        tmp_file.write(json.dumps(record) + "\n")
    tmp_file.close()

    return tmp_file.name, records


def create_synthetic_trec(
    num_queries: int = 10,
    num_docs_per_query: int = 5,
    query_prefix: str = "q",
    doc_prefix: str = "d",
) -> Tuple[str, pd.DataFrame]:
    """Create a synthetic TREC format file for testing.

    Args:
        num_queries: Number of queries.
        num_docs_per_query: Number of documents per query.
        query_prefix: Prefix for query IDs.
        doc_prefix: Prefix for document IDs.

    Returns:
        Tuple of (filepath, DataFrame).
    """
    rows = []
    rank = 1
    for qid in range(num_queries):
        query_id = f"{query_prefix}{qid}"
        for did in range(num_docs_per_query):
            doc_id = f"{doc_prefix}{qid * num_docs_per_query + did}"
            score = 10.0 - did  # Decreasing relevance scores
            rows.append(
                {
                    "qid": query_id,
                    "Q0": "Q0",
                    "docno": doc_id,
                    "rank": rank,
                    "score": score,
                    "run_id": "test_run",
                }
            )
            rank += 1

    df = pd.DataFrame(rows)

    # Write to temporary file
    tmp_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".trec", delete=False
    )
    for _, row in df.iterrows():
        line = (
            f"{row['qid']} {row['Q0']} {row['docno']} {row['rank']} "
            f"{row['score']} {row['run_id']}\n"
        )
        tmp_file.write(line)
    tmp_file.close()

    return tmp_file.name, df


def create_synthetic_corpus(
    num_docs: int = 50,
    num_queries: int = 10,
    doc_prefix: str = "d",
    query_prefix: str = "q",
) -> Dict[str, Dict[str, str]]:
    """Create a synthetic corpus with documents and queries.

    Args:
        num_docs: Number of documents.
        num_queries: Number of queries.
        doc_prefix: Prefix for document IDs.
        query_prefix: Prefix for query IDs.

    Returns:
        Dictionary with 'documents' and 'queries' keys.
    """
    documents = {
        f"{doc_prefix}{i}": f"Document {i} text content about topic "
        f"{i % num_queries}"
        for i in range(num_docs)
    }

    queries = {
        f"{query_prefix}{i}": f"Query {i} about topic {i}" for i in range(num_queries)
    }

    return {"documents": documents, "queries": queries}


def create_synthetic_qrels(
    num_queries: int = 10,
    query_prefix: str = "q",
    doc_prefix: str = "d",
) -> pd.DataFrame:
    """Create a synthetic qrels DataFrame.

    Args:
        num_queries: Number of queries.
        query_prefix: Prefix for query IDs.
        doc_prefix: Prefix for document IDs.

    Returns:
        DataFrame with columns [query_id, doc_id, relevance].
    """
    rows = []
    for qid in range(num_queries):
        # Each query has 1-2 relevant documents
        num_relevant = 1 + (qid % 2)
        for rel_id in range(num_relevant):
            doc_id = f"{doc_prefix}{qid * 5 + rel_id}"
            rows.append(
                {
                    "query_id": f"{query_prefix}{qid}",
                    "doc_id": doc_id,
                    "relevance": 2 - rel_id,  # 2 for first, 1 for second
                }
            )

    return pd.DataFrame(rows)


def cleanup_temp_files(filepaths: List[str]) -> None:
    """Clean up temporary test files.

    Args:
        filepaths: List of file paths to remove.
    """
    for filepath in filepaths:
        Path(filepath).unlink(missing_ok=True)
