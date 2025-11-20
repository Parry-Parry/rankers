"""Utility functions for the rankers package.

This module provides various utility functions for common tasks including:
- Random seed setting for reproducibility
- JSON/JSONL file I/O (with gzip support)
- TREC format file reading and writing
- Teacher score retrieval for knowledge distillation
- IR dataset initialization and manipulation

These utilities support core functionality across the rankers package, particularly
for data handling, reproducibility, and evaluation tasks.
"""

import logging
from collections import defaultdict
from typing import Any, Optional, Union

import ir_datasets as irds
import pandas as pd

logger = logging.getLogger(__name__)


def seed_everything(seed=42):
    """Set random seeds for reproducibility across all libraries.

    Sets seeds for Python's random module, NumPy, and PyTorch (if available).
    Also enables deterministic behavior in PyTorch's cuDNN backend.

    Args:
        seed (int, optional): Random seed value. Defaults to 42.

    Examples:
        Ensuring reproducible results::

            from rankers import seed_everything

            seed_everything(42)
            # Now all random operations will be deterministic
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def not_tested(cls):
    """Decorator to mark a class as not fully tested.

    This decorator wraps a class to log a warning whenever it is instantiated,
    indicating that the class has not been fully tested and should be used with caution.

    Args:
        cls (type): The class to wrap.

    Returns:
        type: A new class that inherits from the original and logs a warning on init.

    Examples:
        Marking an experimental class::

            @not_tested
            class ExperimentalRanker(Ranker):
                pass

            # Will log warning: "ExperimentalRanker is not fully tested"
            model = ExperimentalRanker()
    """

    class NewCls(cls):
        def __init__(self, *args, **kwargs):
            logger.warning(f"{cls.__name__} is not fully tested")
            super().__init__(*args, **kwargs)

    return NewCls


def _pivot(frame, negatives=None):
    new = []
    for row in frame.itertuples():
        new.append({"qid": row.query_id, "docno": row.doc_id_a, "pos": 1})
        if negatives:
            for doc in negatives[row.query_id]:
                new.append({"qid": row.query_id, "docno": doc})
        else:
            new.append({"qid": row.query_id, "docno": row.doc_id_b})
    return pd.DataFrame.from_records(new)


def _qrel_pivot(frame):
    new = []
    for row in frame.itertuples():
        new.append({"qid": row.query_id, "docno": row.doc_id, "score": row.relevance})
    return pd.DataFrame.from_records(new)


def get_teacher_scores(
    model: Any,
    corpus: Optional[pd.DataFrame] = None,
    ir_dataset: Optional[str] = None,
    subset: Optional[int] = None,
    negatives: Optional[dict] = None,
    seed: int = 42,
):
    """Retrieve teacher model scores for knowledge distillation.

    Generates scores from a teacher model on a corpus of query-document pairs,
    typically used for distillation training of student models.

    Args:
        model (Any): The teacher model to generate scores. Must have a transform method.
        corpus (pd.DataFrame, optional): DataFrame with 'query' and 'text' columns.
        ir_dataset (str, optional): Name of an ir_datasets dataset to load.
        subset (int, optional): Number of samples to randomly select from the corpus.
        negatives (dict, optional): Dictionary mapping query IDs to negative document IDs.
        seed (int, optional): Random seed for subset sampling. Defaults to 42.

    Returns:
        dict: Nested dictionary mapping query IDs to document IDs to scores.

    Raises:
        AssertionError: If neither corpus nor ir_dataset is provided.
        AssertionError: If corpus doesn't contain required columns.

    Examples:
        Getting teacher scores for distillation::

            from rankers import get_teacher_scores

            scores = get_teacher_scores(
                model=teacher_model,
                ir_dataset="msmarco-passage/train",
                subset=10000
            )
    """
    assert corpus is not None or ir_dataset is not None, (
        "Either corpus or ir_dataset must be provided"
    )
    if corpus:
        for column in ["query", "text"]:
            assert column in corpus.columns, f"{column} not found in corpus"
    if ir_dataset:
        dataset = irds.load(ir_dataset)
        docs = pd.DataFrame(dataset.docs_iter()).set_index("doc_id")["text"].to_dict()
        queries = (
            pd.DataFrame(dataset.queries_iter()).set_index("query_id")["text"].to_dict()
        )
        corpus = pd.DataFrame(dataset.docpairs_iter())
        if negatives:
            corpus = corpus[["query_id", "doc_id_a"]]
        corpus = _pivot(corpus, negatives)
        corpus["text"] = corpus["docno"].map(docs)
        corpus["query"] = corpus["qid"].map(queries)
        if subset:
            corpus = corpus.sample(n=subset, random_state=seed)

    logger.warning("Retrieving scores, this may take a while...")
    scores = model.transform(corpus)
    lookup = defaultdict(dict)
    for qid, group in scores.groupby("qid"):
        for docno, score in zip(group["docno"], group["score"]):
            lookup[qid][docno] = score
    return lookup


def initialise_irds_eval(dataset: irds.Dataset):
    """Initialize evaluation DataFrame from an ir_datasets Dataset.

    Converts an ir_datasets Dataset's qrels into a pivoted DataFrame format
    suitable for evaluation with rankers.

    Args:
        dataset (irds.Dataset): An ir_datasets Dataset object with qrels.

    Returns:
        pd.DataFrame: DataFrame with columns 'qid', 'docno', and 'score' containing
            relevance judgments.

    Examples:
        Preparing dataset for evaluation::

            import ir_datasets as irds
            from rankers import initialise_irds_eval

            dataset = irds.load("msmarco-passage/dev")
            qrels_df = initialise_irds_eval(dataset)
    """
    qrels = pd.DataFrame(dataset.qrels_iter())
    return _qrel_pivot(qrels)


def load_json(file: str):
    import gzip
    import json

    """
    Load a JSON or JSONL (optionally compressed with gzip) file.

    Parameters:
    file (str): The path to the file to load.

    Returns:
    dict or list: The loaded JSON content. Returns a list for JSONL files, 
                  and a dict for JSON files.

    Raises:
    ValueError: If the file extension is not recognized.
    """
    if file.endswith(".json"):
        with open(file) as f:
            return json.load(f)
    elif file.endswith(".jsonl"):
        with open(file) as f:
            return [json.loads(line) for line in f]
    elif file.endswith(".json.gz"):
        with gzip.open(file, "rt") as f:
            return json.load(f)
    elif file.endswith(".jsonl.gz"):
        with gzip.open(file, "rt") as f:
            return [json.loads(line) for line in f]
    else:
        raise ValueError(f"Unknown file type for {file}")


def save_json(data, file: str):
    import gzip
    import json

    """
    Save data to a JSON or JSONL file (optionally compressed with gzip).

    Parameters:
    data (dict or list): The data to save. Must be a list for JSONL files.
    file (str): The path to the file to save.

    Raises:
    ValueError: If the file extension is not recognized.
    """
    if file.endswith(".json"):
        with open(file, "w") as f:
            json.dump(data, f)
    elif file.endswith(".jsonl"):
        with open(file, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
    elif file.endswith(".json.gz"):
        with gzip.open(file, "wt") as f:
            json.dump(data, f)
    elif file.endswith(".jsonl.gz"):
        with gzip.open(file, "wt") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
    else:
        raise ValueError(f"Unknown file type for {file}")


def type_invariant_equal(a: Union[str, int], b: Union[str, int]) -> bool:
    """Check equality between two values regardless of their type (str or int).

    Compares two values by checking both string and integer equality, useful
    for matching IDs that may be represented as either strings or integers.

    Args:
        a (Union[str, int]): First value to compare.
        b (Union[str, int]): Second value to compare.

    Returns:
        bool: True if values are equal as either strings or integers.

    Examples:
        Comparing mixed-type IDs::

            from rankers import type_invariant_equal

            type_invariant_equal("123", 123)  # True
            type_invariant_equal(123, "123")  # True
            type_invariant_equal("abc", 123)  # False
    """
    return str(a) == str(b) or int(a) == int(b)


def read_trec(filename):
    """Read a TREC-formatted run file into a DataFrame.

    Parses TREC format files (space-separated) commonly used in IR evaluation.
    The standard TREC format has 6 columns: qid, iter, docno, rank, score, name.

    Args:
        filename (str): Path to the TREC-formatted file.

    Returns:
        pd.DataFrame: DataFrame with columns 'qid', 'docno', 'rank', 'score', 'name'.
            The 'iter' column is dropped as it's typically unused.

    Examples:
        Reading a TREC run file::

            from rankers import read_trec

            results = read_trec("run.trec")
            print(results.head())
    """
    df = pd.read_csv(
        filename,
        sep=r"\s+",
        names=["qid", "iter", "docno", "rank", "score", "name"],
        dtype={"qid": str, "docno": str, "rank": int, "score": float},
    )
    df = df.drop(columns="iter")
    return df


def write_trec(df, filename):
    """Write a DataFrame to a TREC-formatted run file.

    Writes results in standard TREC format (6 space-separated columns).
    Automatically adds default values for 'iter' (0) and 'name' ('rankers_run')
    columns if they don't exist.

    Args:
        df (pd.DataFrame): DataFrame with at least 'qid', 'docno', 'rank', 'score' columns.
        filename (str): Path where the TREC file should be written.

    Examples:
        Writing ranking results to TREC format::

            from rankers import write_trec
            import pandas as pd

            results = pd.DataFrame({
                'qid': ['q1', 'q1', 'q2'],
                'docno': ['d1', 'd2', 'd3'],
                'rank': [1, 2, 1],
                'score': [0.95, 0.87, 0.92]
            })
            write_trec(results, "output.trec")
    """
    if "iter" not in df.columns:
        df["iter"] = 0
    if "name" not in df.columns:
        df["name"] = "rankers_run"
    df = df[["qid", "iter", "docno", "rank", "score", "name"]]
    df.to_csv(filename, sep=r"\s+", header=False, index=False)
