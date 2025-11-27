"""Corpus management for neural ranking.

This module provides a simple interface for managing document collections, queries,
and relevance judgments compatible with ir_datasets format.
"""

from types import SimpleNamespace

import pandas as pd


class Corpus:
    """In-memory corpus of documents, queries, and relevance judgments.

    Provides a unified interface for accessing corpus data, compatible with ir_datasets
    API. Useful for small to medium-sized collections or custom datasets.

    Args:
        documents (dict, optional): Mapping of document IDs to text. Defaults to None.
        queries (dict, optional): Mapping of query IDs to text. Defaults to None.
        qrels (pd.DataFrame, optional): Relevance judgments with columns
            'query_id', 'doc_id', 'relevance'. Defaults to None.

    Attributes:
        documents (dict): Document ID to text mapping.
        queries (dict): Query ID to text mapping.
        qrels (pd.DataFrame): Relevance judgments.

    Examples:
        Creating a custom corpus::

            from rankers.datasets import Corpus
            import pandas as pd

            corpus = Corpus(
                documents={"d1": "First document", "d2": "Second document"},
                queries={"q1": "What is IR?"},
                qrels=pd.DataFrame([
                    {"query_id": "q1", "doc_id": "d1", "relevance": 1}
                ])
            )

        Using with datasets::

            from rankers.datasets import TrainingDataset

            dataset = TrainingDataset(
                training_dataset_file="train.jsonl",
                corpus=corpus
            )

    Note:
        For large corpora, consider using ir_datasets or lazy loading to avoid
        memory issues.
    """

    def __init__(
        self, documents: dict = None, queries: dict = None, qrels: pd.DataFrame = None
    ) -> None:
        self.documents = documents
        self.queries = queries
        self.qrels = qrels

        self.__post_init__()

    def __post_init__(self):
        if self.qrels is not None:
            for column in "query_id", "doc_id", "relevance":
                if column not in self.qrels.columns:
                    raise ValueError(
                        f"Format not recognised, Column '{column}' not found in qrels dataframe"
                    )

            self.qrels = self.qrels[["query_id", "doc_id", "relevance"]]

    def docs_store(self):
        return self

    def get(self, doc_id):
        return SimpleNamespace(text=self.documents[doc_id])

    def get_many(self, doc_ids):
        return [SimpleNamespace(text=self.documents[doc_id]) for doc_id in doc_ids]

    def has_documents(self):
        return self.documents is not None

    def has_queries(self):
        return self.queries is not None

    def has_qrels(self):
        return self.qrels is not None

    def queries_iter(self):
        for queryid, text in self.queries.items():
            yield {"query_id": queryid, "text": text}

    def docs_iter(self):
        for docid, text in self.documents.items():
            yield {"doc_id": docid, "text": text}

    def qrels_iter(self):
        for queryid, docid, relevance in self.qrels.itertuples(index=False):
            yield SimpleNamespace(
                query_id=queryid, doc_id=docid, relevance=relevance
            )
    def __call__(self, maybe_idx):
        if isinstance(maybe_idx, int):
            return self.get(maybe_idx)
        elif isinstance(maybe_idx, list):
            return self.get_many(maybe_idx)
        else:
            raise ValueError("Invalid input type. Must be int or list")
