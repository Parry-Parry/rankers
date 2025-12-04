"""Evaluation dataset for testing and validation of neural rankers."""

import json
from typing import Any, Dict, Optional, Union

import pandas as pd
from torch.utils.data import Dataset

from .._optional import is_ir_datasets_available
from .._util import initialise_irds_eval, read_trec
from .corpus import Corpus
from .lazy_text_loader import LazyTextLoader

if is_ir_datasets_available():
    import ir_datasets as irds


class EvaluationDataset(Dataset):
    """Dataset for evaluating ranker models on test and validation data.

    Supports multiple evaluation modes:
    - TREC format: Load ranking results from TREC run files
    - ir_datasets: Load benchmarks from standard IR datasets
    - JSONL format: Build evaluation data from training-format JSONL files with pseudo-qrels

    Args:
        data (pd.DataFrame, optional): DataFrame in TREC format with 'qid', 'docno', 'score' columns.
        corpus (Union[Corpus, irds.Dataset]): Corpus containing documents and queries.
        lazy_load_text (bool, optional): Load document text on-demand. Defaults to True.
        jsonl_file (str, optional): Path to JSONL training file for building evaluation data.
        query_id_key (str, optional): JSON key for query ID. Defaults to "query_id".
        positive_id_key (str, optional): JSON key for positive doc ID. Defaults to "doc_id_a".
        negative_id_key (str, optional): JSON key for negative doc ID. Defaults to "doc_id_b".
        relevance_label (int, optional): Relevance score for positive documents. Defaults to 1.
        include_negatives (bool, optional): Include negative documents in ranking frame. Defaults to True.
        dedupe_qrels (bool, optional): Remove duplicate qrels. Defaults to True.

    Attributes:
        data (pd.DataFrame): Test data with enriched query and document text.
        queries (dict): Mapping of query IDs to query text.
        docs: Document loader (LazyTextLoader or dict).
        qrels (pd.DataFrame): Relevance judgments.

    Examples:
        Loading from TREC file::

            import ir_datasets as irds
            from rankers.datasets import EvaluationDataset

            dataset = irds.load("msmarco-passage/dev")
            eval_data = EvaluationDataset.from_trec("run.trec", dataset)

        Loading from ir_datasets::

            eval_data = EvaluationDataset.from_irds(dataset)

        Building from JSONL (validation mode)::

            eval_data = EvaluationDataset.from_jsonl("train.jsonl", corpus)

        Building from qrels DataFrame::

            eval_data = EvaluationDataset.from_qrels(qrels_df, corpus)
    """

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        corpus: Optional[Union[Corpus, "irds.Dataset"]] = None,
        lazy_load_text: bool = True,
        jsonl_file: Optional[str] = None,
        query_id_key: str = "query_id",
        positive_id_key: str = "doc_id_a",
        negative_id_key: str = "doc_id_b",
        relevance_label: int = 1,
        include_negatives: bool = True,
        dedupe_qrels: bool = True,
    ) -> None:
        super().__init__()
        self.corpus = corpus
        self.lazy_load_text = lazy_load_text
        self.jsonl_file = jsonl_file
        self.query_id_key = query_id_key
        self.positive_id_key = positive_id_key
        self.negative_id_key = negative_id_key
        self.relevance_label = relevance_label
        self.include_negatives = include_negatives
        self.dedupe_qrels = dedupe_qrels

        # If loading from JSONL, build data and qrels
        if jsonl_file is not None:
            assert jsonl_file.endswith("jsonl"), "JSONL file should not be compressed"
            self._build_from_jsonl()
        else:
            # Standard TREC/ir_datasets mode
            assert data is not None, "Either 'data' or 'jsonl_file' must be provided"
            self.data = data

        self.__post_init__()

    def __post_init__(self):
        # Validate required columns
        for column in ("qid", "docno"):
            if column not in self.data.columns:
                raise ValueError(f"Column '{column}' not found in dataframe")

        # Load text loaders
        if not self.lazy_load_text:
            self.docs = pd.DataFrame(self.corpus.docs_iter()).set_index("doc_id")["text"].to_dict()
            self.queries = (
                pd.DataFrame(self.corpus.queries_iter()).set_index("query_id")["text"].to_dict()
            )
        else:
            # Use lazy loading with caching
            self.docs = LazyTextLoader(self.corpus, cache_size=10000, mode="docs")
            self.queries = LazyTextLoader(self.corpus, cache_size=2000, mode="queries")

        # Load qrels if not already built from JSONL
        if not hasattr(self, "qrels"):
            # qrels_iter returns dicts (Corpus) or namedtuples (ir_datasets) - pandas handles both
            qrels_list = list(self.corpus.qrels_iter())
            if qrels_list:
                self.qrels = pd.DataFrame(qrels_list)
            else:
                self.qrels = pd.DataFrame(columns=["query_id", "doc_id", "relevance"])

        # Enrich with text and query fields
        try:
            self.data["text"] = self.data["docno"].map(self.docs)
        except KeyError as e:
            raise KeyError(
                f"Document ID not found in corpus: {e}. "
                "Ensure all documents referenced in the data exist in the corpus."
            ) from e

        try:
            self.data["query"] = self.data["qid"].map(self.queries)
        except KeyError as e:
            raise KeyError(
                f"Query ID not found in corpus: {e}. "
                "Ensure all queries referenced in the data exist in the corpus. "
                "You may need to use a corpus that includes all queries from your evaluation set."
            ) from e

    def _build_from_jsonl(self):
        """Build data and qrels DataFrames from JSONL training format."""
        self.qrels = self._build_qrels_from_jsonl()
        self.data = self._build_data_from_jsonl()

    def _build_qrels_from_jsonl(self) -> pd.DataFrame:
        """Build qrels DataFrame from JSONL positives only."""
        rows: list[dict[str, Any]] = []
        with open(self.jsonl_file, "rb") as f:
            for raw in f:
                if not raw.strip():
                    continue
                rec = json.loads(raw.decode("utf-8"))

                if self.query_id_key not in rec:
                    raise KeyError(f"Key {self.query_id_key} not found in record")
                qid = rec[self.query_id_key]

                if self.positive_id_key not in rec:
                    continue

                pos = rec[self.positive_id_key]
                if pos is None:
                    continue

                if isinstance(pos, list):
                    for d in pos:
                        if d is None:
                            continue
                        rows.append(
                            {
                                "query_id": str(qid),
                                "doc_id": str(d),
                                "relevance": self.relevance_label,
                            }
                        )
                else:
                    rows.append(
                        {
                            "query_id": str(qid),
                            "doc_id": str(pos),
                            "relevance": self.relevance_label,
                        }
                    )

        if rows:
            df = pd.DataFrame(rows)
        else:
            df = pd.DataFrame(columns=["query_id", "doc_id", "relevance"])

        if self.dedupe_qrels and not df.empty:
            df = df.drop_duplicates(subset=["query_id", "doc_id"], keep="first").reset_index(
                drop=True
            )
        return df

    def _build_data_from_jsonl(self) -> pd.DataFrame:
        """Build TREC-style ranking DataFrame from JSONL training format."""
        rows: list[dict[str, Any]] = []
        with open(self.jsonl_file, "rb") as f:
            for raw in f:
                if not raw.strip():
                    continue
                rec = json.loads(raw.decode("utf-8"))

                if self.query_id_key not in rec or self.negative_id_key not in rec:
                    raise KeyError(
                        f"Record missing required keys: {self.query_id_key} / {self.negative_id_key}"
                    )

                qid = rec[self.query_id_key]

                # Positives
                if self.positive_id_key in rec:
                    pos = rec[self.positive_id_key]
                    if pos is not None:
                        if isinstance(pos, list):
                            for d in pos:
                                if d is None:
                                    continue
                                rows.append(
                                    {
                                        "qid": str(qid),
                                        "docno": str(d),
                                        "score": self.relevance_label,
                                    }
                                )
                        else:
                            rows.append(
                                {
                                    "qid": str(qid),
                                    "docno": str(pos),
                                    "score": self.relevance_label,
                                }
                            )

                # Negatives (single or list)
                if self.include_negatives:
                    neg = rec[self.negative_id_key]
                    if isinstance(neg, list):
                        for d in neg:
                            rows.append(
                                {
                                    "qid": str(qid),
                                    "docno": str(d),
                                    "score": 0,
                                }
                            )
                    else:
                        rows.append(
                            {
                                "qid": str(qid),
                                "docno": str(neg),
                                "score": 0,
                            }
                        )

        if rows:
            return pd.DataFrame(rows)
        else:
            return pd.DataFrame(columns=["qid", "docno", "score"])

    @classmethod
    def from_trec(
        cls,
        trec_file: str,
        corpus: Union[Corpus, "irds.Dataset"],
        lazy_load_text: bool = True,
    ) -> "EvaluationDataset":
        """Load evaluation data from TREC run file.

        Args:
            trec_file (str): Path to TREC format file.
            corpus (Union[Corpus, irds.Dataset]): Corpus containing documents and queries.
            lazy_load_text (bool, optional): Load text on-demand. Defaults to True.

        Returns:
            EvaluationDataset: Initialized dataset.
        """
        data = read_trec(trec_file)
        return cls(data=data, corpus=corpus, lazy_load_text=lazy_load_text)

    @classmethod
    def from_irds(
        cls,
        corpus: "irds.Dataset",
        lazy_load_text: bool = True,
    ) -> "EvaluationDataset":
        """Load evaluation data from ir_datasets benchmark.

        Args:
            corpus (irds.Dataset): IR dataset to load.
            lazy_load_text (bool, optional): Load text on-demand. Defaults to True.

        Returns:
            EvaluationDataset: Initialized dataset.
        """
        if not is_ir_datasets_available():
            raise ImportError(
                "ir_datasets is not available, please install ir_datasets to use this function"
            )
        data = initialise_irds_eval(corpus)
        return cls(data=data, corpus=corpus, lazy_load_text=lazy_load_text)

    @classmethod
    def from_jsonl(
        cls,
        jsonl_file: str,
        corpus: Union[Corpus, "irds.Dataset"],
        query_id_key: str = "query_id",
        positive_id_key: str = "doc_id_a",
        negative_id_key: str = "doc_id_b",
        relevance_label: int = 1,
        include_negatives: bool = True,
        dedupe_qrels: bool = True,
        lazy_load_text: bool = True,
    ) -> "EvaluationDataset":
        """Build evaluation data from training-format JSONL file.

        Creates both qrels and ranking DataFrame from positive/negative document pairs,
        useful for validation during training.

        Args:
            jsonl_file (str): Path to JSONL training file.
            corpus (Union[Corpus, irds.Dataset]): Corpus containing documents and queries.
            query_id_key (str, optional): JSON key for query ID. Defaults to "query_id".
            positive_id_key (str, optional): JSON key for positive doc ID. Defaults to "doc_id_a".
            negative_id_key (str, optional): JSON key for negative doc ID. Defaults to "doc_id_b".
            relevance_label (int, optional): Relevance score for positives. Defaults to 1.
            include_negatives (bool, optional): Include negatives in ranking frame. Defaults to True.
            dedupe_qrels (bool, optional): Remove duplicate qrels. Defaults to True.
            lazy_load_text (bool, optional): Load text on-demand. Defaults to True.

        Returns:
            EvaluationDataset: Initialized dataset with pseudo-qrels from positives.
        """
        return cls(
            corpus=corpus,
            lazy_load_text=lazy_load_text,
            jsonl_file=jsonl_file,
            query_id_key=query_id_key,
            positive_id_key=positive_id_key,
            negative_id_key=negative_id_key,
            relevance_label=relevance_label,
            include_negatives=include_negatives,
            dedupe_qrels=dedupe_qrels,
        )

    @classmethod
    def from_qrels(
        cls,
        qrels_df: pd.DataFrame,
        corpus: Union[Corpus, "irds.Dataset"],
        lazy_load_text: bool = True,
    ) -> "EvaluationDataset":
        """Create evaluation dataset from qrels DataFrame.

        Args:
            qrels_df (pd.DataFrame): DataFrame with columns [query_id, doc_id, relevance].
            corpus (Union[Corpus, irds.Dataset]): Corpus containing documents and queries.
            lazy_load_text (bool, optional): Load text on-demand. Defaults to True.

        Returns:
            EvaluationDataset: Initialized dataset.
        """
        # Convert qrels to TREC format
        data = qrels_df.rename(columns={"query_id": "qid", "doc_id": "docno", "relevance": "score"})
        return cls(data=data, corpus=corpus, lazy_load_text=lazy_load_text)

    def __len__(self):
        return self.data.qid.nunique()
