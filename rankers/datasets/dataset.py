import random
from torch.utils.data import Dataset
import pandas as pd
import torch
from typing import Union

import json
from .._util import load_json, initialise_irds_eval, read_trec
from .._optional import is_ir_datasets_available
from .corpus import Corpus

if is_ir_datasets_available():
    import ir_datasets as irds


class LazyTextLoader:
    def __init__(self, corpus: Union[Corpus, irds.Dataset]) -> None:
        self.docstore = corpus.docs_store()

    def __getitem__(self, doc_id):
        if type(doc_id) is list:
            return [self.docstore.get(str(id)).text for id in doc_id]
        return self.docstore.get(str(doc_id)).text


class TrainingDataset(Dataset):
    def __init__(
        self,
        training_dataset_file: str,
        corpus: Union[Corpus, irds.Dataset],
        teacher_file: str = None,
        group_size: int = 2,
        no_positive: bool = False,
        lazy_load_text: bool = True,
        top_k_group: bool = False,
        precomputed: bool = False,
        text_field: str = "text",
        query_field: str = "text",
    ) -> None:
        assert training_dataset_file.endswith(
            "jsonl"
        ), "Training dataset should be a JSONL file and should not be compressed"

        self.training_dataset_file = training_dataset_file
        self.corpus = corpus
        self.teacher_file = teacher_file
        self.group_size = group_size
        self.no_positive = no_positive
        self.lazy_load_text = lazy_load_text
        self.n_neg = self.group_size - 1 if not self.no_positive else self.group_size
        self.top_k_group = top_k_group
        self.precomputed = precomputed
        self.text_field = text_field
        self.query_field = query_field

        self._get = self._precomputed_get if self.precomputed else self._standard_get

        self.line_offsets = self._get_line_offsets()
        super().__init__()
        self.__post_init__()

    def _get_line_offsets(self):
        """Store byte offsets for each line in an uncompressed JSONL file, skipping blank lines."""
        offsets = []
        with open(self.training_dataset_file, "r", encoding="utf-8") as f:
            while True:
                offset = f.tell()
                line = f.readline().strip()
                if not line:
                    if f.tell() == f.seek(0, 2):  # Check if we've reached the end of the file
                        break
                    continue
                offsets.append(offset)
        return offsets

    def _get_line_by_index(self, idx):
        """Retrieve a line by index, using offsets for uncompressed files."""
        with open(self.training_dataset_file, "r", encoding="utf-8") as f:
            f.seek(self.line_offsets[idx])
            return json.loads(f.readline())

    def __post_init__(self):
        assert (
            self.corpus is not None or self.precomputed
        ), "Cannot instantiate a text-based dataset without a lookup"

        # Initialize documents and queries from corpus
        if not self.precomputed:
            if not self.lazy_load_text:
                self.docs = (
                    pd.DataFrame(self.corpus.docs_iter())
                    .set_index("doc_id")["text"]
                    .to_dict()
                )
            else:
                self.docs = LazyTextLoader(self.corpus)

            self.queries = (
                pd.DataFrame(self.corpus.queries_iter())
                .set_index("query_id")["text"]
                .to_dict()
            )

        # Load teacher data if available
        if self.teacher_file:
            self.teacher = load_json(self.teacher_file)
            self.labels = True
        else:
            self.labels = False

        # Use _get_line_by_index to check multi-negative configuration
        first_entry = self._get_line_by_index(0)
        self.multi_negatives = isinstance(first_entry["doc_id_b"], list)
        total_negs = len(first_entry["doc_id_b"]) if self.multi_negatives else 1
        assert (
            self.n_neg <= total_negs
        ), f"Only found {total_negs} negatives, cannot take {self.n_neg} negatives"

    def __len__(self):
        # Length based on line offsets for uncompressed, or generator count for compressed
        return (
            len(self.line_offsets)
            if self.line_offsets
            else sum(1 for _ in self._data_generator())
        )

    def _teacher(self, query_id, doc_id):
        if query_id not in self.teacher:
            raise KeyError(f"Query ID {query_id} not found")
        if doc_id not in self.teacher[query_id]:
            raise KeyError(f"Doc ID {doc_id} not found for query {query_id}")
        return self.teacher[query_id][doc_id]

    def _precomputed_get(self, data):
        query, query_id, doc_id_a, doc_id_a_text, doc_id_b, doc_id_b_text = (
            data["query_id"],
            data[self.query_field],
            data["doc_id_a"],
            data[f"{self.text_field}_a"],
            data["doc_id_b"],
            data[f"{self.text_field}_b"],
        )
        return (query, query_id, [doc_id_a], [doc_id_a_text], doc_id_b, doc_id_b_text)

    def _standard_get(self, data):
        query_id, doc_id_a, doc_id_b = (
            data["query_id"],
            data["doc_id_a"],
            data["doc_id_b"],
        )
        query = self.queries[str(query_id)]
        doc_id_a_text = [self.docs[str(doc_id_a)]] if not self.no_positive else []

        if self.multi_negatives:
            doc_id_b_text = (
                [self.docs[str(doc)] for doc in doc_id_b]
                if not self.lazy_load_text
                else self.docs[doc_id_b]
            )
        else:
            doc_id_b_text = (
                [self.docs[str(doc_id_b)]]
                if not self.lazy_load_text
                else self.docs[doc_id_b]
            )

        return (query, query_id, doc_id_a, doc_id_a_text, doc_id_b, doc_id_b_text)

    def __getitem__(self, idx):
        # Retrieve the line corresponding to idx
        item = self._get_line_by_index(idx)

        query, query_id, doc_id_a, doc_id_a_text, doc_id_b, doc_id_b_text = self._get(
            item
        )

        # Append teacher scores if available
        if self.labels:
            doc_id_a_scores = (
                [self._teacher(str(query_id), str(doc_id_a))]
                if not self.no_positive
                else []
            )
            doc_id_b_scores = (
                [self._teacher(str(query_id), str(doc)) for doc in doc_id_b]
                if self.multi_negatives
                else [self._teacher(str(query_id), str(doc_id_b))]
            )

            if len(doc_id_b_text) > (self.n_neg):
                if self.top_k_group:
                    texts, scores = zip(
                        *sorted(
                            zip(
                                doc_id_a_text + doc_id_b_text,
                                doc_id_a_scores + doc_id_b_scores,
                            ),
                            key=lambda x: x[1],
                            reverse=True,
                        )
                    )
                    texts, scores = texts[: self.group_size], scores[: self.group_size]
                else:
                    # sample n_neg from the negatives
                    doc_id_b_text, doc_id_b_scores = zip(
                        *random.sample(
                            list(zip(doc_id_b_text, doc_id_b_scores)), self.n_neg
                        )
                    )
                    texts, scores = doc_id_a_text + [*doc_id_b_text], doc_id_a_scores + [*doc_id_b_scores]
                return (query, texts, scores)
            return (
                query,
                doc_id_a_text + doc_id_b_text,
                doc_id_a_scores + doc_id_b_scores,
            )
        else:
            if len(doc_id_b_text) > (self.n_neg):
                doc_id_b_text = random.sample(doc_id_b_text, self.n_neg)
            return (query, doc_id_a_text + doc_id_b_text)


class TestDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        corpus: Union[Corpus, irds.Dataset],
        lazy_load_text: bool = True,
    ) -> None:
        super().__init__()
        self.data = data
        self.corpus = corpus
        self.lazy_load_text = lazy_load_text

        self.__post_init__()

    def __post_init__(self):
        for column in "query_id", "docno", "score":
            if column not in self.data.columns:
                raise ValueError(
                    f"Format not recognised, Column '{column}' not found in dataframe"
                )
        if not self.lazy_load_text:
            self.docs = (
                pd.DataFrame(self.corpus.docs_iter())
                .set_index("doc_id")["text"]
                .to_dict()
            )
        else:
            self.docs = LazyTextLoader(self.corpus)
        self.queries = (
            pd.DataFrame(self.corpus.queries_iter())
            .set_index("query_id")["text"]
            .to_dict()
        )
        self.qrels = pd.DataFrame(self.corpus.qrels_iter())

        self.data["text"] = self.data["docno"].map(
            self.docs
        )
        self.data["query"] = self.data["qid"].map(
            self.queries
        )

    @classmethod
    def from_trec(
        cls,
        trec_file: str,
        ir_dataset: irds.Dataset,
    ) -> "TestDataset":
        data = read_trec(trec_file)
        return cls(data, ir_dataset)

    @classmethod
    def from_irds(
        cls,
        ir_dataset: irds.Dataset,
    ) -> "TestDataset":
        if not is_ir_datasets_available():
            raise ImportError(
                "ir_datasets is not available, please install ir_datasets to use this function"
            )
        data = initialise_irds_eval(ir_dataset)
        return cls(data, ir_dataset)

    def __len__(self):
        return len(self.data.query_id.unique())
