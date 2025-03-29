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
        query_id_key: str = "query_id",
        positive_id_key: str = "doc_id_a",
        negative_id_key: str = "doc_id_b",
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
        self.query_id_key = query_id_key
        self.positive_id_key = positive_id_key
        self.negative_id_key = negative_id_key
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

        # check required keys are present in the first entry
        assert (
            self.query_id_key_key in first_entry
        ), f"Key {self.query_id_key} not found in the first entry"
        if not self.no_positive:
            assert (
                self.positive_id_key in first_entry
            ), f"Key {self.positive_id_key} not found in the first entry"
        assert (
            self.negative_id_key in first_entry
        ), f"Key {self.negative_id_key} not found in the first entry"

        self.multi_negatives = isinstance(first_entry[self.negative_id_key], list)
        total_negs = (
            len(first_entry[self.negative_id_key]) if self.multi_negatives else 1
        )
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
            raise KeyError(f"Doc ID {doc_id} not found for query {self.query_id_key}")
        return self.teacher[self.query_id_key][doc_id]

    def _precomputed_get(self, data):
        query_id, query_text, positive_id, positive_text, negative_id, negative_text = (
            data[self.query_id_key],
            data[self.query_field],
            data[self.positive_id_key],
            data[f"{self.positive_id_key}_text"],
            data[self.negative_id_key],
            data[f"{self.negative_id_key}_text"],
        )
        return (
            query_id,
            query_text,
            [positive_id],
            [positive_text],
            negative_id,
            negative_text,
        )

    def _standard_get(self, data):
        query_id, positive_id, negative_id = (
            data[self.query_id_key],
            data[self.positive_id_key],
            data[self.negative_id_key],
        )
        query_text = self.queries[str(query_id)]
        positive_text = [self.docs[str(positive_id)]] if not self.no_positive else []

        if self.multi_negatives:
            negative_text = (
                [self.docs[str(doc)] for doc in negative_id]
                if not self.lazy_load_text
                else self.docs[negative_id]
            )
        else:
            negative_text = (
                [self.docs[str(negative_id)]]
                if not self.lazy_load_text
                else self.docs[negative_id]
            )

        return (
            query_id,
            query_text,
            positive_id,
            positive_text,
            negative_id,
            negative_text,
        )

    def __getitem__(self, idx):
        # Retrieve the line corresponding to idx
        item = self._get_line_by_index(idx)

        query_id, query_text, positive_id, positive_text, negative_id, negative_text = (
            self._get(item)
        )

        # Append teacher scores if available
        if self.labels:
            positive_score = (
                [self._teacher(str(query_id), str(positive_id))]
                if not self.no_positive
                else []
            )
            negative_score = (
                [self._teacher(str(query_id), str(doc)) for doc in negative_id]
                if self.multi_negatives
                else [self._teacher(str(query_id), str(negative_id))]
            )

            if len(negative_id) > (self.n_neg):
                if self.top_k_group:
                    texts, scores = zip(
                        *sorted(
                            zip(
                                positive_text + negative_text,
                                positive_score + negative_score,
                            ),
                            key=lambda x: x[1],
                            reverse=True,
                        )
                    )
                    texts, scores = texts[: self.group_size], scores[: self.group_size]
                else:
                    # sample n_neg from the negatives
                    negative_text, negative_score = zip(
                        *random.sample(
                            list(zip(negative_text, negative_score)), self.n_neg
                        )
                    )
                    texts, scores = positive_text + [*negative_text], positive_score + [
                        *negative_score
                    ]
                return (query_text, texts, scores)
            return (
                query_text,
                positive_text + negative_text,
                positive_score + negative_score,
            )
        else:
            if len(negative_text) > (self.n_neg):
                negative_text = random.sample(negative_text, self.n_neg)
            return (query_text, positive_text + negative_text)


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
        for column in "qid", "docno", "score":
            if column not in self.data.columns:
                raise ValueError(
                    f"Format not recognised (Should be TREC), Column '{column}' not found in dataframe"
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

        self.data["text"] = self.data["docno"].map(self.docs)
        self.data["query"] = self.data["qid"].map(self.queries)

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
        return len(self.data.qid.unique())
