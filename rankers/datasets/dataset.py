import os
import json
import mmap
import random
from typing import Union, Iterable, List, Dict, Any, Tuple, Optional

import pandas as pd
from torch.utils.data import Dataset

from .._util import load_json, initialise_irds_eval, read_trec
from .._optional import is_ir_datasets_available
from .corpus import Corpus

if is_ir_datasets_available():
    import ir_datasets as irds


class LazyTextLoader:
    def __init__(self, corpus: Union["irds.Dataset", Corpus]) -> None:
        self.docstore = corpus.docs_store()

    def __getitem__(self, doc_id):
        if isinstance(doc_id, list):
            return [self.docstore.get(str(i)).text for i in doc_id]
        return self.docstore.get(str(doc_id)).text

    def __call__(self, id_):
        return self[id_]


class TrainingDataset(Dataset):
    def __init__(
        self,
        training_dataset_file: str,
        corpus: Union[Corpus, "irds.Dataset"],
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

        # Persistent, per-process/worker file resources (lazy-opened)
        self._fh = None           # file handle
        self._mm = None           # mmap object
        self._opened_in_pid = None

        self.line_offsets = self._get_line_offsets()
        super().__init__()
        self.__post_init__()

    def _ensure_open(self, use_mmap: bool = True):
        """Ensure a process/worker-local file handle (and mmap) is open."""
        pid = os.getpid()
        if self._fh is not None and self._opened_in_pid == pid:
            return
        self._close()
        self._fh = open(self.training_dataset_file, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ) if use_mmap else None
        self._opened_in_pid = pid

    def _close(self):
        """Close mmap and file handle if open."""
        try:
            if self._mm is not None:
                self._mm.close()
        finally:
            self._mm = None
            if self._fh is not None:
                self._fh.close()
            self._fh = None
        self._opened_in_pid = None

    def __del__(self):
        try:
            self._close()
        except Exception:
            pass

    def __getstate__(self):
        """Drop unpicklable state so DataLoader workers reopen cleanly."""
        state = self.__dict__.copy()
        state["_fh"] = None
        state["_mm"] = None
        state["_opened_in_pid"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _get_line_offsets(self) -> List[int]:
        """Store byte offsets for each non-blank line in an uncompressed JSONL file."""
        offsets: List[int] = []
        with open(self.training_dataset_file, "rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(pos)
        return offsets

    def _get_line_by_index(self, idx: int) -> Dict[str, Any]:
        """Retrieve a JSON object by index, using saved byte offsets."""
        self._ensure_open(use_mmap=True)
        offset = self.line_offsets[idx]
        if self._mm is not None:
            self._mm.seek(offset)
            raw = self._mm.readline()
        else:
            self._fh.seek(offset)
            raw = self._fh.readline()
        return json.loads(raw.decode("utf-8"))

    def iter_jsonl(self) -> Iterable[Tuple[int, Dict[str, Any]]]:
        """Yield (index, record) by streaming the file once (low memory)."""
        with open(self.training_dataset_file, "rb") as f:
            i = 0
            while True:
                raw = f.readline()
                if not raw:
                    break
                if not raw.strip():
                    continue
                yield i, json.loads(raw.decode("utf-8"))
                i += 1

    def __post_init__(self):
        assert (
            self.corpus is not None or self.precomputed
        ), "Cannot instantiate a text-based dataset without a lookup"

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

        if self.teacher_file:
            self.teacher = load_json(self.teacher_file)
            self.labels = True
        else:
            self.labels = False

        first_entry = self._get_line_by_index(0)

        assert (
            self.query_id_key in first_entry
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
        return len(self.line_offsets)

    def _teacher(self, query_id, doc_id):
        if query_id not in self.teacher:
            raise KeyError(f"Query ID {query_id} not found")
        if doc_id not in self.teacher[query_id]:
            raise KeyError(f"Doc ID {doc_id} not found for query {query_id}")
        return self.teacher[query_id][doc_id]

    def _precomputed_get(self, data: Dict[str, Any]):
        query_id = data[self.query_id_key]
        query_text = data[self.query_field]
        positive_id = data[self.positive_id_key] if not self.no_positive else None
        positive_text = data[f"{self.positive_id_key}_text"] if not self.no_positive else None
        negative_id = data[self.negative_id_key]
        negative_text = data[f"{self.negative_id_key}_text"]
        return (
            query_id,
            query_text,
            [positive_id],
            [positive_text],
            negative_id,
            negative_text,
        )

    def _standard_get(self, data: Dict[str, Any]):
        query_id = data[self.query_id_key]
        positive_id = data[self.positive_id_key] if not self.no_positive else None
        negative_id = data[self.negative_id_key]

        query_text = self.queries[str(query_id)]
        positive_text = [self.docs[str(positive_id)]] if not self.no_positive else []

        if self.multi_negatives:
            if not self.lazy_load_text:
                negative_text = [self.docs[str(doc)] for doc in negative_id]
            else:
                negative_text = self.docs[negative_id]  # LazyTextLoader handles list
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

    # ---- sampling / grouping ----
    def __getitem__(self, idx: int):
        item = self._get_line_by_index(idx)

        query_id, query_text, positive_id, positive_text, negative_id, negative_text = (
            self._get(item)
        )

        if self.labels:
            positive_score = (
                [self._teacher(str(query_id), str(positive_id))]
                if not self.no_positive
                else []
            )
            if self.multi_negatives:
                negative_score = [self._teacher(str(query_id), str(doc)) for doc in negative_id]
            else:
                negative_score = [self._teacher(str(query_id), str(negative_id))]

            # Downsample negatives if more than needed
            n_negs_present = len(negative_id) if self.multi_negatives else 1
            if n_negs_present > self.n_neg:
                if self.top_k_group:
                    texts, scores = zip(
                        *sorted(
                            zip(positive_text + negative_text, positive_score + negative_score),
                            key=lambda x: x[1],
                            reverse=True,
                        )
                    )
                    texts, scores = texts[: self.group_size], scores[: self.group_size]
                else:
                    neg_pairs = list(zip(negative_text, negative_score))
                    sampled = random.sample(neg_pairs, self.n_neg)
                    s_neg_text, s_neg_score = zip(*sampled)
                    texts = positive_text + list(s_neg_text)
                    scores = positive_score + list(s_neg_score)
                return (query_text, texts, scores)

            # exactly the requested count (or fewer)
            return (
                query_text,
                positive_text + negative_text,
                positive_score + negative_score,
            )
        else:
            # No labels path
            n_total_negs = len(negative_text) if isinstance(negative_text, list) else 1
            if n_total_negs > self.n_neg:
                if isinstance(negative_text, list):
                    negative_text = random.sample(negative_text, self.n_neg)
                else:
                    negative_text = [negative_text]
            return (query_text, positive_text + negative_text)


class TestDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        corpus: Union[Corpus, "irds.Dataset"],
        lazy_load_text: bool = True,
    ) -> None:
        super().__init__()
        self.data = data
        self.corpus = corpus
        self.lazy_load_text = lazy_load_text
        self.__post_init__()

    def __post_init__(self):
        for column in ("qid", "docno", "score"):
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
        ir_dataset: "irds.Dataset",
    ) -> "TestDataset":
        data = read_trec(trec_file)
        return cls(data, ir_dataset)

    @classmethod
    def from_irds(
        cls,
        ir_dataset: "irds.Dataset",
    ) -> "TestDataset":
        if not is_ir_datasets_available():
            raise ImportError(
                "ir_datasets is not available, please install ir_datasets to use this function"
            )
        data = initialise_irds_eval(ir_dataset)
        return cls(data, ir_dataset)

    def __len__(self):
        return self.data.qid.nunique()


class ValidationDataset(Dataset):
    """
    Build a TREC-style DataFrame (qid, query, docno, text) from a training-format JSONL.

    - Accepts the same key configuration as TrainingDataset (query_id_key, positive_id_key,
      negative_id_key), and the same corpus for text lookup.
    - Includes the positive (if present) and all negatives per example.
    - No scores are produced (column 'score' is intentionally omitted).
    - Exposes the resulting DataFrame via `data` and provides len() = number of rows.
    """

    def __init__(
        self,
        training_dataset_file: str,
        corpus: Union[Corpus, "irds.Dataset"],
        query_id_key: str = "query_id",
        positive_id_key: str = "doc_id_a",
        negative_id_key: str = "doc_id_b",
        lazy_load_text: bool = True,
        include_positive: bool = True,
    ) -> None:
        super().__init__()
        assert training_dataset_file.endswith("jsonl"), "validation expects an uncompressed JSONL"

        self.training_dataset_file = training_dataset_file
        self.corpus = corpus
        self.query_id_key = query_id_key
        self.positive_id_key = positive_id_key
        self.negative_id_key = negative_id_key
        self.lazy_load_text = lazy_load_text
        self.include_positive = include_positive

        # Build text lookup (lazy by default)
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

        # Materialize TREC-style DataFrame
        self.data = self._build_dataframe()

    def _build_dataframe(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []

        # Stream once; do not depend on offsets here.
        with open(self.training_dataset_file, "rb") as f:
            for raw in f:
                if not raw.strip():
                    continue
                rec = json.loads(raw.decode("utf-8"))

                # Validate keys
                if self.query_id_key not in rec:
                    raise KeyError(f"Key {self.query_id_key} not found in record")
                if self.negative_id_key not in rec:
                    raise KeyError(f"Key {self.negative_id_key} not found in record")

                qid = rec[self.query_id_key]
                qtext = self.queries[str(qid)]

                # Positive (optional)
                if self.include_positive and (self.positive_id_key in rec):
                    pos = rec[self.positive_id_key]
                    if pos is not None:
                        rows.append({
                            "qid": qid,
                            "query": qtext,
                            "docno": str(pos),
                            "text": self.docs[str(pos)],
                        })

                # Negatives (single or list)
                neg = rec[self.negative_id_key]
                if isinstance(neg, list):
                    for d in neg:
                        rows.append({
                            "qid": qid,
                            "query": qtext,
                            "docno": str(d),
                            "text": self.docs[str(d)],
                        })
                else:
                    rows.append({
                        "qid": qid,
                        "query": qtext,
                        "docno": str(neg),
                        "text": self.docs[str(neg)],
                    })

        # DataFrame with exact columns requested
        df = pd.DataFrame(rows, columns=["qid", "query", "docno", "text"])
        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        # Optional: support Dataset-style indexing over rows
        row = self.data.iloc[idx]
        # Return a tuple matching the TREC-style expectation
        return row["qid"], row["query"], row["docno"], row["text"]
