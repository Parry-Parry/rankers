"""Training dataset for neural ranker training with query-document pairs."""

import json
import mmap
import os
import random
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from torch.utils.data import Dataset

from .._util import load_json
from .corpus import Corpus
from .lazy_text_loader import LazyTextLoader

try:
    import ir_datasets as irds
except ImportError:
    irds = None


class TrainingDataset(Dataset):
    """Dataset for training neural rankers with query-document pairs.

    Loads training data from JSONL files containing query-positive-negative tuples.
    Supports various training configurations including group sizes, teacher scores
    for distillation, and lazy text loading for memory efficiency.

    The dataset handles fork-safe file access for multiprocessing and provides
    efficient random access through memory-mapped files.

    Args:
        training_dataset_file (str): Path to JSONL training file.
        corpus (Union[Corpus, irds.Dataset]): Document corpus for text retrieval.
        teacher_file (str, optional): Path to teacher scores file. Defaults to None.
        group_size (int, optional): Number of documents per query (1 positive + negatives).
            Use -1 to include all available negatives from data. Defaults to -1.
        no_positive (bool, optional): Whether to exclude positive documents. Defaults to False.
        lazy_load_text (bool, optional): Load document text on-demand. Defaults to True.
        top_k_group (bool, optional): Select top-k documents from group. Defaults to False.
        precomputed (bool, optional): Whether text is pre-computed in dataset. Defaults to False.
        query_id_key (str, optional): JSON key for query ID. Defaults to "query_id".
        positive_id_key (str, optional): JSON key for positive doc ID. Defaults to "doc_id_a".
        negative_id_key (str, optional): JSON key for negative doc ID. Defaults to "doc_id_b".
        text_field (str, optional): Field name for document text. Defaults to "text".
        query_field (str, optional): Field name for query text. Defaults to "text".

    Examples:
        Basic training dataset (uses all negatives by default)::

            from rankers.datasets import TrainingDataset, Corpus

            corpus = Corpus.from_ir_datasets("msmarco-passage")
            # Uses all available negatives per query
            dataset = TrainingDataset(
                training_dataset_file="train.jsonl",
                corpus=corpus
            )

        With fixed group size::

            # Use specific group size (1 positive + 7 negatives)
            dataset = TrainingDataset(
                training_dataset_file="train.jsonl",
                corpus=corpus,
                group_size=8
            )

        With teacher distillation::

            dataset = TrainingDataset(
                training_dataset_file="train.jsonl",
                corpus=corpus,
                teacher_file="teacher_scores.json",
                group_size=8
            )

    Note:
        The JSONL file must not be compressed for memory-mapped access.
        Each line should be a JSON object with query and document IDs.
    """

    def __init__(
        self,
        training_dataset_file: str,
        corpus: Union[Corpus, "irds.Dataset"],
        teacher_file: str = None,
        group_size: int = -1,
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
        assert training_dataset_file.endswith("jsonl"), (
            "Training dataset should be a JSONL file and should not be compressed"
        )

        self.training_dataset_file = training_dataset_file
        self.corpus = corpus
        self.teacher_file = teacher_file
        self.group_size = group_size
        self.no_positive = no_positive
        self.lazy_load_text = lazy_load_text
        # If group_size is -1, use all negatives available in data
        self.n_neg = (
            (self.group_size - 1 if not self.no_positive else self.group_size)
            if self.group_size > 0
            else -1
        )
        self.top_k_group = top_k_group
        self.precomputed = precomputed
        self.query_id_key = query_id_key
        self.positive_id_key = positive_id_key
        self.negative_id_key = negative_id_key
        self.text_field = text_field
        self.query_field = query_field

        self._get = self._precomputed_get if self.precomputed else self._standard_get

        # Persistent, per-process/worker file resources (lazy-opened)
        self._fh = None  # file handle
        self._mm = None  # mmap object
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
        assert self.corpus is not None or self.precomputed, (
            "Cannot instantiate a text-based dataset without a lookup"
        )

        if not self.precomputed:
            if not self.lazy_load_text:
                # Load both docs and queries into memory
                self.docs = (
                    pd.DataFrame(self.corpus.docs_iter()).set_index("doc_id")["text"].to_dict()
                )
                self.queries = (
                    pd.DataFrame(self.corpus.queries_iter()).set_index("query_id")["text"].to_dict()
                )
            else:
                # Use lazy loading with caching for both docs and queries
                self.docs = LazyTextLoader(self.corpus, cache_size=10000, mode="docs")
                self.queries = LazyTextLoader(self.corpus, cache_size=5000, mode="queries")

        if self.teacher_file:
            self.teacher = load_json(self.teacher_file)
            self.labels = True
        else:
            self.labels = False

        # Validate first entry without triggering full mmap initialization
        # Read directly instead of using _get_line_by_index
        with open(self.training_dataset_file, "rb") as f:
            f.seek(self.line_offsets[0])
            first_line = f.readline()
            first_entry = json.loads(first_line.decode("utf-8"))

        assert self.query_id_key in first_entry, (
            f"Key {self.query_id_key} not found in the first entry"
        )
        if not self.no_positive:
            assert self.positive_id_key in first_entry, (
                f"Key {self.positive_id_key} not found in the first entry"
            )
        assert self.negative_id_key in first_entry, (
            f"Key {self.negative_id_key} not found in the first entry"
        )

        self.multi_negatives = isinstance(first_entry[self.negative_id_key], list)
        total_negs = len(first_entry[self.negative_id_key]) if self.multi_negatives else 1
        # If n_neg is -1, use all negatives; otherwise validate requested amount
        if self.n_neg > 0:
            assert self.n_neg <= total_negs, (
                f"Only found {total_negs} negatives, cannot take {self.n_neg} negatives"
            )

    def __len__(self):
        return len(self.line_offsets)

    def get_cache_stats(self):
        """Get cache performance statistics for lazy loaders.

        Returns:
            dict: Dictionary with 'docs' and 'queries' cache statistics,
                  or None if not using lazy loading.

        Examples:
            Monitoring cache performance during training::

                dataset = TrainingDataset(
                    training_dataset_file="train.jsonl",
                    corpus=corpus,
                    lazy_load_text=True
                )

                # After some training
                stats = dataset.get_cache_stats()
                print(f"Document cache hit rate: {stats['docs']['hit_rate']:.2%}")
                print(f"Query cache hit rate: {stats['queries']['hit_rate']:.2%}")
        """
        if not self.lazy_load_text or self.precomputed:
            return None

        stats = {}
        if hasattr(self.docs, "get_cache_stats"):
            stats["docs"] = self.docs.get_cache_stats()
        if hasattr(self.queries, "get_cache_stats"):
            stats["queries"] = self.queries.get_cache_stats()

        return stats if stats else None

    def _teacher(self, query_id, doc_id):
        if query_id not in self.teacher:
            raise KeyError(f"Query ID {query_id} not found")
        if doc_id not in self.teacher[query_id]:
            raise KeyError(f"Doc ID {doc_id} not found for query {query_id}")
        return self.teacher[query_id][doc_id]

    def _precomputed_get(self, data: Dict[str, Any]):
        query_id = str(data[self.query_id_key])
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
        query_id = str(data[self.query_id_key])
        positive_id = str(data[self.positive_id_key]) if not self.no_positive else None
        negative_id = data[self.negative_id_key]
        # Cast negative_id to string if it's a single value
        if not isinstance(negative_id, list):
            negative_id = str(negative_id)
        else:
            # Cast all IDs in the list to strings
            negative_id = [str(nid) for nid in negative_id]

        query_text = self.queries[str(query_id)]
        positive_text = [self.docs[str(positive_id)]] if not self.no_positive else []

        if self.multi_negatives:
            if not self.lazy_load_text:
                negative_text = [self.docs[str(doc)] for doc in negative_id]
            else:
                negative_text = self.docs[negative_id]  # LazyTextLoader handles list
        else:
            negative_text = (
                [self.docs[str(negative_id)]] if not self.lazy_load_text else self.docs[negative_id]
            )

        return (
            query_id,
            query_text,
            positive_id,
            positive_text,
            negative_id,
            negative_text,
        )

    def __getitem__(self, idx: int):
        item = self._get_line_by_index(idx)

        query_id, query_text, positive_id, positive_text, negative_id, negative_text = self._get(
            item
        )

        if self.labels:
            positive_score = (
                [self._teacher(str(query_id), str(positive_id))] if not self.no_positive else []
            )
            if self.multi_negatives:
                negative_score = [self._teacher(str(query_id), str(doc)) for doc in negative_id]
            else:
                negative_score = [self._teacher(str(query_id), str(negative_id))]

            # Downsample negatives if more than needed (unless n_neg is -1 for all)
            n_negs_present = len(negative_id) if self.multi_negatives else 1
            if self.n_neg > 0 and n_negs_present > self.n_neg:
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
            if self.n_neg > 0 and n_total_negs > self.n_neg:
                if isinstance(negative_text, list):
                    negative_text = random.sample(negative_text, self.n_neg)
                else:
                    negative_text = [negative_text]
            return (query_text, positive_text + negative_text)
