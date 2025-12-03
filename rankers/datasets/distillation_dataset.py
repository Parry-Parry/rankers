"""Distillation dataset for training with automatically generated scores from ranking order."""

import json
import mmap
import os
import random
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from torch.utils.data import Dataset

from .corpus import Corpus
from .lazy_text_loader import LazyTextLoader

try:
    import ir_datasets as irds
except ImportError:
    irds = None


class DistillationDataset(Dataset):
    """Dataset for knowledge distillation training with auto-generated scores from rank order.

    Loads training data from JSONL files where documents in the ranked list are assumed to be
    ordered by relevance (highest to lowest). Scores are automatically generated based on rank
    position without requiring a separate teacher file.

    The dataset handles fork-safe file access for multiprocessing and provides
    efficient random access through memory-mapped files.

    Args:
        training_dataset_file (str): Path to JSONL training file.
        corpus (Union[Corpus, irds.Dataset]): Document corpus for text retrieval.
        group_size (int, optional): Number of documents per query to use.
            Use -1 to include all available documents from data. Defaults to -1.
        lazy_load_text (bool, optional): Load document text on-demand. Defaults to True.
        top_k_group (bool, optional): Select top-k documents from group by score. Defaults to False.
        precomputed (bool, optional): Whether text is pre-computed in dataset. Defaults to False.
        query_id_key (str, optional): JSON key for query ID. Defaults to "query_id".
        ranked_id_key (str, optional): JSON key for ranked doc IDs list (ordered by relevance).
            Defaults to "doc_id_b".
        score_function (callable, optional): Function to compute scores from rank position.
            Takes rank (0-based index) and returns score. Defaults to exponential decay: 2^(-rank/10).
        text_field (str, optional): Field name for document text. Defaults to "text".
        query_field (str, optional): Field name for query text. Defaults to "text".

    Examples:
        Basic distillation dataset with default exponential scoring::

            from rankers.datasets import DistillationDataset, Corpus

            corpus = Corpus.from_ir_datasets("msmarco-passage")
            dataset = DistillationDataset(
                training_dataset_file="ranked_results.jsonl",
                corpus=corpus
            )

        With fixed group size and custom scoring::

            def custom_score(rank):
                # RRF-style scoring
                return 1.0 / (rank + 60)

            dataset = DistillationDataset(
                training_dataset_file="ranked_results.jsonl",
                corpus=corpus,
                group_size=100,
                score_function=custom_score
            )

    Note:
        The JSONL file must not be compressed for memory-mapped access.
        Each line should be a JSON object with query ID and a ranked list of document IDs
        (best to worst ranking).
    """

    def __init__(
        self,
        training_dataset_file: str,
        corpus: Union[Corpus, "irds.Dataset"],
        group_size: int = -1,
        lazy_load_text: bool = True,
        top_k_group: bool = False,
        precomputed: bool = False,
        query_id_key: str = "query_id",
        ranked_id_key: str = "doc_id_b",
        score_function: callable = None,
        text_field: str = "text",
        query_field: str = "text",
    ) -> None:
        assert training_dataset_file.endswith("jsonl"), (
            "Training dataset should be a JSONL file and should not be compressed"
        )

        self.training_dataset_file = training_dataset_file
        self.corpus = corpus
        self.group_size = group_size
        self.lazy_load_text = lazy_load_text
        # If group_size is -1, use all documents available in data
        self.n_docs = self.group_size if self.group_size > 0 else -1
        self.top_k_group = top_k_group
        self.precomputed = precomputed
        self.query_id_key = query_id_key
        self.ranked_id_key = ranked_id_key
        self.text_field = text_field
        self.query_field = query_field

        # Default exponential decay scoring: 2^(-rank/10)
        if score_function is None:
            self.score_function = lambda rank: 2.0 ** (-rank / 10.0)
        else:
            self.score_function = score_function

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
        self._mm = (
            mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
            if use_mmap
            else None
        )
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
        state["score_function"] = None  # Functions can't be pickled
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Restore default score function if it was cleared
        if self.score_function is None:
            self.score_function = lambda rank: 2.0 ** (-rank / 10.0)

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
                    pd.DataFrame(self.corpus.docs_iter())
                    .set_index("doc_id")["text"]
                    .to_dict()
                )
                self.queries = (
                    pd.DataFrame(self.corpus.queries_iter())
                    .set_index("query_id")["text"]
                    .to_dict()
                )
            else:
                # Use lazy loading with caching for both docs and queries
                self.docs = LazyTextLoader(self.corpus, cache_size=10000, mode="docs")
                self.queries = LazyTextLoader(
                    self.corpus, cache_size=5000, mode="queries"
                )

        # Validate first entry without triggering full mmap initialization
        with open(self.training_dataset_file, "rb") as f:
            f.seek(self.line_offsets[0])
            first_line = f.readline()
            first_entry = json.loads(first_line.decode("utf-8"))

        assert self.query_id_key in first_entry, (
            f"Key {self.query_id_key} not found in the first entry"
        )
        assert self.ranked_id_key in first_entry, (
            f"Key {self.ranked_id_key} not found in the first entry"
        )

        ranked_docs = first_entry[self.ranked_id_key]
        if not isinstance(ranked_docs, list):
            raise ValueError(
                f"DistillationDataset requires {self.ranked_id_key} to be a list of ranked documents"
            )

        total_docs = len(ranked_docs)
        # If n_docs is -1, use all documents; otherwise validate requested amount
        if self.n_docs > 0:
            assert self.n_docs <= total_docs, (
                f"Only found {total_docs} ranked documents, cannot take {self.n_docs} documents"
            )

    def __len__(self):
        return len(self.line_offsets)

    def get_cache_stats(self):
        """Get cache performance statistics for lazy loaders.

        Returns:
            dict: Dictionary with 'docs' and 'queries' cache statistics,
                  or None if not using lazy loading.
        """
        if not self.lazy_load_text or self.precomputed:
            return None

        stats = {}
        if hasattr(self.docs, "get_cache_stats"):
            stats["docs"] = self.docs.get_cache_stats()
        if hasattr(self.queries, "get_cache_stats"):
            stats["queries"] = self.queries.get_cache_stats()

        return stats if stats else None

    def _precomputed_get(self, data: Dict[str, Any]):
        query_id = str(data[self.query_id_key])
        query_text = data[self.query_field]
        ranked_id = data[self.ranked_id_key]
        ranked_text = data[f"{self.ranked_id_key}_text"]
        return (
            query_id,
            query_text,
            ranked_id,
            ranked_text,
        )

    def _standard_get(self, data: Dict[str, Any]):
        query_id = str(data[self.query_id_key])
        ranked_id = data[self.ranked_id_key]
        # Cast all IDs to strings
        ranked_id = [str(rid) for rid in ranked_id]

        query_text = self.queries[str(query_id)]

        if not self.lazy_load_text:
            ranked_text = [self.docs[str(doc)] for doc in ranked_id]
        else:
            ranked_text = self.docs[ranked_id]  # LazyTextLoader handles list

        return (
            query_id,
            query_text,
            ranked_id,
            ranked_text,
        )

    def __getitem__(self, idx: int):
        item = self._get_line_by_index(idx)

        query_id, query_text, ranked_id, ranked_text = self._get(item)

        # Generate scores based on rank position
        scores = [self.score_function(rank) for rank in range(len(ranked_text))]

        # Downsample documents if more than needed (unless group_size is -1 for all)
        n_docs_present = len(ranked_id)
        if self.n_docs > 0 and n_docs_present > self.n_docs:
            if self.top_k_group:
                # Keep top-k by score
                texts_scores = list(zip(ranked_text, scores))
                texts_scores.sort(key=lambda x: x[1], reverse=True)
                texts, scores = zip(*texts_scores[: self.group_size])
                return (query_text, list(texts), list(scores))
            else:
                # Randomly sample documents
                doc_pairs = list(zip(ranked_text, scores))
                sampled = random.sample(doc_pairs, self.n_docs)
                s_text, s_scores = zip(*sampled)
                return (query_text, list(s_text), list(s_scores))

        # Return all documents with their scores
        return (query_text, ranked_text, scores)
