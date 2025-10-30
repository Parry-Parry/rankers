"""Dataset classes for training and evaluating neural rankers.

This module provides PyTorch Dataset implementations for ranking tasks, including
support for training with negatives, test evaluation, and efficient lazy loading
of large corpora.
"""

import os
import json
import mmap
import random
from functools import cached_property
from typing import Union, Iterable, List, Dict, Any, Tuple

import pandas as pd
from torch.utils.data import Dataset

from .._util import load_json, initialise_irds_eval, read_trec
from .._optional import is_ir_datasets_available
from .corpus import Corpus

if is_ir_datasets_available():
    import ir_datasets as irds


class LazyTextLoader:
    """Lazy loader for document/query text with LRU caching.

    Provides on-demand loading of text without loading the entire corpus into memory.
    Uses LRU caching to avoid repeated I/O for frequently accessed items, which is
    critical for training performance when documents appear in multiple batches.

    Args:
        corpus (Union[irds.Dataset, Corpus]): Corpus containing documents/queries.
        cache_size (int, optional): Maximum number of items to cache. Defaults to 10000.
        mode (str, optional): Type of text to load - 'docs' or 'queries'. Defaults to 'docs'.

    Attributes:
        cache_size (int): Maximum cache size.
        mode (str): Loading mode.

    Examples:
        Loading documents on demand with caching::

            import ir_datasets as irds
            from rankers.datasets import LazyTextLoader

            dataset = irds.load("msmarco-passage/dev")
            loader = LazyTextLoader(dataset, cache_size=20000)

            # Load single document (cache miss)
            text = loader["doc123"]

            # Load same document again (cache hit - no I/O!)
            text = loader["doc123"]

            # Load multiple documents
            texts = loader[["doc1", "doc2", "doc3"]]

            # Check cache performance
            stats = loader.get_cache_stats()
            print(f"Cache hit rate: {stats['hit_rate']:.2%}")

    Note:
        Cache size should be tuned based on available memory and corpus size.
        For training with batch_size=32 and group_size=8, ~10k cache covers
        several hundred batches worth of documents.
    """

    def __init__(
        self,
        corpus: Union["irds.Dataset", Corpus],
        cache_size: int = 10000,
        mode: str = 'docs'
    ) -> None:
        from functools import lru_cache

        self.mode = mode
        self.cache_size = cache_size

        if mode == 'docs':
            self.store = corpus.doc_store()
        elif mode == 'queries':
            # For queries, try to build from iterator since most corpora don't have query store
            self._query_cache = {}
            try:
                for q in corpus.queries_iter():
                    qid = q.get('query_id') or q.get('qid')
                    text = q.get('text', '')
                    if qid:
                        self._query_cache[str(qid)] = text
            except Exception:
                # Fallback to docstore if available
                self.store = getattr(corpus, 'doc_store', lambda: None)()
        else:
            self.store = corpus.doc_store()

        # Create cached retrieval function with LRU cache
        self._get_cached = lru_cache(maxsize=cache_size)(self._get_single)

    def _get_single(self, item_id: str) -> str:
        """Retrieve single item text (will be cached by lru_cache).

        Args:
            item_id (str): Item ID to retrieve.

        Returns:
            str: Item text.
        """
        # If using query cache, check it first
        if self.mode == 'queries' and hasattr(self, '_query_cache'):
            return self._query_cache.get(item_id, '')

        # Otherwise use the store
        try:
            result = self.store.get(item_id)
            # Handle both object with .text attribute and direct string return
            return result.text if hasattr(result, 'text') else result
        except (AttributeError, KeyError):
            return ''

    def __getitem__(self, item_id):
        """Load item text by ID with caching.

        Args:
            item_id (Union[str, List[str]]): Item ID or list of IDs.

        Returns:
            Union[str, List[str]]: Item text or list of texts.
        """
        if isinstance(item_id, list):
            return [self._get_cached(str(i)) for i in item_id]
        return self._get_cached(str(item_id))

    def __call__(self, id_):
        """Callable interface for loading items.

        Args:
            id_: Item ID(s) to load.

        Returns:
            Item text(s).
        """
        return self[id_]

    def get_cache_stats(self):
        """Get cache performance statistics.

        Returns:
            dict: Cache statistics including hits, misses, size, and hit rate.
        """
        info = self._get_cached.cache_info()
        total = info.hits + info.misses
        hit_rate = info.hits / total if total > 0 else 0.0
        return {
            'hits': info.hits,
            'misses': info.misses,
            'size': info.currsize,
            'maxsize': info.maxsize,
            'hit_rate': hit_rate
        }

    def clear_cache(self):
        """Clear the LRU cache."""
        self._get_cached.cache_clear()


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
            Defaults to 2.
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
        Basic training dataset::

            from rankers.datasets import TrainingDataset, Corpus

            corpus = Corpus.from_ir_datasets("msmarco-passage")
            dataset = TrainingDataset(
                training_dataset_file="train.jsonl",
                corpus=corpus,
                group_size=8  # 1 positive + 7 negatives
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
        assert training_dataset_file.endswith("jsonl"), (
            "Training dataset should be a JSONL file and should not be compressed"
        )

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
                self.docs = LazyTextLoader(self.corpus, cache_size=10000, mode='docs')
                self.queries = LazyTextLoader(self.corpus, cache_size=5000, mode='queries')

        if self.teacher_file:
            self.teacher = load_json(self.teacher_file)
            self.labels = True
        else:
            self.labels = False

        # Validate first entry without triggering full mmap initialization
        # Read directly instead of using _get_line_by_index
        with open(self.training_dataset_file, 'rb') as f:
            f.seek(self.line_offsets[0])
            first_line = f.readline()
            first_entry = json.loads(first_line.decode('utf-8'))

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
        total_negs = (
            len(first_entry[self.negative_id_key]) if self.multi_negatives else 1
        )
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
        if hasattr(self.docs, 'get_cache_stats'):
            stats['docs'] = self.docs.get_cache_stats()
        if hasattr(self.queries, 'get_cache_stats'):
            stats['queries'] = self.queries.get_cache_stats()

        return stats if stats else None

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
        positive_text = (
            data[f"{self.positive_id_key}_text"] if not self.no_positive else None
        )
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
                negative_score = [
                    self._teacher(str(query_id), str(doc)) for doc in negative_id
                ]
            else:
                negative_score = [self._teacher(str(query_id), str(negative_id))]

            # Downsample negatives if more than needed
            n_negs_present = len(negative_id) if self.multi_negatives else 1
            if n_negs_present > self.n_neg:
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
            if n_total_negs > self.n_neg:
                if isinstance(negative_text, list):
                    negative_text = random.sample(negative_text, self.n_neg)
                else:
                    negative_text = [negative_text]
            return (query_text, positive_text + negative_text)


class TestDataset(Dataset):
    """Dataset for evaluating ranker models on test data.

    Loads test/evaluation data in TREC format with query-document pairs and relevance
    judgments. Integrates with ir_datasets for standard IR benchmark evaluation.

    Args:
        data (pd.DataFrame): DataFrame in TREC format with 'qid', 'docno', 'score' columns.
        corpus (Union[Corpus, irds.Dataset]): Corpus containing documents and queries.
        lazy_load_text (bool, optional): Load document text on-demand. Defaults to True.

    Attributes:
        data (pd.DataFrame): Test data with enriched query and document text.
        queries (dict): Mapping of query IDs to query text.
        docs: Document loader (LazyTextLoader or dict).
        qrels (pd.DataFrame): Relevance judgments from corpus.

    Examples:
        Loading from TREC file::

            import ir_datasets as irds
            from rankers.datasets import TestDataset

            dataset = irds.load("msmarco-passage/dev")
            test_data = TestDataset.from_trec("run.trec", dataset)

        Loading from ir_datasets::

            test_data = TestDataset.from_irds(dataset)

        Evaluating a model::

            from rankers.modelling import Dot

            model = Dot.from_pretrained("model-path")
            # Use with DataLoader for batch evaluation
            results = model.evaluate(test_data)
    """

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
            self.queries = (
                pd.DataFrame(self.corpus.queries_iter())
                .set_index("query_id")["text"]
                .to_dict()
            )
        else:
            # Use lazy loading with caching
            self.docs = LazyTextLoader(self.corpus, cache_size=10000, mode='docs')
            self.queries = LazyTextLoader(self.corpus, cache_size=2000, mode='queries')

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
    Validation helper that:
      (1) Constructs qrels from positives only (others are assumed nonrelevant).
      (2) Exposes a cached TREC-style ranking DataFrame via `.data`, built from the
          training-format JSONL, with columns: qid, query, docno, text, label.
          - label = 1 for positives, 0 for negatives (useful for accuracy checks).

    Attributes
    ----------
    qrels : pd.DataFrame
        Columns: ["qid", "docno", "relevance"] with positives only.
    data : pd.DataFrame (cached property via @cached + @property)
        Columns: ["qid", "query", "docno", "text", "label"], built from training JSONL.

    Notes
    -----
    - `corpus` is required to resolve query and document text.
    - Any document not listed in `qrels` is implicitly nonrelevant.
    """

    def __init__(
        self,
        training_dataset_file: str,
        corpus: Union[Corpus, "irds.Dataset"],
        query_id_key: str = "query_id",
        positive_id_key: str = "doc_id_a",
        negative_id_key: str = "doc_id_b",
        lazy_load_text: bool = True,
        relevance_label: int = 1,
        include_positive: bool = True,
        dedupe_qrels: bool = True,
    ) -> None:
        super().__init__()
        assert training_dataset_file.endswith("jsonl"), (
            "validation expects an uncompressed JSONL"
        )
        self.training_dataset_file = training_dataset_file
        self.corpus = corpus
        self.query_id_key = query_id_key
        self.positive_id_key = positive_id_key
        self.negative_id_key = negative_id_key
        self.lazy_load_text = lazy_load_text
        self.relevance_label = relevance_label
        self.include_positive = include_positive
        self.dedupe_qrels = dedupe_qrels

        # Text lookup
        if not self.lazy_load_text:
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
            # Use lazy loading with caching
            self.docs = LazyTextLoader(self.corpus, cache_size=10000, mode='docs')
            self.queries = LazyTextLoader(self.corpus, cache_size=2000, mode='queries')

        # Build qrels once (positives only)
        self.qrels = self._build_qrels()

    def _build_qrels(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        with open(self.training_dataset_file, "rb") as f:
            for raw in f:
                if not raw.strip():
                    continue
                rec = json.loads(raw.decode("utf-8"))

                if self.query_id_key not in rec:
                    raise KeyError(f"Key {self.query_id_key} not found in record")
                qid = rec[self.query_id_key]

                if not self.include_positive or self.positive_id_key not in rec:
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
                                "qid": qid,
                                "docno": str(d),
                                "relevance": self.relevance_label,
                            }
                        )
                else:
                    rows.append(
                        {
                            "qid": qid,
                            "docno": str(pos),
                            "relevance": self.relevance_label,
                        }
                    )

        df = pd.DataFrame(rows, columns=["qid", "docno", "relevance"])
        if self.dedupe_qrels and not df.empty:
            df = df.drop_duplicates(subset=["qid", "docno"], keep="first").reset_index(
                drop=True
            )
        return df

    @cached_property
    def data(self) -> pd.DataFrame:
        """
        Cached TREC-style ranking file derived from the training JSONL.
        Columns:
          - qid   : query identifier (from training JSONL)
          - query : query text (from corpus)
          - docno : document identifier (positive and negatives)
          - text  : document text (from corpus)
          - label : 1 for positive docs, 0 for negatives
        """
        rows: List[Dict[str, Any]] = []
        with open(self.training_dataset_file, "rb") as f:
            for raw in f:
                if not raw.strip():
                    continue
                rec = json.loads(raw.decode("utf-8"))

                if self.query_id_key not in rec or self.negative_id_key not in rec:
                    raise KeyError(
                        f"Record missing required keys: {self.query_id_key} / {self.negative_id_key}"
                    )

                qid = rec[self.query_id_key]
                qtext = self.queries[str(qid)]

                # Positive (optional)
                if self.include_positive and (self.positive_id_key in rec):
                    pos = rec[self.positive_id_key]
                    if pos is not None:
                        if isinstance(pos, list):
                            for d in pos:
                                if d is None:
                                    continue
                                rows.append(
                                    {
                                        "qid": qid,
                                        "query": qtext,
                                        "docno": str(d),
                                        "text": self.docs[str(d)],
                                        "label": 1,
                                    }
                                )
                        else:
                            rows.append(
                                {
                                    "qid": qid,
                                    "query": qtext,
                                    "docno": str(pos),
                                    "text": self.docs[str(pos)],
                                    "label": 1,
                                }
                            )

                # Negatives (single or list)
                neg = rec[self.negative_id_key]
                if isinstance(neg, list):
                    for d in neg:
                        rows.append(
                            {
                                "qid": qid,
                                "query": qtext,
                                "docno": str(d),
                                "text": self.docs[str(d)],
                                "label": 0,
                            }
                        )
                else:
                    rows.append(
                        {
                            "qid": qid,
                            "query": qtext,
                            "docno": str(neg),
                            "text": self.docs[str(neg)],
                            "label": 0,
                        }
                    )

        return pd.DataFrame(rows, columns=["qid", "query", "docno", "text", "label"])

    def __len__(self):
        # Number of rows in ranking DataFrame
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        return row["qid"], row["query"], row["docno"], row["text"], row["label"]
