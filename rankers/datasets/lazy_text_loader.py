"""Lazy text loader with LRU caching for efficient corpus access."""

from typing import Union


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
        corpus,
        cache_size: int = 10000,
        mode: str = "docs",
    ) -> None:
        from functools import lru_cache

        self.mode = mode
        self.cache_size = cache_size

        if mode == "docs":
            self.store = corpus.docs_store()
        elif mode == "queries":
            # For queries, try to build from iterator since most corpora don't have query store
            self._query_cache = {}
            for q in corpus.queries_iter():
                # Handle both Corpus (returns dicts) and ir_datasets (returns namedtuples)
                if hasattr(q, '_asdict'):
                    # ir_datasets namedtuple - convert to dict
                    q = q._asdict()

                # Now q is always a dict
                qid = q.get("query_id") or q.get("qid")
                text = q.get("text")

                if not qid:
                    raise KeyError("Query record missing both 'query_id' and 'qid' fields")
                if not text:
                    raise KeyError(f"Query {qid} missing required 'text' field")
                self._query_cache[str(qid)] = text
        else:
            self.store = corpus.docs_store()

        # Create cached retrieval function with LRU cache
        self._get_cached = lru_cache(maxsize=cache_size)(self._get_single)

    def _get_single(self, item_id: str) -> str:
        """Retrieve single item text (will be cached by lru_cache).

        Args:
            item_id (str): Item ID to retrieve.

        Returns:
            str: Item text.

        Raises:
            KeyError: If item ID is not found in corpus.
        """
        item_id = str(item_id)
        # If using query cache, check it first
        if self.mode == "queries" and hasattr(self, "_query_cache"):
            if item_id not in self._query_cache:
                raise KeyError(f"Query ID {item_id} not found in corpus")
            return self._query_cache[item_id]

        # Otherwise use the store
        if self.store is None:
            raise KeyError(f"Item ID {item_id} not found: no corpus store available")
        result = self.store.get(item_id)
        if result is None:
            raise KeyError(f"Item ID {item_id} not found in corpus")
        # Handle both object with .text attribute and direct string return
        return result.text if hasattr(result, "text") else result

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
            "hits": info.hits,
            "misses": info.misses,
            "size": info.currsize,
            "maxsize": info.maxsize,
            "hit_rate": hit_rate,
        }

    def clear_cache(self):
        """Clear the LRU cache."""
        self._get_cached.cache_clear()
