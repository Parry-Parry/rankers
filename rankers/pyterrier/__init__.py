"""PyTerrier integration for neural ranking models.

This module provides transformer classes that wrap rankers models for use in PyTerrier
pipelines. These transformers follow PyTerrier's API and can be seamlessly integrated
into retrieval pipelines.

Available Transformers:
    - **DotTransformer**: Bi-encoder (dot-product) model transformer
    - **CatTransformer**: Cross-encoder (concatenation) model transformer
    - **SparseTransformer**: Sparse neural model transformer
    - **Seq2SeqTransformer**: Sequence-to-sequence model transformer

All transformers implement PyTerrier's Transformer interface and can be used in
standard PyTerrier pipeline operations (>>, %, etc.).

Examples:
    Using a dot-product ranker in a PyTerrier pipeline::

        import pyterrier as pt
        from rankers.pyterrier import DotTransformer

        pt.init()

        # Create retrieval pipeline
        bm25 = pt.BatchRetrieve(index, wmodel="BM25")
        ranker = DotTransformer("sentence-transformers/all-MiniLM-L6-v2")

        pipeline = bm25 >> ranker
        results = pipeline.search("information retrieval")

Note:
    PyTerrier must be installed to use this module. Install with: pip install python-terrier
"""

from . import cat, dot, sparse
