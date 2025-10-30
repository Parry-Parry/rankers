"""Neural ranking models.

This module provides implementations of various neural ranking architectures:

- **base**: Base Ranker class with common functionality
- **dot**: Dot-product (bi-encoder) models for efficient dense retrieval
- **cat**: Concatenation (cross-encoder) models for reranking
- **sparse**: Sparse neural models (e.g., SPLADE)
- **bge**: BGE (BAAI General Embedding) model implementations
- **seq2seq**: Sequence-to-sequence generative ranking models

All models inherit from the base Ranker class and follow HuggingFace's
PreTrainedModel interface for easy integration with the transformers ecosystem.
"""

from . import base, cat, sparse, dot, bge, seq2seq
