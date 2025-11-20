"""Fixtures for creating actual BERT models for testing."""

from typing import Optional

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


def create_tiny_bert_model():
    """Create a tiny BERT model with minimal parameters for fast testing.

    Returns:
        BertModel with 2 layers, 64 hidden size, suitable for testing.
    """
    config = BertConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
    )
    return BertModel(config)


class ModelConfig:
    """Simple config class for TinyDotModel."""

    def __init__(self, group_size=2, hidden_size=64, vocab_size=1000):
        """Initialize config."""
        self.group_size = group_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

    def to_dict(self):
        """Convert config to dictionary."""
        return {
            "group_size": self.group_size,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
        }


class TinyDotModel(nn.Module):
    """Tiny dot-product ranking model using BERT for testing.

    Uses a minimal BERT encoder for efficient testing.
    """

    def __init__(self):
        """Initialize the model with tiny BERT."""
        super().__init__()
        self.encoder = create_tiny_bert_model()
        hidden_size = 64

        # Simple projection and scoring
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.doc_proj = nn.Linear(hidden_size, hidden_size)

        self.config = ModelConfig(
            group_size=2,
            hidden_size=hidden_size,
            vocab_size=1000,
        )

    def forward(
        self,
        query_ids: Optional[torch.Tensor] = None,
        doc_ids: Optional[torch.Tensor] = None,
        loss_fn=None,
        **kwargs
    ):
        """Forward pass.

        Args:
            query_ids: Query input IDs (batch_size, seq_len).
            doc_ids: Document input IDs (batch_size, num_docs, seq_len) or (batch_size, seq_len).
            loss_fn: Loss function to apply.
            **kwargs: Additional arguments.

        Returns:
            Dictionary with 'loss' key.
        """
        # Handle various input shapes
        batch_size = 1 if query_ids is None else query_ids.shape[0]

        # Get dummy scores for testing
        logits = torch.randn(batch_size, 1)

        output = {"logits": logits}

        if loss_fn is not None:
            # Compute loss
            labels = kwargs.get("labels", torch.ones_like(logits))
            output["loss"] = loss_fn(logits, labels).mean()

        return output

    def to_pyterrier(self, batch_size: int = 32):
        """Convert to PyTerrier compatible interface.

        Args:
            batch_size: Batch size for inference.

        Returns:
            A PyTerrier-compatible transformer.
        """
        return PyTerrierTransformer(self, batch_size)


class PyTerrierTransformer:
    """PyTerrier compatible transformer wrapper."""

    def __init__(self, model: TinyDotModel, batch_size: int = 32):
        """Initialize the transformer.

        Args:
            model: The ranking model.
            batch_size: Batch size for inference.
        """
        self.model = model
        self.batch_size = batch_size

    def transform(self, df):
        """Transform dataset to ranking frame.

        Args:
            df: Input DataFrame with columns [qid, docno, ...].

        Returns:
            DataFrame with ranking results.
        """
        import pandas as pd

        # Return input dataframe with scores
        result_df = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(df)

        if "score" not in result_df.columns:
            # Generate deterministic scores based on docno for reproducibility
            result_df["score"] = (
                result_df["docno"]
                .str.extract(r"(\d+)", expand=False)
                .astype(float)
            )

        return result_df[["qid", "docno", "score"]]
