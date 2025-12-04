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
        # Add required fields for HF Trainer compatibility
        self.eos_token_id = 102
        self.bos_token_id = 101
        self.pad_token_id = 0
        self.model_type = "tiny_dot"

    def to_dict(self):
        """Convert config to dictionary."""
        return {
            "group_size": self.group_size,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "eos_token_id": self.eos_token_id,
            "bos_token_id": self.bos_token_id,
            "pad_token_id": self.pad_token_id,
            "model_type": self.model_type,
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

    def forward(self, loss_fn=None, **kwargs):
        """Forward pass.

        Args:
            loss_fn: Loss function to apply.
            **kwargs: Additional arguments including query_ids, doc_ids,
                     labels.

        Returns:
            Dictionary with 'loss' key.
        """
        # Get batch size and device from any available input
        query_ids = kwargs.get("query_ids")
        doc_ids = kwargs.get("doc_ids")
        batch_size = 1
        device = self.encoder.device  # Get device from model
        if query_ids is not None:
            batch_size = query_ids.shape[0]
            device = query_ids.device
        elif doc_ids is not None:
            batch_size = doc_ids.shape[0]
            device = doc_ids.device

        # Create logits by using model projections (ensures gradients flow)
        # Use dummy embeddings to test gradient flow
        dummy_embedding = torch.randn(batch_size, 64, device=device, requires_grad=True)
        # Apply projection layers to ensure parameters are used
        proj_output = self.query_proj(dummy_embedding)
        logits = proj_output[:, :2]  # Shape (batch_size, 2)

        output = {"logits": logits, "to_log": {}}

        if loss_fn is not None:
            # Compute loss - provide proper shape labels
            labels = kwargs.get("labels")
            if labels is None:
                # Create dummy labels matching group_size (1 positive + 1
                # negative)
                labels = torch.zeros(batch_size, 2, device=logits.device)
                labels[:, 0] = 1  # First doc is relevant
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
                result_df["docno"].str.extract(r"(\d+)", expand=False).astype(float)
            )

        return result_df[["qid", "docno", "score"]]
