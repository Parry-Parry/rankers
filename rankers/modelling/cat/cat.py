"""Concatenation (cross-encoder) ranking model.

This module implements a cross-encoder architecture where query and document are
concatenated and processed together through a single transformer, producing a
direct relevance score. This provides higher accuracy than bi-encoders at the
cost of computational efficiency.
"""

import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from ..._optional import is_pyterrier_available
from ..base import Ranker


class CatConfig(PretrainedConfig):
    """Configuration for Cat (cross-encoder) ranking model.

    This configuration extends HuggingFace's PretrainedConfig for cross-encoder models.
    Cross-encoders concatenate query and document text and process them jointly.

    Attributes:
        model_type (str): Model type identifier "Cat".

    Examples:
        Creating a Cat configuration::

            config = CatConfig()
            # Configuration inherits from the base model's config
    """

    model_type = "Cat"


class Cat(Ranker):
    """Cross-encoder (concatenation) ranking model.

    The Cat model concatenates query and document text and processes them together
    through a single transformer encoder, using a classification head to predict
    relevance. This joint encoding allows for complex query-document interactions
    but requires scoring all query-document pairs at query time.

    This architecture is typically used for reranking a small set of candidates
    retrieved by a more efficient first-stage ranker.

    Args:
        model (PreTrainedModel): HuggingFace sequence classification model.
        tokenizer (PreTrainedTokenizer): Tokenizer for text processing.
        config (AutoConfig): Configuration object.

    Attributes:
        model_type (str): Model identifier "Cat".
        architecture_class: AutoModelForSequenceClassification class.
        config_class: CatConfig class.
        transformer_class: PyTerrier transformer class (set if pyterrier available).

    Examples:
        Basic usage::

            from rankers.modelling import Cat

            # Load pretrained cross-encoder
            model = Cat.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

            # Initialize from base model
            model = Cat.from_pretrained("bert-base-uncased", num_labels=2)

        Reranking pipeline::

            # Typically used as second-stage reranker
            first_stage = Dot.from_pretrained("retriever-model")
            reranker = Cat.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

            # In PyTerrier: bm25 >> first_stage >> reranker.to_pyterrier()

    Note:
        Cat models score query-document pairs independently and cannot pre-compute
        document representations. Use Dot models for efficient large-scale retrieval.
    """

    model_type = "Cat"
    architecture_class = AutoModelForSequenceClassification
    config_class = CatConfig
    transformer_class = None

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: AutoConfig,
    ):
        super().__init__(model, tokenizer, config)

        if is_pyterrier_available():
            from ...pyterrier.cat import CatTransformer

            self.transformer_class = CatTransformer

    @classmethod
    def from_pretrained(cls, model_name_or_path, config=None, num_labels: int = 2, **kwargs):
        """Load a pretrained Cat model.

        Args:
            model_name_or_path (str): HuggingFace model ID or path to model directory.
            config (CatConfig, optional): Pre-initialized configuration. Defaults to None.
            num_labels (int, optional): Number of classification labels. Defaults to 2.
            **kwargs: Additional arguments passed to the model loader.

        Returns:
            Cat: Initialized Cat model ready for inference or fine-tuning.

        Examples:
            Loading a pretrained cross-encoder::

                model = Cat.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

            Initializing from a base transformer::

                model = Cat.from_pretrained("bert-base-uncased", num_labels=2)
        """
        return super().from_pretrained(model_name_or_path, config, num_labels=num_labels, **kwargs)

    def prepare_outputs(self, logits, labels=None, group_size=2):
        """Prepare model outputs for loss computation.

        Applies log-softmax to classification logits and extracts relevance scores.
        Reshapes outputs to match the expected format for ranking loss functions.

        Args:
            logits (torch.Tensor): Raw classification logits from the model.
            labels (torch.Tensor, optional): Ground truth relevance labels.

        Returns:
            tuple: Contains:
                - scores (torch.Tensor): Log-softmax scores for positive class.
                - labels (torch.Tensor): Reshaped labels (if provided).

        Note:
            Assumes binary classification with shape (batch_size * group_size, 2).
            Extracts positive class probabilities (index 1).
        """
        if group_size == -1:
            group_size = 1
        return F.log_softmax(logits.reshape(-1, group_size, 2), dim=-1)[:, :, 1], (
            labels.view(-1, group_size) if labels is not None else None
        )


AutoConfig.register("Cat", CatConfig)
AutoModelForSequenceClassification.register(CatConfig, Cat)
