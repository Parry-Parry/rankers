"""Dot-product (bi-encoder) ranking model.

This module implements a bi-encoder architecture where queries and documents are encoded
separately and their relevance is computed via dot product. This allows for efficient
retrieval through pre-computed document embeddings.
"""

import os
from copy import deepcopy

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from ..._optional import is_pyterrier_available
from ...train.loss.torch import batched_dot_product, cross_dot_product
from ..base import Ranker


class DotConfig(PretrainedConfig):
    """Configuration for Dot (bi-encoder) ranking model.

    This configuration class stores all hyperparameters for the dot-product ranking model,
    including pooling strategy, model architecture choices, and optional components.

    Args:
        model_name_or_path (str, optional): HuggingFace model identifier or path.
            Defaults to "bert-base-uncased".
        pooling_type (str, optional): Pooling strategy for embeddings. Options:
            "cls", "mean", "late_interaction", "none". Defaults to "cls".
        inbatch_loss (optional): Loss function for in-batch negatives. Defaults to None.
        model_tied (bool, optional): Whether to share weights between query and document
            encoders. Defaults to True.
        use_pooler (bool, optional): Whether to use an additional projection layer.
            Defaults to False.
        pooler_dim_in (int, optional): Input dimension for pooler. Defaults to 768.
        pooler_dim_out (int, optional): Output dimension for pooler. Defaults to 768.
        pooler_tied (bool, optional): Whether to share pooler weights between query
            and document. Defaults to True.
        **kwargs: Additional configuration parameters.

    Examples:
        Creating a custom configuration::

            config = DotConfig(
                model_name_or_path="bert-base-uncased",
                pooling_type="mean",
                use_pooler=True,
                pooler_dim_out=256
            )
    """

    model_type = "Dot"

    def __init__(
        self,
        model_name_or_path: str = "bert-base-uncased",
        pooling_type="cls",
        inbatch_loss=None,
        model_tied=True,
        use_pooler=False,
        pooler_dim_in=768,
        pooler_dim_out=768,
        pooler_tied=True,
        **kwargs,
    ):
        self.model_name_or_path = model_name_or_path
        self.pooling_type = pooling_type
        self.inbatch_loss = inbatch_loss
        self.model_tied = model_tied
        self.use_pooler = use_pooler
        self.pooler_dim_in = pooler_dim_in
        self.pooler_dim_out = pooler_dim_out
        self.pooler_tied = pooler_tied
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "bert-base-uncased",
        pooling_type="cls",
        inbatch_loss=None,
        model_tied=True,
        use_pooler=False,
        pooler_dim_in=768,
        pooler_dim_out=768,
        pooler_tied=True,
    ) -> "DotConfig":
        config = super().from_pretrained(model_name_or_path)
        config.model_name_or_path = model_name_or_path
        config.pooling_type = pooling_type
        config.inbatch_loss = inbatch_loss
        config.model_tied = model_tied
        config.use_pooler = use_pooler
        config.pooler_dim_in = pooler_dim_in
        config.pooler_dim_out = pooler_dim_out
        config.pooler_tied = pooler_tied
        return config


class Pooler(nn.Module):
    """Learnable projection layer for query and document embeddings.

    Projects embeddings to a lower-dimensional space. Can optionally use separate
    projections for queries and documents.

    Args:
        config (DotConfig): Configuration containing pooler dimensions and tying settings.

    Attributes:
        dense_q (nn.Linear): Projection layer for queries.
        dense_d (nn.Linear): Projection layer for documents (may be shared with dense_q).
    """

    def __init__(self, config):
        super().__init__()
        self.dense_q = nn.Linear(config.pooler_dim_in, config.pooler_dim_out)
        self.dense_d = (
            nn.Linear(config.pooler_dim_in, config.pooler_dim_out)
            if not config.pooler_tied
            else self.dense_q
        )

    @classmethod
    def from_pretrained(cls, model_name_or_path: str = "bert-base-uncased") -> "Pooler":
        """Load a pooler from a pretrained model directory.

        Args:
            model_name_or_path (str, optional): Path to model directory.
                Defaults to "bert-base-uncased".

        Returns:
            Pooler: Initialized pooler module.
        """
        config = DotConfig.from_pretrained(model_name_or_path)
        model = cls(config)
        return model

    def forward(self, hidden_states, d=False):
        """Project hidden states to output dimension.

        Args:
            hidden_states (torch.Tensor): Input embeddings to project.
            d (bool, optional): Whether to use document projection. Defaults to False.

        Returns:
            torch.Tensor: Projected embeddings.
        """
        return self.dense_d(hidden_states) if d else self.dense_q(hidden_states)


class Dot(Ranker):
    """Bi-encoder (dot-product) ranking model.

    The Dot model uses separate encoders for queries and documents, computing relevance
    scores via dot product of their embeddings. This architecture enables efficient
    retrieval by pre-computing and indexing document embeddings.

    The model supports various features:
    - Multiple pooling strategies (CLS, mean, late interaction)
    - Optional learnable projection layers
    - Tied or untied query/document encoders
    - In-batch negative training

    Args:
        model (PreTrainedModel): Query encoder (HuggingFace transformer).
        tokenizer (PreTrainedTokenizer): Tokenizer for text processing.
        config (DotConfig): Configuration object.
        model_d (PreTrainedModel, optional): Document encoder. If None, shares weights
            with query encoder when config.model_tied=True. Defaults to None.
        pooler (Pooler, optional): Projection layer. Created from config if None and
            config.use_pooler=True. Defaults to None.

    Attributes:
        model_type (str): Model identifier "Dot".
        architecture_class: AutoModel class for loading.
        config_class: DotConfig class.
        transformer_class: PyTerrier transformer class (set if pyterrier available).

    Examples:
        Basic usage::

            from rankers.modelling import Dot

            # Load pretrained model
            model = Dot.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

            # Custom configuration
            config = DotConfig(pooling_type="mean", use_pooler=True)
            model = Dot.from_pretrained("bert-base-uncased", config=config)

        Training with in-batch negatives::

            from rankers.train.loss import get_loss

            config = DotConfig(
                inbatch_loss=get_loss("cross_entropy"),
                model_tied=False  # Separate query/doc encoders
            )
            model = Dot.from_pretrained("bert-base-uncased", config=config)
    """

    model_type = "Dot"
    architecture_class = AutoModel
    config_class = DotConfig
    transformer_class = None

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: DotConfig,
        model_d: PreTrainedModel = None,
        pooler: Pooler = None,
    ):
        super().__init__(model, tokenizer, config)
        self.model = model
        self.tokenizer = tokenizer
        if model_d:
            self.model_d = model_d
        else:
            self.model_d = self.model if config.model_tied else deepcopy(self.model)
        self.pooling = {
            "mean": lambda x: x.mean(dim=1),
            "cls": lambda x: x[:, 0],
            "late_interaction": lambda x: x,
            "none": lambda x: x,
        }[config.pooling_type]
        self.pooling_type = config.pooling_type

        if config.use_pooler:
            self.pooler = Pooler(config) if pooler is None else pooler
        else:
            self.pooler = lambda x, _y=True: x

        if config.inbatch_loss is not None:
            from rankers.train.loss import LOSS_REGISTRY

            self.inbatch_loss_fn = LOSS_REGISTRY.get(config.inbatch_loss)
        else:
            self.inbatch_loss_fn = None

        if is_pyterrier_available():
            from ...pyterrier.dot import DotTransformer

            self.transformer_class = DotTransformer

    def prepare_outputs(self, query_reps, docs_batch_reps, labels=None, group_size=2):
        if group_size == -1:
            group_size = 1
        batch_size = query_reps.size(0)
        emb_q = query_reps.reshape(batch_size, 1, -1)
        emb_d = docs_batch_reps.reshape(batch_size, group_size, -1)
        if self.pooling_type == "late_interaction":
            pred = emb_q @ emb_d.permute(0, 2, 1)
            pred = pred.max(1).values
            pred = pred.sum(-1)
        else:
            pred = batched_dot_product(emb_q, emb_d)

        if self.config.inbatch_loss is not None:
            if self.pooling_type == "late_interaction":
                inbatch_d = emb_d[:, 0]
                inbatch_pred = emb_q @ inbatch_d.permute(0, 2, 1)
                inbatch_pred = inbatch_pred.max(1).values
                inbatch_pred = inbatch_pred.sum(-1)
            else:
                inbatch_d = emb_d[:, 0]
                inbatch_pred = cross_dot_product(emb_q.view(batch_size, -1), inbatch_d)
        else:
            inbatch_pred = None

        if labels is not None:
            labels = labels.reshape(batch_size, group_size)

        return pred, labels, inbatch_pred

    def _cls(self, x: torch.Tensor) -> torch.Tensor:
        return self.pooler(x[:, 0])

    def _mean(self, x: torch.Tensor) -> torch.Tensor:
        return self.pooler(x.mean(dim=1))

    def _encode_d(self, **text):
        return self.pooling(self.model_d(**text).last_hidden_state)

    def _encode_q(self, **text):
        return self.pooling(self.model(**text).last_hidden_state)

    def forward(self, loss=None, queries=None, docs_batch=None, labels=None, group_size=2):
        """Forward pass computing ranking loss.

        Encodes queries and documents separately, computes dot-product scores, and
        applies the loss function. Optionally includes in-batch negative loss.

        Args:
            loss (callable, optional): Loss function to apply.
            queries (dict, optional): Tokenized query inputs (input_ids, attention_mask, etc.).
            docs_batch (dict, optional): Tokenized document inputs.
            labels (torch.Tensor, optional): Relevance labels.
            group_size (int, optional): Number of documents per query in the batch.

        Returns:
            tuple: Contains:
                - loss_value (torch.Tensor): Total loss (main + in-batch).
                - to_log (dict): Metrics dictionary with 'loss' and 'inbatch_loss'.
                - pred (torch.Tensor): Predicted relevance scores.

        Note:
            All tensors are automatically moved to the appropriate device.
        """
        queries = (
            {k: v.to(self.model.device) for k, v in queries.items()}
            if queries is not None
            else None
        )
        docs_batch = (
            {k: v.to(self.model_d.device) for k, v in docs_batch.items()}
            if docs_batch is not None
            else None
        )
        labels = labels.to(self.model_d.device) if labels is not None else None

        query_reps = self._encode_q(**queries) if queries is not None else None
        docs_batch_reps = self._encode_d(**docs_batch) if docs_batch is not None else None

        pred, labels, inbatch_pred = self.prepare_outputs(
            query_reps, docs_batch_reps, labels, group_size=group_size
        )
        inbatch_loss = (
            self.inbatch_loss_fn(
                inbatch_pred, torch.eye(inbatch_pred.shape[0]).to(inbatch_pred.device)
            )
            if inbatch_pred is not None
            else 0.0
        )

        loss_value = loss(pred, labels) if labels is not None else loss(pred)
        if type(loss_value) is tuple:
            loss_value, to_log = loss_value
        else:
            to_log = {
                "loss": loss_value.item(),
            }
        to_log["inbatch_loss"] = (
            inbatch_loss.item() if isinstance(inbatch_loss, torch.Tensor) else inbatch_loss
        )
        loss_value += inbatch_loss
        return (loss_value, to_log, pred)

    def save_pretrained(self, model_dir, **kwargs):
        """Save the complete model to a directory.

        Saves all model components including query encoder, document encoder (if untied),
        pooler (if used), tokenizer, and configuration.

        Args:
            model_dir (str): Directory path to save the model.
            **kwargs: Additional arguments (currently unused).

        Note:
            The document encoder is saved to model_dir/model_d if not tied.
            The pooler is saved to model_dir/pooler if used.
        """
        self.model.save_pretrained(model_dir)
        if not self.config.model_tied:
            self.model_d.save_pretrained(model_dir + "/model_d")
        if self.config.use_pooler:
            self.pooler.save_pretrained(model_dir + "/pooler")
        self.config.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)

    def load_state_dict(self, model_dir):
        """Load state dict from a directory"""
        self.config = DotConfig.from_pretrained(model_dir)
        self.model.load_state_dict(self.architecture_class.from_pretrained(model_dir).state_dict())
        if not self.config.model_tied:
            self.model_d.load_state_dict(
                self.architecture_class.from_pretrained(model_dir + "/model_d").state_dict()
            )
        if self.config.use_pooler:
            self.pooler.load_state_dict(
                self.architecture_class.from_pretrained(model_dir + "/pooler").state_dict()
            )

    @classmethod
    def from_pretrained(cls, model_name_or_path, config=None, **kwargs) -> "Dot":
        """Load a pretrained Dot model.

        Loads from either a saved Dot model directory (with separate components) or
        a HuggingFace model hub identifier for initialization.

        Args:
            model_name_or_path (str): Path to model directory or HuggingFace model ID.
            config (DotConfig, optional): Pre-initialized configuration. If None,
                loads or creates from model_name_or_path. Defaults to None.
            **kwargs: Additional arguments passed to the underlying model loader.

        Returns:
            Dot: Initialized Dot model ready for inference or fine-tuning.

        Examples:
            Loading a saved Dot model::

                model = Dot.from_pretrained("./my_saved_model")

            Initializing from HuggingFace model::

                model = Dot.from_pretrained("bert-base-uncased")

            With custom configuration::

                config = DotConfig(pooling_type="mean")
                model = Dot.from_pretrained("bert-base-uncased", config=config)
        """
        if os.path.isdir(model_name_or_path):
            config = (
                cls.config_class.from_pretrained(model_name_or_path) if config is None else config
            )
            model = cls.architecture_class.from_pretrained(model_name_or_path, **kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            model_d = (
                None
                if config.model_tied
                else cls.architecture_class.from_pretrained(
                    model_name_or_path + "/model_d", **kwargs
                )
            )
            pooler = (
                None
                if not config.use_pooler
                else Pooler.from_pretrained(model_name_or_path + "/pooler")
            )

            return cls(model, tokenizer, config, model_d, pooler)
        config = cls.config_class(model_name_or_path, **kwargs) if config is None else config
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = cls.architecture_class.from_pretrained(model_name_or_path)
        return cls(model, tokenizer, config)


AutoConfig.register("Dot", DotConfig)
AutoModel.register(DotConfig, Dot)
