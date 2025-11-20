"""Base class for all neural ranking models.

This module provides the abstract Ranker class that all ranking model implementations
inherit from. It extends HuggingFace's PreTrainedModel and provides common functionality
for loading, saving, and using ranking models.
"""

from abc import abstractmethod

from transformers import AutoConfig, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


class Ranker(PreTrainedModel):
    """Base wrapper class for neural ranking models.

    All ranking model implementations (Dot, Cat, Sparse, etc.) inherit from this class.
    It provides common functionality for model loading, saving, adapter support, and
    PyTerrier integration.

    This class extends HuggingFace's PreTrainedModel, allowing rankers to integrate
    seamlessly with the transformers ecosystem.

    Args:
        model (PreTrainedModel): The underlying HuggingFace transformer model.
        tokenizer (PreTrainedTokenizer): Tokenizer for text processing.
        config (AutoConfig): Configuration object for the model.

    Attributes:
        model_type (str): Type identifier for the model family.
        architecture_class: HuggingFace model class (set by subclasses).
        config_class: Configuration class (set by subclasses).
        transformer_class: PyTerrier transformer class (set by subclasses).

    Examples:
        Subclasses should override prepare_outputs and set class attributes::

            class MyRanker(Ranker):
                architecture_class = AutoModel
                config_class = AutoConfig

                def prepare_outputs(self, logits, labels=None):
                    # Process model outputs for ranking
                    return logits, labels
    """

    model_type = "Ranker"
    architecture_class = None
    config_class = None
    transformer_class = None

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: AutoConfig,
    ):
        super().__init__(config)
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def prepare_outputs(self, logits, labels=None):
        """Prepare model outputs for loss computation.

        This abstract method must be implemented by subclasses to transform raw model
        logits into a format suitable for loss computation.

        Args:
            logits: Raw logits from the model forward pass.
            labels (optional): Ground truth labels for supervised learning.

        Returns:
            tuple: Processed predictions and labels.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("prepare_outputs must be implemented by subclasses")

    def add_adapter(self, config, adapter_name="adapter", **kwargs):
        """Add a parameter-efficient adapter to the model.

        Args:
            config: Adapter configuration.
            adapter_name (str, optional): Name for the adapter. Defaults to "adapter".
            **kwargs: Additional arguments passed to the underlying model.
        """
        self.model.add_adapter(config, adapter_name=adapter_name, **kwargs)

    def set_adapter(self, adapter_name="adapter"):
        """Set the active adapter for the model.

        Args:
            adapter_name (str, optional): Name of adapter to activate. Defaults to "adapter".
        """
        self.model.set_adapter(adapter_name)

    def disable_adapters(self):
        """Disable all adapters and use the base model."""
        self.model.disable_adapters()

    def enable_adapters(self):
        """Enable the currently set adapter."""
        self.model.enable_adapters()

    def get_adapter_state_dict(self, adapter_name="adapter"):
        """Get the state dictionary for a specific adapter.

        Args:
            adapter_name (str, optional): Name of the adapter. Defaults to "adapter".

        Returns:
            dict: State dictionary containing adapter parameters.
        """
        return self.model.get_adapter_state_dict(adapter_name)

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, config=None, **kwargs
    ) -> "Ranker":
        """Load a pretrained ranker model.

        Loads a model from a HuggingFace model hub identifier or local directory.
        Automatically loads the model, tokenizer, and configuration.

        Args:
            model_name_or_path (str): Model identifier or path to model directory.
            config (optional): Pre-initialized configuration. If None, loads from model_name_or_path.
            **kwargs: Additional arguments passed to the model's from_pretrained method.

        Returns:
            Ranker: Initialized ranker model.

        Examples:
            Loading a pretrained model::

                from rankers.modelling import Dot

                model = Dot.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        """
        config = (
            cls.config_class.from_pretrained(model_name_or_path)
            if config is None
            else config
        )
        model = cls.architecture_class.from_pretrained(model_name_or_path, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return cls(model, tokenizer, config)

    def save_pretrained(self, model_dir, **kwargs):
        """Save the model, tokenizer, and configuration to a directory.

        Args:
            model_dir (str): Directory path where model artifacts will be saved.
            **kwargs: Additional arguments (currently unused).

        Examples:
            Saving a trained model::

                model.save_pretrained("./my_ranker")
                # Can later be loaded with: Dot.from_pretrained("./my_ranker")
        """
        self.config.save_pretrained(model_dir)
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)

    def load_state_dict(self, model_dir):
        """Load model state dictionary from a directory.

        Args:
            model_dir (str): Directory containing the saved model.

        Returns:
            dict: Result of loading the state dictionary.
        """
        return self.model.load_state_dict(
            self.architecture_class.from_pretrained(model_dir).state_dict()
        )

    def to_pyterrier(self, batch_size=None):
        """Convert the ranker to a PyTerrier transformer.

        Creates a PyTerrier-compatible transformer that wraps this ranker for use
        in PyTerrier pipelines.

        Args:
            batch_size (int, optional): Batch size for inference. If None, uses default.

        Returns:
            Transformer: PyTerrier transformer wrapping this model.

        Raises:
            AssertionError: If transformer_class is not set or PyTerrier is not installed.

        Examples:
            Using a ranker in PyTerrier::

                import pyterrier as pt
                pt.init()

                ranker = model.to_pyterrier(batch_size=32)
                pipeline = pt.BatchRetrieve(index) >> ranker
        """
        assert self.transformer_class is not None, (
            "transformer_class must be set by subclasses, do you have pyterrier installed?"
        )
        return self.transformer_class.from_model(
            self, self.tokenizer, text_field="text", batch_size=batch_size
        )

    def forward(self, loss, sequences, labels=None):
        """Forward pass computing loss for a batch.

        Processes input sequences through the model and computes loss using the provided
        loss function.

        Args:
            loss: Loss function to apply to model outputs.
            sequences (dict): Dictionary of tokenized inputs (input_ids, attention_mask, etc.).
            labels (optional): Ground truth labels for supervised learning.

        Returns:
            tuple: Contains:
                - loss_value (torch.Tensor): Computed loss value.
                - to_log (dict): Dictionary of metrics to log.
                - pred: Model predictions after prepare_outputs.

        Note:
            Inputs and labels are automatically moved to the model's device.
        """
        sequences = {k: v.to(self.model.device) for k, v in sequences.items()}
        labels = labels.to(self.model.device) if labels is not None else None
        logits = self.model(**sequences).logits
        pred, labels = self.prepare_outputs(logits, labels)
        loss_value = loss(pred) if labels is None else loss(pred, labels)
        if type(loss_value) is tuple:
            loss_value, to_log = loss_value  # unpack tuple
        else:
            to_log = {"loss": loss_value.item()}
        return (loss_value, to_log, pred)
