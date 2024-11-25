from transformers import PreTrainedModel, PreTrainedTokenizer, AutoConfig, AutoTokenizer
from abc import abstractmethod


class Ranker(PreTrainedModel):
    """Wrapper for a ranker model

    Parameters
    ----------
    model : PreTrainedModel
        the underlying HF model
    config : AutoConfig
        the configuration for the model
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
        raise NotImplementedError("prepare_outputs must be implemented by subclasses")

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, num_labels=2, config=None, **kwargs
    ) -> "Cat":
        """Load model from a directory"""
        config = (
            cls.config_class.from_pretrained(model_name_or_path, num_labels=num_labels)
            if config is None
            else config
        )
        model = cls.architecture_class.from_pretrained(model_name_or_path, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return cls(model, tokenizer, config)

    def save_pretrained(self, model_dir, **kwargs):
        """Save model"""
        self.config.save_pretrained(model_dir)
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)

    def load_state_dict(self, model_dir):
        """Load state dict from a directory"""
        return self.model.load_state_dict(
            self.architecture_class.from_pretrained(model_dir).state_dict()
        )

    def to_pyterrier(self, batch_size=None):
        assert (
            self.transformer_class is not None
        ), "transformer_class must be set by subclasses, do you have pyterrier installed?"
        return self.transformer_class.from_model(
            self.model, self.tokenizer, text_field="text", batch_size=batch_size
        )

    def forward(self, loss, sequences, labels=None):
        """Compute the loss given (pairs, labels)"""
        sequences = {k: v.to(self.model.device) for k, v in sequences.items()}
        labels = labels.to(self.model.device) if labels is not None else None
        logits = self.model(**sequences).logits
        pred, labels = self.prepare_outputs(logits, labels)
        loss_value = loss(pred) if labels is None else loss(pred, labels)
        return (loss_value, pred)