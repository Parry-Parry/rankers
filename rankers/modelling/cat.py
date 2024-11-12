import pyterrier as pt
from transformers import PreTrainedModel, PretrainedConfig, PreTrainedTokenizer, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import torch
import torch.nn.functional as F

class CatConfig(PretrainedConfig):
    model_type = 'Cat'

class Cat(PreTrainedModel):
    """Wrapper for Cat Model
    
    Parameters
    ----------
    model : PreTrainedModel
        the underlying HF model
    config : AutoConfig
        the configuration for the model
    """
    model_type = 'Cat'
    architecture_class = AutoModelForSequenceClassification
    config_class = CatConfig
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

        from .pyterrier.cat import CatTransformer
        self.transformer_class = CatTransformer
    
    def prepare_outputs(self, logits, labels=None):
        """Prepare outputs"""
        return F.log_softmax(logits.reshape(-1, self.config.group_size, 2), dim=-1)[:, :, 1], labels.view(-1, self.config.group_size) if labels is not None else None

    def forward(self, loss, sequences, labels=None):
        """Compute the loss given (pairs, labels)"""
        sequences = {k: v.to(self.model.device) for k, v in sequences.items()}
        labels = labels.to(self.model.device) if labels is not None else None
        logits = self.model(**sequences).logits
        pred, labels = self.prepare_outputs(logits, labels)
        loss_value = loss(pred) if labels is None else loss(pred, labels)
        return (loss_value, pred)

    def save_pretrained(self, model_dir, **kwargs):
        """Save model"""
        self.config.save_pretrained(model_dir)
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
    
    def load_state_dict(self, model_dir):
        """Load state dict from a directory"""
        return self.model.load_state_dict(self.architecture_class.from_pretrained(model_dir).state_dict())

    def to_pyterrier(self) -> "pt.Transformer":
        return self.transformer_class.from_model(self.model, self.tokenizer, text_field='text')

    @classmethod
    def from_pretrained(cls, model_name_or_path : str, num_labels=2, config=None, **kwargs) -> "Cat":
        """Load model from a directory"""
        config = cls.config_class.from_pretrained(model_name_or_path, num_labels=num_labels) if config is None else config
        model = cls.architecture_class.from_pretrained(model_name_or_path, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return cls(model, tokenizer, config)

AutoConfig.register("Cat", CatConfig)
AutoModelForSequenceClassification.register(CatConfig, Cat)