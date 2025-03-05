from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    PreTrainedTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
)
import torch
import torch.nn.functional as F
from ..._optional import is_pyterrier_available
from ..base import Ranker


class CatConfig(PretrainedConfig):
    model_type = "Cat"


class Cat(Ranker):
    """Wrapper for Cat Model

    Parameters
    ----------
    model : PreTrainedModel
        the underlying HF model
    tokenizer : PreTrainedTokenizer
        the tokenizer for the model
    config : AutoConfig
        the configuration for the model
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
        return super().from_pretrained(model_name_or_path, config, num_labels=num_labels, **kwargs)

    def prepare_outputs(self, logits, labels=None):
        """Prepare outputs"""
        breakpoint()
        return F.log_softmax(logits.reshape(-1, self.config.group_size, 2), dim=-1)[
            :, :, 1
        ], (labels.view(-1, self.config.group_size) if labels is not None else None)


AutoConfig.register("Cat", CatConfig)
AutoModelForSequenceClassification.register(CatConfig, Cat)
