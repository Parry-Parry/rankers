from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForSequenceClassification, AutoTokenizer
from transformers import OptionalDependencyNotAvailable
import pandas as pd
from more_itertools import chunked
import numpy as np
import jax 
import jax.numpy as jnp
import jax.nn as nn
from ..._optional import is_pyterrier_available
from .cat import CatConfig

PT_AVAILIBLE = is_pyterrier_available()

if PT_AVAILIBLE:
    import pyterrier as pt
    if not pt.started():
        pt.init()

class FlaxCat(PreTrainedModel):
    """Wrapper for Cat Model
    
    Parameters
    ----------
    classifier : PreTrainedModel
        the classifier model
    config : AutoConfig
        the configuration for the model
    """
    model_architecture = 'Cat'
    def __init__(
        self,
        classifier: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: CatConfig,
    ):
        super().__init__(config)
        self.classifier = classifier
        self.tokenizer = tokenizer
    
    def prepare_outputs(self, logits):
        """Prepare outputs"""
        return nn.log_softmax(jnp.reshape(logits, (-1, self.config.group_size, 2)), axis=-1)[:, :, 1]
            
    def forward(self, loss, sequences, labels=None):
        """Compute the loss given (pairs, labels)"""
        logits = self.classifier(**sequences).logits
        pred = self.prepare_outputs(logits)
        loss_value = loss(pred) if labels is None else loss(pred, labels)
        return (loss_value, pred)

    def save_pretrained(self, model_dir, **kwargs):
        """Save classifier"""
        self.config.save_pretrained(model_dir)
        self.classifier.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
    

    def load_state_dict(self, model_dir):
        """Load state dict from a directory"""
        return self.classifier.load_state_dict(AutoModelForSequenceClassification.from_pretrained(model_dir).state_dict())
    

    def to_pyterrier(self) -> "pt.Transformer":
        if not PT_AVAILIBLE: raise OptionalDependencyNotAvailable()
        return FlaxCatTransformer.from_model(self.classifier, self.tokenizer, text_field='text')

    @classmethod
    def from_pretrained(cls, model_dir_or_name : str, num_labels=2):
        """Load classifier from a directory"""
        config = CatConfig.from_pretrained(model_dir_or_name)
        classifier = AutoModelForSequenceClassification.from_pretrained(model_dir_or_name, num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained(model_dir_or_name)
        return cls(classifier, tokenizer, config)

class FlaxCatTransformer(pt.Transformer):
    def __init__(self, 
                 model : PreTrainedModel, 
                 tokenizer : PreTrainedTokenizer, 
                 config : CatConfig, 
                 batch_size : int, 
                 text_field : str = 'text', 
                 verbose : bool = False
                 ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.batch_size = batch_size
        self.text_field = text_field
        self.verbose = verbose
    
    @classmethod
    def from_pretrained(cls, 
                        model_name_or_path : str, 
                        batch_size : int = 64, 
                        text_field : str = 'text', 
                        ):
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        config = CatConfig.from_pretrained(model_name_or_path)
        return cls(model, tokenizer, config, batch_size, text_field)

    @classmethod 
    def from_model(cls, 
                   model : PreTrainedModel, 
                   tokenizer : PreTrainedTokenizer,
                   batch_size : int = 64, 
                   text_field : str = 'text', 
                   ): 
        config = model.config
        return cls(model, tokenizer, config, batch_size, text_field, model.device)
    
    def transform(self, inp : pd.DataFrame) -> pd.DataFrame:
        scores = []
        it = inp[['query', self.text_field]].itertuples(index=False)
        if self.verbose:
            it = pt.tqdm(it, total=len(inp), unit='record', desc='Cat scoring')
        for chunk in chunked(it, self.batch_size):
            queries, texts = map(list, zip(*chunk))
            inps = self.tokenizer(queries, texts, return_tensors='np', padding=True, truncation=True)
            scores.append(nn.log_softmax(self.model(**inps).logits, axist=-1)[:, 1])
        res = inp.assign(score=np.concatenate(scores))
        pt.model.add_ranks(res)
        res = res.sort_values(['qid', 'rank'])
        return res