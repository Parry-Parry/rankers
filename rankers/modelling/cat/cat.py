<<<<<<<< HEAD:rankers/modelling/cat.py
import pyterrier as pt
if not pt.started():
    pt.init()
from transformers import PreTrainedModel, PreTrainedConfig, PreTrainedTokenizer, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
========
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from transformers.utils import OptionalDependencyNotAvailable
>>>>>>>> 4793ad1e1b9c84eedd56401a37b5411435079091:rankers/modelling/cat/cat.py
from typing import Union
import torch
import pandas as pd
from more_itertools import chunked
import numpy as np
import torch.nn.functional as F
from ..._optional import is_pyterrier_available

<<<<<<<< HEAD:rankers/modelling/cat.py
========
PT_AVAILIBLE = is_pyterrier_available()

if PT_AVAILIBLE:
    import pyterrier as pt
    if not pt.started():
        pt.init()

class Cat(PreTrainedModel):
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
        config: AutoConfig,
    ):
        super().__init__(config)
        self.classifier = classifier
        self.tokenizer = tokenizer
    
    def prepare_outputs(self, logits, labels=None):
        """Prepare outputs"""
        return F.log_softmax(logits.reshape(-1, self.config.group_size, 2), dim=-1)[:, :, 1], labels.view(-1, self.config.group_size) if labels is not None else None

    def forward(self, loss, sequences, labels=None):
        """Compute the loss given (pairs, labels)"""
        sequences = {k: v.to(self.classifier.device) for k, v in sequences.items()}
        labels = labels.to(self.classifier.device) if labels is not None else None
        logits = self.classifier(**sequences).logits
        pred, labels = self.prepare_outputs(logits, labels)
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
        return CatTransformer.from_model(self.classifier, self.tokenizer, text_field='text')

    @classmethod
    def from_pretrained(cls, model_dir_or_name : str, num_labels=2):
        """Load classifier from a directory"""
        config = AutoConfig.from_pretrained(model_dir_or_name)
        classifier = AutoModelForSequenceClassification.from_pretrained(model_dir_or_name, num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained(model_dir_or_name)
        return cls(classifier, tokenizer, config)

>>>>>>>> 4793ad1e1b9c84eedd56401a37b5411435079091:rankers/modelling/cat/cat.py
class CatTransformer(pt.Transformer):
    cls_architecture = AutoModelForSequenceClassification
    cls_config = AutoConfig
    def __init__(self, 
                 model : PreTrainedModel, 
                 tokenizer : PreTrainedTokenizer, 
                 config : PreTrainedConfig, 
                 batch_size : int, 
                 text_field : str = 'text', 
                 device : Union[str, torch.device] = None,
                 verbose : bool = False
                 ) -> None:
        super().__init__()
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.config = config
        self.batch_size = batch_size
        self.text_field = text_field
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        self.verbose = verbose
    
    @classmethod
    def from_pretrained(cls, 
                        model_name_or_path : str, 
                        batch_size : int = 64, 
                        text_field : str = 'text', 
                        config : PreTrainedConfig = None,
                        device : Union[str, torch.device] = None,
                        verbose : bool = False,
                        **kwargs
                        ):
        config = cls.cls_config.from_pretrained(model_name_or_path) if config is None else config
        model = cls.cls_architecture.from_pretrained(model_name_or_path, config=config, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return cls(model, tokenizer, config, batch_size, text_field, device, verbose)

    @classmethod 
    def from_model(cls, 
                   model : PreTrainedModel, 
                   tokenizer : PreTrainedTokenizer,
                   batch_size : int = 64, 
                   text_field : str = 'text', 
                   verbose : bool = False
                   ): 
        config = model.config
        return cls(model, tokenizer, config, batch_size, text_field, model.device, verbose)
    
    def transform(self, inp : pd.DataFrame) -> pd.DataFrame:
        scores = []
        it = inp[['query', self.text_field]].itertuples(index=False)
        if self.verbose:
            it = pt.tqdm(it, total=len(inp), unit='record', desc='Cat scoring')
        with torch.no_grad():
            for chunk in chunked(it, self.batch_size):
                queries, texts = map(list, zip(*chunk))
                inps = self.tokenizer(queries, texts, return_tensors='pt', padding=True, truncation=True)
                inps = {k: v.to(self.model.device) for k, v in inps.items()}
                scores.append(F.log_softmax(self.model(**inps).logits, dim=-1)[:, 1].cpu().detach().numpy())
        res = inp.assign(score=np.concatenate(scores))
        res = res.sort_values(['qid', 'score'], ascending=[True, False])
        return pt.model.add_ranks(res)

class PairTransformer(pt.Transformer):
    cls_architecture = AutoModelForSequenceClassification
    cls_config = AutoConfig
    def __init__(self, 
                 model : PreTrainedModel, 
                 tokenizer : PreTrainedTokenizer, 
                 config : PreTrainedConfig, 
                 batch_size : int, 
                 text_field : str = 'text', 
                 device : Union[str, torch.device] = None,
                 verbose : bool = False
                 ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.batch_size = batch_size
        self.text_field = text_field
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.verbose = verbose
    
<<<<<<<< HEAD:rankers/modelling/cat.py
========
    @classmethod
    def from_pretrained(cls, 
                        model_name_or_path : str, 
                        batch_size : int = 64, 
                        text_field : str = 'text', 
                        device : Union[str, torch.device] = None,
                        verbose : bool = False
                        ):
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path).cuda().eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        config = AutoConfig.from_pretrained(model_name_or_path)
        return cls(model, tokenizer, config, batch_size, text_field, device, verbose)

>>>>>>>> 4793ad1e1b9c84eedd56401a37b5411435079091:rankers/modelling/cat/cat.py
    @classmethod 
    def from_model(cls, 
                   model : PreTrainedModel, 
                   tokenizer : PreTrainedTokenizer,
                   batch_size : int = 64, 
                   text_field : str = 'text', 
                   verbose : bool = False
                   ): 
        config = model.config
        return cls(model, tokenizer, config, batch_size, text_field, model.device, verbose)
    
<<<<<<<< HEAD:rankers/modelling/cat.py
    @classmethod
    def from_pretrained(cls, 
                        model_name_or_path : str, 
                        batch_size : int = 64, 
                        text_field : str = 'text', 
                        config : PreTrainedConfig = None,
                        device : Union[str, torch.device] = None,
                        verbose : bool = False,
                        **kwargs
                        ):
        config = cls.cls_config.from_pretrained(model_name_or_path) if config is None else config
        model = cls.cls_architecture.from_pretrained(model_name_or_path, config=config, **kwargs).cuda().eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return cls(model, tokenizer, config, batch_size, text_field, device, verbose)
    
========
>>>>>>>> 4793ad1e1b9c84eedd56401a37b5411435079091:rankers/modelling/cat/cat.py
    def transform(self, inp : pd.DataFrame) -> pd.DataFrame:
        scores = []
        it = inp[['query', self.text_field]].itertuples(index=False)
        if self.verbose:
<<<<<<<< HEAD:rankers/modelling/cat.py
            it = pt.tqdm(it, total=len(inp), unit='record', desc='Duo scoring')
========
            it = pt.tqdm(it, total=len(inp), unit='record', desc='Cat scoring')
>>>>>>>> 4793ad1e1b9c84eedd56401a37b5411435079091:rankers/modelling/cat/cat.py
        with torch.no_grad():
            for chunk in chunked(it, self.batch_size):
                queries, texts = map(list, zip(*chunk))
                inps = self.tokenizer(queries, texts, return_tensors='pt', padding=True, truncation=True)
<<<<<<<< HEAD:rankers/modelling/cat.py
                inps = {k: v.to(self.device) for k, v in inps.items()}
                scores.append(self.model(**inps).logits.cpu().detach().numpy())
        res = inp.assign(score=np.concatenate(scores))
        res = res.sort_values(['qid', 'score'], ascending=[True, False])
        return pt.model.add_ranks(res)

class Cat(PreTrainedModel):
    """Wrapper for Cat Model
    
    Parameters
    ----------
    model : PreTrainedModel
        the underlying HF model
    config : AutoConfig
        the configuration for the model
    """
    model_architecture = 'Cat'
    cls_architecture = AutoModelForSequenceClassification
    cls_config = AutoConfig
    transformer_architecture = CatTransformer
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: AutoConfig,
    ):
        super().__init__(config)
        self.model = model
        self.tokenizer = tokenizer
    
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
        return self.model.load_state_dict(self.cls_architecture.from_pretrained(model_dir).state_dict())

    def to_pyterrier(self) -> "pt.Transformer":
        return self.transformer_architecture.from_model(self.model, self.tokenizer, text_field='text')

    @classmethod
    def from_pretrained(cls, model_dir_or_name : str, num_labels=2, config=None, **kwargs) -> "Cat":
        """Load model from a directory"""
        config = cls.cls_config.from_pretrained(model_dir_or_name) if config is None else config
        model = cls.cls_architecture.from_pretrained(model_dir_or_name, num_labels=num_labels, config=config **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_dir_or_name)
        return cls(model, tokenizer, config)
========
                inps = {k: v.to(self.model.device) for k, v in inps.items()}
                scores.append(F.log_softmax(self.model(**inps).logits, dim=-1)[:, 1].cpu().detach().numpy())
        res = inp.assign(score=np.concatenate(scores))
        res = res.sort_values(['qid', 'score'], ascending=[True, False])
        return pt.model.add_ranks(res)
>>>>>>>> 4793ad1e1b9c84eedd56401a37b5411435079091:rankers/modelling/cat/cat.py
