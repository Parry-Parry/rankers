import pyterrier as pt
from transformers import PreTrainedModel, PretrainedConfig, PreTrainedTokenizer, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from typing import Union
import torch
import pandas as pd
from more_itertools import chunked
import numpy as np
import torch.nn.functional as F

class CatTransformer(pt.Transformer):
    cls_architecture = AutoModelForSequenceClassification
    cls_config = AutoConfig
    def __init__(self, 
                 model : PreTrainedModel, 
                 tokenizer : PreTrainedTokenizer, 
                 config : PretrainedConfig, 
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
                        config : PretrainedConfig = None,
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
                 config : PretrainedConfig, 
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
    
    @classmethod
    def from_pretrained(cls, 
                        model_name_or_path : str, 
                        batch_size : int = 64, 
                        text_field : str = 'text', 
                        config : PretrainedConfig = None,
                        device : Union[str, torch.device] = None,
                        verbose : bool = False,
                        **kwargs
                        ):
        config = cls.cls_config.from_pretrained(model_name_or_path) if config is None else config
        model = cls.cls_architecture.from_pretrained(model_name_or_path, config=config, **kwargs).cuda().eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return cls(model, tokenizer, config, batch_size, text_field, device, verbose)
    
    def transform(self, inp : pd.DataFrame) -> pd.DataFrame:
        scores = []
        it = inp[['query', self.text_field]].itertuples(index=False)
        if self.verbose:
            it = pt.tqdm(it, total=len(inp), unit='record', desc='Duo scoring')
        with torch.no_grad():
            for chunk in chunked(it, self.batch_size):
                queries, texts = map(list, zip(*chunk))
                inps = self.tokenizer(queries, texts, return_tensors='pt', padding=True, truncation=True)
                inps = {k: v.to(self.device) for k, v in inps.items()}
                scores.append(self.model(**inps).logits.cpu().detach().numpy())
        res = inp.assign(score=np.concatenate(scores))
        res = inp.assign(score=np.concatenate(scores))
        res = res.sort_values(['qid', 'score'], ascending=[True, False])
        return pt.model.add_ranks(res)
    
__all__ = ['CatTransformer', 'PairTransformer']