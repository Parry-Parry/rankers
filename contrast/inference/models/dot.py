import pyterrier as pt
if not pt.started():
    pt.init()
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoConfig, AutoModel, AutoTokenizer
from typing import Union
import torch
import pandas as pd
import numpy as np
from more_itertools import chunked
from ...modelling.dot import DotConfig
from enum import Enum

class PoolingType(Enum):
    MEAN = 'mean'
    CLS = 'cls'

class DotTransformer(pt.Transformer):
    def __init__(self, model : PreTrainedModel, tokenizer : PreTrainedTokenizer, config : DotConfig, batch_size : int, text_field : str = 'text', device : Union[str, torch.device] = None) -> None:
        super().__init__()
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.config = config
        self.batch_size = batch_size
        self.text_field = text_field
        self.pooling = self._mean if config.mode == PoolingType.MEAN else self._cls

    @classmethod
    def from_pretrained(cls, 
                        model_name_or_path : str, 
                        batch_size : int = 64, 
                        pooling : str = 'cls', 
                        text_field : str = 'text', 
                        device : Union[str, torch.device] = None):
        model = AutoModel.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        config = DotConfig.from_pretrained(pooling, model_name_or_path)
        return cls(model, tokenizer, config, batch_size, pooling, text_field, device)
    
    @classmethod 
    def from_model(cls, 
                   model : PreTrainedModel, 
                   batch_size : int = 64, 
                   pooling : str = 'cls',
                   text_field : str = 'text', 
                   ): 
        tokenizer = AutoTokenizer.from_pretrained(model.config)
        config = DotConfig.from_pretrained(pooling, model.config) # TODO: Make sure this is correct
        return cls(model, tokenizer, config, batch_size, text_field, model.device)
    
    def _cls(self, x : torch.Tensor) -> torch.Tensor:
        return x[:,0,:]
    
    def _mean(self, x : torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)
    
    def encode_queries(self, texts, batch_size=None) -> np.ndarray:
        results = []
        with torch.no_grad():
            for chunk in chunked(texts, batch_size or self.batch_size):
                inps = self.tokenizer(list(chunk), return_tensors='pt', padding=True, truncation=True)
                inps = {k: v.to(self.device) for k, v in inps.items()}
                res = self.pooling(self.model(**inps).last_hidden_state)
                results.append(res.cpu().numpy())
        if not results:
            return np.empty(shape=(0, 0))
        return np.concatenate(results, axis=0)

    def encode_docs(self, texts, batch_size=None) -> np.ndarray:
        results = []
        with torch.no_grad():
            for chunk in chunked(texts, batch_size or self.batch_size):
                inps = self.tokenizer(list(chunk), return_tensors='pt', padding=True, truncation=True)
                inps = {k: v.to(self.device) for k, v in inps.items()}
                res = self.pooling(self.model(**inps).last_hidden_state)
                results.append(res.cpu().numpy())
        if not results:
            return np.empty(shape=(0, 0))
        return np.concatenate(results, axis=0)
    
    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        columns = set(inp.columns)
        modes = [
            (['qid', 'query', self.text_field], self.scorer),
            (['qid', 'query_vec', self.text_field], self.scorer),
            (['qid', 'query', 'doc_vec'], self.scorer),
            (['qid', 'query_vec', 'doc_vec'], self.scorer),
            (['query'], self.query_encoder),
            ([self.text_field], self.doc_encoder),
        ]
        for fields, fn in modes:
            if all(f in columns for f in fields):
                return fn()(inp)
        message = f'Unexpected input with columns: {inp.columns}. Supports:'
        for fields, fn in modes:
            message += f'\n - {fn.__doc__.strip()}: {fields}'
        raise RuntimeError(message)

class BiQueryEncoder(pt.Transformer):
    def __init__(self, bi_encoder_model: DotTransformer, verbose=None, batch_size=None):
        self.bi_encoder_model = bi_encoder_model
        self.verbose = verbose if verbose is not None else bi_encoder_model.verbose
        self.batch_size = batch_size if batch_size is not None else bi_encoder_model.batch_size

    def encode(self, texts, batch_size=None) -> np.array:
        return self.bi_encoder_model.encode_queries(texts, batch_size=batch_size or self.batch_size)

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        assert all(c in inp.columns for c in ['query'])
        it = inp['query'].values
        it, inv = np.unique(it, return_inverse=True)
        if self.verbose:
            it = pt.tqdm(it, desc='Encoding Queries', unit='query')
        enc = self.encode(it)
        return inp.assign(query_vec=[enc[i] for i in inv])

    def __repr__(self):
        return f'{repr(self.bi_encoder_model)}.query_encoder()'
    

class BiScorer(pt.Transformer):
    def __init__(self, bi_encoder_model: DotTransformer, verbose=None, batch_size=None, text_field=None, sim_fn=None):
        self.bi_encoder_model = bi_encoder_model
        self.verbose = verbose if verbose is not None else bi_encoder_model.verbose
        self.batch_size = batch_size if batch_size is not None else bi_encoder_model.batch_size
        self.text_field = text_field if text_field is not None else bi_encoder_model.text_field
        self.sim_fn = sim_fn if sim_fn is not None else bi_encoder_model.sim_fn

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        assert 'query_vec' in inp.columns or 'query' in inp.columns
        assert 'doc_vec' in inp.columns or self.text_field in inp.columns
        if 'query_vec' in inp.columns:
            query_vec = inp['query_vec']
        else:
            query_vec = self.bi_encoder_model.query_encoder(batch_size=self.batch_size, verbose=self.verbose)(inp)['query_vec']
        if 'doc_vec' in inp.columns:
            doc_vec = inp['doc_vec']
        else:
            doc_vec = self.bi_encoder_model.doc_encoder(batch_size=self.batch_size, verbose=self.verbose)(inp)['doc_vec']
            scores = (query_vec * doc_vec).apply(np.sum)
        outp = inp.assign(score=scores)
        return pt.model.add_ranks(outp)

    def __repr__(self):
        return f'{repr(self.bi_encoder_model)}.scorer()'