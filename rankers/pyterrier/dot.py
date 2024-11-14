from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, AutoModel, AutoTokenizer
from ...modelling.dot import Dot, Pooler, DotConfig
from typing import Union
import torch
import pandas as pd
from more_itertools import chunked
import numpy as np
import pyterrier as pt

class DotTransformer(pt.Transformer):
    cls_architecture = AutoModel
    cls_config = DotConfig
    def __init__(self, 
                 model : PreTrainedModel, 
                 tokenizer : PreTrainedTokenizer, 
                 config : DotConfig, 
                 batch_size : int, 
                 text_field : str = 'text', 
                 device : Union[str, torch.device] = None,
                 verbose : bool = False
                 ) -> None:
        super().__init__()
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.eval().to(self.device)
        self.tokenizer = tokenizer
        self.config = config
        self.batch_size = batch_size
        self.text_field = text_field
        self.pooling = {
            'mean': lambda x: x.mean(dim=1),
            'cls' : lambda x: x[:, 0],
            'late_interaction': lambda x: x,
            'none': lambda x: x,
        }[config.pooling_type]
        self.verbose = verbose

    @classmethod
    def from_pretrained(cls, 
                        model_name_or_path : str,
                        batch_size : int = 64,
                        pooling : str = 'cls',
                        config : PretrainedConfig = None,
                        text_field : str = 'text',
                        device : Union[str, torch.device] = None,
                        verbose : bool = False,
                        **kwargs
                        ):
        config = cls.cls_config.from_pretrained(model_name_or_path) if config is None else config
        config.pooling_type = pooling
        pooler = None if not config.use_pooler else Pooler.from_pretrained(model_name_or_path+"/pooler")
        model_d = None if config.model_tied else cls.cls_architecture.from_pretrained(model_name_or_path + "/model_d", **kwargs)
        model_q = cls.cls_architecture.from_pretrained(model_name_or_path, **kwargs)
        model = Dot(model_q, config, model_d, pooler)
        return cls(model, AutoTokenizer.from_pretrained(model_name_or_path), config, batch_size, text_field, device, verbose)
    
    @classmethod 
    def from_model(cls, 
                   model : PreTrainedModel, 
                   tokenizer : PreTrainedTokenizer,
                   batch_size : int = 64, 
                   text_field : str = 'text', 
                   verbose : bool = False
                   ):
        config = model.config
        return cls(model.eval(), tokenizer, config, batch_size, text_field, model.device, verbose)
    
    def encode_queries(self, texts, batch_size=None) -> np.ndarray:
        results = []
        with torch.no_grad():
            for chunk in chunked(texts, batch_size or self.batch_size):
                inps = self.tokenizer(list(chunk), return_tensors='pt', padding=True, truncation=True)
                inps = {k: v.to(self.device) for k, v in inps.items()}
                res = self.model._encode_q(**inps)
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
                res = self.model._encode_d(**inps)
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
            (['query'], self.query_model),
            ([self.text_field], self.doc_model),
        ]
        for fields, fn in modes:
            if all(f in columns for f in fields):
                return fn()(inp)
        message = f'Unexpected input with columns: {inp.columns}. Supports:'
        for fields, fn in modes:
            message += f'\n - {fn.__doc__.strip()}: {fields}'
        raise RuntimeError(message)
    
    def query_model(self, verbose=None, batch_size=None) -> pt.Transformer:
        """
        Query encoding
        """
        return BiQuerymodel(self, verbose=verbose, batch_size=batch_size)

    def doc_model(self, verbose=None, batch_size=None) -> pt.Transformer:
        """
        Doc encoding
        """
        return BiDocmodel(self, verbose=verbose, batch_size=batch_size)

    def scorer(self, verbose=None, batch_size=None) -> pt.Transformer:
        """
        Scoring (re-ranking)
        """
        return BiScorer(self, verbose=verbose, batch_size=batch_size)

class BiQuerymodel(pt.Transformer):
    def __init__(self, bi_model: DotTransformer, verbose=None, batch_size=None):
        self.bi_model = bi_model
        self.verbose = verbose if verbose is not None else bi_model.verbose
        self.batch_size = batch_size if batch_size is not None else bi_model.batch_size

    def encode(self, texts, batch_size=None) -> np.array:
        return self.bi_model.encode_queries(texts, batch_size=batch_size or self.batch_size)

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        assert all(c in inp.columns for c in ['query'])
        it = inp['query'].values
        it, inv = np.unique(it, return_inverse=True)
        if self.verbose:
            it = pt.tqdm(it, desc='Encoding Queries', unit='query')
        enc = self.encode(it)
        return inp.assign(query_vec=[enc[i] for i in inv])

    def __repr__(self):
        return f'{repr(self.bi_model)}.query_model()'
    
class BiDocmodel(pt.Transformer):
    def __init__(self, bi_model: DotTransformer, verbose=None, batch_size=None, text_field=None):
        self.bi_model = bi_model
        self.verbose = verbose if verbose is not None else bi_model.verbose
        self.batch_size = batch_size if batch_size is not None else bi_model.batch_size
        self.text_field = text_field if text_field is not None else bi_model.text_field

    def encode(self, texts, batch_size=None) -> np.array:
        return self.bi_model.encode_docs(texts, batch_size=batch_size or self.batch_size)

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        assert all(c in inp.columns for c in [self.text_field])
        it = inp[self.text_field]
        if self.verbose:
            it = pt.tqdm(it, desc='Encoding Docs', unit='doc')
        return inp.assign(doc_vec=list(self.encode(it)))

    def __repr__(self):
        return f'{repr(self.bi_model)}.doc_model()'

class BiScorer(pt.Transformer):
    def __init__(self, bi_model: DotTransformer, verbose=None, batch_size=None, text_field=None):
        self.bi_model = bi_model
        self.verbose = verbose if verbose is not None else bi_model.verbose
        self.batch_size = batch_size if batch_size is not None else bi_model.batch_size
        self.text_field = text_field if text_field is not None else bi_model.text_field

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        assert 'query_vec' in inp.columns or 'query' in inp.columns
        assert 'doc_vec' in inp.columns or self.text_field in inp.columns
        if 'query_vec' in inp.columns:
            query_vec = inp['query_vec']
        else:
            query_vec = self.bi_model.query_model(batch_size=self.batch_size, verbose=self.verbose)(inp)['query_vec']
        if 'doc_vec' in inp.columns:
            doc_vec = inp['doc_vec']
        else:
            doc_vec = self.bi_model.doc_model(batch_size=self.batch_size, verbose=self.verbose)(inp)['doc_vec']
            if self.bi_model.config.pooling_type == 'late_interaction':
                scores = doc_vec @ query_vec.permute(0, 2, 1)
                scores = scores.max(1).values
                scores = scores.sum(-1)
            else:
                scores = (query_vec * doc_vec).apply(np.sum)
        outp = inp.assign(score=scores)
        return pt.model.add_ranks(outp)

    def __repr__(self):
        return f'{repr(self.bi_model)}.scorer()'
    
__all__ = ['DotTransformer']