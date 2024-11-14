from typing import List, Tuple
import jax
import jax.numpy as jnp
from jax import jit
from transformers import FlaxPreTrainedModel, PreTrainedTokenizer, AutoModel, AutoTokenizer
from .dot import BiDocmodel, BiQuerymodel, BiScorer
from ...modelling.dot.dot import DotConfig
from ...modelling.dot.flaxdot import Pooler, FlaxDot
from more_itertools import chunked
import pyterrier as pt
import pandas as pd
import numpy as np

class FlaxDotTransformer(pt.Transformer):
    def __init__(self, 
                 model : FlaxPreTrainedModel, 
                 tokenizer : PreTrainedTokenizer, 
                 config : DotConfig, 
                 batch_size : int, 
                 text_field : str = 'text', 
                 ) -> None:
        super().__init__()
        self.model = model.eval().to(self.device)
        self.tokenizer = tokenizer
        self.config = config
        self.batch_size = batch_size
        self.text_field = text_field
        self.pooling = {
            'mean': lambda x: jnp.mean(x, axis=1),
            'cls' : lambda x: x[:, 0],
            'none': lambda x: x,
        }[config.mode]

    @classmethod
    def from_pretrained(cls, 
                        model_name_or_path : str, 
                        batch_size : int = 64, 
                        pooling : str = 'cls', 
                        text_field : str = 'text'):
        config = DotConfig.from_pretrained(model_name_or_path)
        config.mode = pooling
        pooler = None if not config.use_pooler else Pooler.from_pretrained(model_name_or_path+"/pooler")
        encoder_d = None if config.encoder_tied else AutoModel.from_pretrained(model_name_or_path + "/encoder_d")
        encoder_q = AutoModel.from_pretrained(model_name_or_path)
        model = FlaxDot(encoder_q, config, encoder_d, pooler)
        return cls(model, AutoTokenizer.from_pretrained(model_name_or_path), config, batch_size, text_field)
    
    @classmethod 
    def from_model(cls, 
                   model : FlaxPreTrainedModel, 
                   tokenizer : PreTrainedTokenizer,
                   batch_size : int = 64, 
                   text_field : str = 'text', 
                   ): 
        config = model.config
        return cls(model, tokenizer, config, batch_size, text_field)
    
    def encode_queries(self, texts, batch_size=None) -> np.ndarray:
        results = []

        for chunk in chunked(texts, batch_size or self.batch_size):
            inps = self.tokenizer(list(chunk), return_tensors='pt', padding=True, truncation=True)
            inps = {k: v.to(self.device) for k, v in inps.items()}
            res = self.model._encode_q(**inps)
            results.append(res)
        if not results:
            return np.empty(shape=(0, 0))
        return np.concatenate(results, axis=0)

    def encode_docs(self, texts, batch_size=None) -> np.ndarray:
        results = []
        for chunk in chunked(texts, batch_size or self.batch_size):
            inps = self.tokenizer(list(chunk), return_tensors='pt', padding=True, truncation=True)
            res = self.model._encode_d(**inps)
            results.append(res)
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
    
    def query_encoder(self, verbose=None, batch_size=None) -> pt.Transformer:
        """
        Query encoding
        """
        return BiQuerymodel(self, verbose=verbose, batch_size=batch_size)

    def doc_encoder(self, verbose=None, batch_size=None) -> pt.Transformer:
        """
        Doc encoding
        """
        return BiDocmodel(self, verbose=verbose, batch_size=batch_size)

    def scorer(self, verbose=None, batch_size=None) -> pt.Transformer:
        """
        Scoring (re-ranking)
        """
        return BiScorer(self, verbose=verbose, batch_size=batch_size)