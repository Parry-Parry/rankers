from copy import deepcopy
import os
import torch
from torch import nn
import pyterrier as pt
if not pt.started():
    pt.init()
from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig, AutoModel, AutoTokenizer
from typing import Union
import pandas as pd
import numpy as np
from more_itertools import chunked
from enum import Enum

class DotConfig(PretrainedConfig):
    """Configuration for Dot Model
    
    Parameters
    ----------
    model_name_or_path : str
        the model name or path
    mode : str
        the pooling mode for the model
    encoder_tied : bool
        whether the encoder is tied
    use_pooler : bool
        whether to use the pooler
    pooler_dim_in : int
        the input dimension for the pooler
    pooler_dim_out : int
        the output dimension for the pooler
    pooler_tied : bool
        whether the pooler is tied
    """
    model_type = "dot"
    def __init__(self, 
                 model_name_or_path : str='bert-base-uncased',
                 mode='cls', 
                 encoder_tied=True,
                 use_pooler=False,
                 pooler_dim_in=768,
                 pooler_dim_out=768,
                 pooler_tied=True,
                 **kwargs):
        self.model_name_or_path = model_name_or_path
        self.mode = mode
        self.encoder_tied = encoder_tied
        self.use_pooler = use_pooler
        self.pooler_dim_in = pooler_dim_in
        self.pooler_dim_out = pooler_dim_out
        self.pooler_tied = pooler_tied
        super().__init__(**kwargs)

class Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_q = nn.Linear(config.pooler_dim_in, config.pooler_dim_out)
        self.dense_d = nn.Linear(config.pooler_dim_in, config.pooler_dim_out) if not config.pooler_tied else self.dense_q
    
    @classmethod
    def from_pretrained(cls, model_name_or_path : str='bert-base-uncased') -> 'Pooler':
        config = DotConfig.from_pretrained(model_name_or_path)
        model = cls(config)
        return model
    
    def forward(self, hidden_states, d=False):
        return self.dense_d(hidden_states) if d else self.dense_q(hidden_states)

class Dot(PreTrainedModel):
    """
    Dot Model for Fine-Tuning 

    Parameters
    ----------
    encoder : PreTrainedModel
        the encoder model
    config : DotConfig
        the configuration for the model
    encoder_d : PreTrainedModel
        the document encoder model
    pooler : Pooler
        the pooling layer
    """
    def __init__(
        self,
        encoder : PreTrainedModel,
        config : DotConfig,
        encoder_d : PreTrainedModel = None,
        pooler : Pooler = None
    ):
        super().__init__(config)
        self.encoder = encoder
        if encoder_d is None: self.encoder_d = self.encoder if config.encoder_tied else deepcopy(self.encoder)
        self.pooling = {
            'mean': self._mean,
            'cls' : self._cls,
        }[config.mode]

        if config.use_pooler: self.pooler = Pooler(config) if pooler is None else pooler
        else: self.pooler = lambda x, y=True : x
    
    def _cls(self, x : torch.Tensor) -> torch.Tensor:
        return x[:, 0]
    
    def _mean(self, x : torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)
    
    def _encode_d(self, **text):
        return self.pooler(self.encoder_d(**text).last_hidden_state)
    
    def _encode_q(self, **text):
        return self.pooler(self.encoder(**text).last_hidden_state)

    def forward(self, 
                loss = None, 
                queries = None, 
                docs_batch = None, 
                labels=None):
        """Compute the loss given (queries, docs, labels)"""
        queries = {k: v.to(self.encoder.device) for k, v in queries.items()} if queries is not None else None
        docs_batch = {k: v.to(self.encoder_d.device) for k, v in docs_batch.items()} if docs_batch is not None else None
        labels = labels.to(self.encoder_d.device) if labels is not None else None
        q_reps = self._encode_q(**queries) if queries is not None else None
        docs_batch_rep = self._encode_d(**docs_batch) if docs_batch is not None else None
    
        return loss(q_reps, docs_batch_rep) if labels is None else loss(q_reps, docs_batch_rep, labels)

    def save_pretrained(self, model_dir):
        """Save both query and document encoder"""
        self.config.save_pretrained(model_dir + "/config.json")
        self.encoder.save_pretrained(model_dir)
        if not self.config.encoder_tied: self.encoder_d.save_pretrained(model_dir + "/encoder_d")
        if self.config.use_pooler: self.pooler.save_pretrained(model_dir + "/pooler")
    
    def load_state_dict(self, model_dir):
        """Load state dict from a directory"""
        self.config = DotConfig.from_pretrained(model_dir)
        self.encoder.load_state_dict(AutoModel.from_pretrained(model_dir).state_dict())
        if not self.config.encoder_tied: self.encoder_d.load_state_dict(AutoModel.from_pretrained(model_dir + "/encoder_d").state_dict())
        if self.config.use_pooler: self.pooler.load_state_dict(AutoModel.from_pretrained(model_dir + "/pooler").state_dict())

    @classmethod
    def from_pretrained(cls, model_dir_or_name, **kwargs):
        """Load encoder"""
        if os.path.isdir(model_dir_or_name):
            config = DotConfig.from_pretrained(model_dir_or_name, **kwargs)
            encoder = AutoModel.from_pretrained(model_dir_or_name)
            encoder_d = None if config.encoder_tied else AutoModel.from_pretrained(model_dir_or_name + "/encoder_d") 
            pooler = None if not config.use_pooler else Pooler.from_pretrained(model_dir_or_name + "/pooler")

            return cls(encoder, config, encoder_d, pooler)
        config = DotConfig(model_dir_or_name, **kwargs)
        encoder = AutoModel.from_pretrained(model_dir_or_name)
        return cls(encoder, config)
    
    def eval(self) -> "DotTransformer":
        return DotTransformer.from_model(self, text_field='text')

class DotTransformer(pt.Transformer):
    def __init__(self, 
                 model : PreTrainedModel, 
                 tokenizer : PreTrainedTokenizer, 
                 config : DotConfig, 
                 batch_size : int, 
                 text_field : str = 'text', 
                 device : Union[str, torch.device] = None,
                 ) -> None:
        super().__init__()
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.eval().to(self.device)
        self.tokenizer = tokenizer
        self.config = config
        self.batch_size = batch_size
        self.text_field = text_field
        self.pooling = {
            'mean': self._mean,
            'cls' : self._cls,
        }[config.mode]

    @classmethod
    def from_pretrained(cls, 
                        model_name_or_path : str, 
                        batch_size : int = 64, 
                        pooling : str = 'cls', 
                        text_field : str = 'text', 
                        device : Union[str, torch.device] = None):
        config = DotConfig.from_pretrained(model_name_or_path)
        config.mode = pooling
        pooler = None if not config.use_pooler else Pooler.from_pretrained(model_name_or_path+"/pooler")
        encoder_d = None if config.encoder_tied else AutoModel.from_pretrained(model_name_or_path + "/encoder_d")
        encoder_q = AutoModel.from_pretrained(model_name_or_path)
        model = Dot(encoder_q, config, encoder_d, pooler)
        return cls(model, AutoTokenizer.from_pretrained(model_name_or_path), config, batch_size, text_field, device)
    
    @classmethod 
    def from_model(cls, 
                   model : PreTrainedModel, 
                   batch_size : int = 64, 
                   text_field : str = 'text', 
                   ): 
        tokenizer = AutoTokenizer.from_pretrained(model.config)
        config = model.config
        return cls(model, tokenizer, config, batch_size, text_field, model.device)
    
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