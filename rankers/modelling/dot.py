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
from ..train.loss import batched_dot_product, cross_dot_product, LOSS_REGISTRY

class DotConfig(PretrainedConfig):
    """Configuration for Dot Model
    
    Parameters
    ----------
    model_name_or_path : str
        the model name or path
    mode : str
        the pooling mode for the model
    model_tied : bool
        whether the model is tied
    use_pooler : bool
        whether to use the pooler
    pooler_dim_in : int
        the input dimension for the pooler
    pooler_dim_out : int
        the output dimension for the pooler
    pooler_tied : bool
        whether the pooler is tied
    """
    model_architecture = "Dot"
    def __init__(self, 
                 model_name_or_path : str='bert-base-uncased',
                 mode='cls', 
                 inbatch_loss=None,
                 model_tied=True,
                 use_pooler=False,
                 pooler_dim_in=768,
                 pooler_dim_out=768,
                 pooler_tied=True,
                 **kwargs):
        self.model_name_or_path = model_name_or_path
        self.mode = mode
        self.inbatch_loss = inbatch_loss
        self.model_tied = model_tied
        self.use_pooler = use_pooler
        self.pooler_dim_in = pooler_dim_in
        self.pooler_dim_out = pooler_dim_out
        self.pooler_tied = pooler_tied
        super().__init__(**kwargs)
    
    @classmethod
    def from_pretrained(cls, 
                        model_name_or_path : str='bert-base-uncased',
                        mode='cls', 
                        inbatch_loss=None,
                        model_tied=True,
                        use_pooler=False,
                        pooler_dim_in=768,
                        pooler_dim_out=768,
                        pooler_tied=True,
                          ) -> 'DotConfig':
        config = super().from_pretrained(model_name_or_path)
        config.model_name_or_path = model_name_or_path
        config.mode = mode
        config.inbatch_loss = inbatch_loss
        config.model_tied = model_tied
        config.use_pooler = use_pooler
        config.pooler_dim_in = pooler_dim_in
        config.pooler_dim_out = pooler_dim_out
        config.pooler_tied = pooler_tied
        return config

class DotTransformer(pt.Transformer):
    cls_architecture = AutoModel
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
            'none': lambda x: x,
        }[config.mode]
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
        config = DotConfig.from_pretrained(model_name_or_path) if config is None else config
        config.mode = pooling
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
    def __init__(self, bi_model_model: DotTransformer, verbose=None, batch_size=None):
        self.bi_model_model = bi_model_model
        self.verbose = verbose if verbose is not None else bi_model_model.verbose
        self.batch_size = batch_size if batch_size is not None else bi_model_model.batch_size

    def encode(self, texts, batch_size=None) -> np.array:
        return self.bi_model_model.encode_queries(texts, batch_size=batch_size or self.batch_size)

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        assert all(c in inp.columns for c in ['query'])
        it = inp['query'].values
        it, inv = np.unique(it, return_inverse=True)
        if self.verbose:
            it = pt.tqdm(it, desc='Encoding Queries', unit='query')
        enc = self.encode(it)
        return inp.assign(query_vec=[enc[i] for i in inv])

    def __repr__(self):
        return f'{repr(self.bi_model_model)}.query_model()'
    
class BiDocmodel(pt.Transformer):
    def __init__(self, bi_model_model: DotTransformer, verbose=None, batch_size=None, text_field=None):
        self.bi_model_model = bi_model_model
        self.verbose = verbose if verbose is not None else bi_model_model.verbose
        self.batch_size = batch_size if batch_size is not None else bi_model_model.batch_size
        self.text_field = text_field if text_field is not None else bi_model_model.text_field

    def encode(self, texts, batch_size=None) -> np.array:
        return self.bi_model_model.encode_docs(texts, batch_size=batch_size or self.batch_size)

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        assert all(c in inp.columns for c in [self.text_field])
        it = inp[self.text_field]
        if self.verbose:
            it = pt.tqdm(it, desc='Encoding Docs', unit='doc')
        return inp.assign(doc_vec=list(self.encode(it)))

    def __repr__(self):
        return f'{repr(self.bi_model_model)}.doc_model()'

class BiScorer(pt.Transformer):
    def __init__(self, bi_model_model: DotTransformer, verbose=None, batch_size=None, text_field=None):
        self.bi_model_model = bi_model_model
        self.verbose = verbose if verbose is not None else bi_model_model.verbose
        self.batch_size = batch_size if batch_size is not None else bi_model_model.batch_size
        self.text_field = text_field if text_field is not None else bi_model_model.text_field

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        assert 'query_vec' in inp.columns or 'query' in inp.columns
        assert 'doc_vec' in inp.columns or self.text_field in inp.columns
        if 'query_vec' in inp.columns:
            query_vec = inp['query_vec']
        else:
            query_vec = self.bi_model_model.query_model(batch_size=self.batch_size, verbose=self.verbose)(inp)['query_vec']
        if 'doc_vec' in inp.columns:
            doc_vec = inp['doc_vec']
        else:
            doc_vec = self.bi_model_model.doc_model(batch_size=self.batch_size, verbose=self.verbose)(inp)['doc_vec']
            scores = (query_vec * doc_vec).apply(np.sum)
        outp = inp.assign(score=scores)
        return pt.model.add_ranks(outp)

    def __repr__(self):
        return f'{repr(self.bi_model_model)}.scorer()'

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
    model : PreTrainedModel
        the model model
    config : DotConfig
        the configuration for the model
    model_d : PreTrainedModel
        the document model model
    pooler : Pooler
        the pooling layer
    """
    model_architecture = 'Dot'
    cls_architecture = AutoModel
    transformer_architecture = DotTransformer
    def __init__(
        self,
        model : PreTrainedModel,
        tokenizer : PreTrainedTokenizer,
        config : DotConfig,
        model_d : PreTrainedModel = None,
        pooler : Pooler = None,
    ):
        super().__init__(config)
        self.model = model
        self.tokenizer = tokenizer
        if model_d: self.model_d = model_d
        else: self.model_d = self.model if config.model_tied else deepcopy(self.model)
        self.pooling = {
            'mean': self._mean,
            'cls' : self._cls,
            'none': lambda x: x,
        }[config.mode]

        if config.use_pooler: self.pooler = Pooler(config) if pooler is None else pooler
        else: self.pooler = lambda x, y =True : x

        if config.inbatch_loss is not None:
            if config.inbatch_loss not in LOSS_REGISTRY.available:
                raise ValueError(f"Unknown loss: {config.inbatch_loss}, choices are {LOSS_REGISTRY.available}")
            self.inbatch_loss_fn = LOSS_REGISTRY.get(config.inbatch_loss)
        else:
            self.inbatch_loss_fn = None

    def prepare_outputs(self, query_reps, docs_batch_reps, labels=None):
        batch_size = query_reps.size(0)
        emb_q = query_reps.reshape(batch_size, 1, -1)
        emb_d = docs_batch_reps.reshape(batch_size, self.config.group_size, -1)
        pred = batched_dot_product(emb_q, emb_d)

        if self.config.inbatch_loss is not None:
            inbatch_d = emb_d[:, 0]
            inbatch_pred = cross_dot_product(emb_q.view(batch_size, -1), inbatch_d)
        else:
            inbatch_pred = None

        if labels is not None: labels = labels.reshape(batch_size, self.config.group_size)

        return pred, labels, inbatch_pred
    
    def _cls(self, x : torch.Tensor) -> torch.Tensor:
        return self.pooler(x[:, 0])
    
    def _mean(self, x : torch.Tensor) -> torch.Tensor:
        return self.pooler(x.mean(dim=1))
    
    def _encode_d(self, **text):
        return self.pooling(self.model_d(**text).last_hidden_state)
    
    def _encode_q(self, **text):
        return self.pooling(self.model(**text).last_hidden_state)

    def forward(self, 
                loss = None, 
                queries = None, 
                docs_batch = None, 
                labels=None):
        """Compute the loss given (queries, docs, labels)"""
        queries = {k: v.to(self.model.device) for k, v in queries.items()} if queries is not None else None
        docs_batch = {k: v.to(self.model_d.device) for k, v in docs_batch.items()} if docs_batch is not None else None
        labels = labels.to(self.model_d.device) if labels is not None else None

        query_reps = self._encode_q(**queries) if queries is not None else None
        docs_batch_reps = self._encode_d(**docs_batch) if docs_batch is not None else None

        pred, labels, inbatch_pred = self.prepare_outputs(query_reps, docs_batch_reps, labels)
        inbatch_loss = self.inbatch_loss_fn(inbatch_pred, torch.eye(inbatch_pred.shape[0]).to(inbatch_pred.device)) if inbatch_pred is not None else 0.
        
        loss_value = loss(pred, labels) if labels is not None else loss(pred)
        loss_value += inbatch_loss
        return (loss_value, pred) 

    def save_pretrained(self, model_dir, **kwargs):
        """Save both query and document model"""
        self.config.save_pretrained(model_dir)
        self.model.save_pretrained(model_dir)
        if not self.config.model_tied: self.model_d.save_pretrained(model_dir + "/model_d")
        if self.config.use_pooler: self.pooler.save_pretrained(model_dir + "/pooler")
        self.tokenizer.save_pretrained(model_dir)


    def load_state_dict(self, model_dir):
        """Load state dict from a directory"""
        self.config = DotConfig.from_pretrained(model_dir)
        self.model.load_state_dict(self.cls_architecture.from_pretrained(model_dir).state_dict())
        if not self.config.model_tied: self.model_d.load_state_dict(self.cls_architecture.from_pretrained(model_dir + "/model_d").state_dict())
        if self.config.use_pooler: self.pooler.load_state_dict(self.cls_architecture.from_pretrained(model_dir + "/pooler").state_dict())

    @classmethod
    def from_pretrained(cls, model_dir_or_name, config = None, **kwargs) -> "Dot":
        """Load model"""
        if os.path.isdir(model_dir_or_name):
            config = DotConfig.from_pretrained(model_dir_or_name) if config is None else config
            model = cls.cls_architecture.from_pretrained(model_dir_or_name, **kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_dir_or_name)
            model_d = None if config.model_tied else cls.cls_architecture.from_pretrained(model_dir_or_name + "/model_d", **kwargs) 
            pooler = None if not config.use_pooler else Pooler.from_pretrained(model_dir_or_name + "/pooler")

            return cls(model, tokenizer, config, model_d, pooler)
        config = DotConfig(model_dir_or_name, **kwargs) if config is None else config
        tokenizer = AutoTokenizer.from_pretrained(model_dir_or_name)
        model = cls.cls_architecture.from_pretrained(model_dir_or_name)
        return cls(model, tokenizer, config)
    
    def to_pyterrier(self) -> "DotTransformer":
        return self.transformer_architecture.from_model(self, self.tokenizer, text_field='text')