from copy import deepcopy
import os
from transformers import FlaxPreTrainedModel, PreTrainedTokenizer, AutoModel, AutoTokenizer, AutoConfig
from transformers.utils import OptionalDependencyNotAvailable
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from more_itertools import chunked
from ...train.loss.flax import batched_dot_product, cross_dot_product
from ..._optional import is_pyterrier_available
from .dot import DotConfig

PT_AVAILIBLE = is_pyterrier_available()

if PT_AVAILIBLE:
    import pyterrier as pt
    if not pt.started():
        pt.init()

class Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_q = nn.Dense(config.pooler_dim_out)
        self.dense_d = nn.Dense(config.pooler_dim_out) if not config.pooler_tied else self.dense_q
    
    def __call__(self, hidden_states, d=False):
        return self.dense_d(hidden_states) if d else self.dense_q(hidden_states)

class FlaxDot(FlaxPreTrainedModel):
    """
    Dot Model for Fine-Tuning 

    Parameters
    ----------
    encoder : FlaxPreTrainedModel
        the encoder model
    config : DotConfig
        the configuration for the model
    encoder_d : FlaxPreTrainedModel
        the document encoder model
    pooler : Pooler
        the pooling layer
    """
    def __init__(
        self,
        encoder : FlaxPreTrainedModel,
        tokenizer : PreTrainedTokenizer,
        config : AutoConfig,
        encoder_d : FlaxPreTrainedModel = None,
        pooler : Pooler = None
    ):
        super().__init__(config)
        self.encoder = encoder
        self.tokenizer = tokenizer
        if encoder_d: self.encoder_d = encoder_d
        else: self.encoder_d = self.encoder if config.encoder_tied else deepcopy(self.encoder)
        self.pooling = {
            'mean': self._mean,
            'cls' : self._cls,
        }[config.mode]

        if config.use_pooler: self.pooler = Pooler(config) if pooler is None else pooler
        else: self.pooler = lambda x, y=True : x

    def prepare_outputs(self, query_reps, docs_batch_reps, labels=None):
        batch_size = jnp.size(query_reps, 0)
        emb_q = jnp.reshape(query_reps, (batch_size, 1, -1))
        emb_d = jnp.reshape(docs_batch_reps, (batch_size, self.config.group_size, -1))
        pred = batched_dot_product(emb_q, emb_d)

        if self.config.inbatch_negatives:
            inbatch_d = jnp.reshape(emb_d[:, 0, :], (batch_size, 1, -1))
            inbatch_pred = cross_dot_product(emb_q, inbatch_d)
        else: inbatch_pred = None

        if labels is not None: labels = jnp.reshape(labels, (batch_size, self.config.group_size))

        return pred, labels, inbatch_pred
    
    def _cls(self, x : jax.Array) -> jax.Array:
        return self.pooler(x[:, 0])
    
    def _mean(self, x : jax.Array) -> jax.Array:
        return self.pooler(jnp.mean(x, axis=1))
    
    def _encode_d(self, **text):
        return self.pooling(self.encoder_d(**text).last_hidden_state)
    
    def _encode_q(self, **text):
        return self.pooling(self.encoder(**text).last_hidden_state)

    def forward(self, 
                loss = None, 
                queries = None, 
                docs_batch = None, 
                labels=None):
        """Compute the loss given (queries, docs, labels)"""
        queries = {k: v.to(self.encoder.device) for k, v in queries.items()} if queries is not None else None
        docs_batch = {k: v.to(self.encoder_d.device) for k, v in docs_batch.items()} if docs_batch is not None else None
        labels = labels.to(self.encoder_d.device) if labels is not None else None
        query_reps = self._encode_q(**queries) if queries is not None else None
        docs_batch_reps = self._encode_d(**docs_batch) if docs_batch is not None else None
        pred, labels, inbatch_pred = self.prepare_outputs(query_reps, docs_batch_reps, labels)
        loss_value = loss(pred, labels) if labels is not None else loss(pred)

        if inbatch_pred is not None:
            # log likelihood of the inbatch negatives should all be 0
            inbatch_labels = jnp.eye(inbatch_pred.size(0))
            inbatch_loss = loss(inbatch_pred, inbatch_labels)
        else:
            inbatch_loss = 0

        return (loss_value + inbatch_loss, pred) 

    def save_pretrained(self, model_dir, **kwargs):
        """Save both query and document encoder"""
        self.config.save_pretrained(model_dir)
        self.encoder.save_pretrained(model_dir)
        if not self.config.encoder_tied: self.encoder_d.save_pretrained(model_dir + "/encoder_d")
        if self.config.use_pooler: self.pooler.save_pretrained(model_dir + "/pooler")
        self.tokenizer.save_pretrained(model_dir)


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
        tokenizer = AutoTokenizer.from_pretrained(model_dir_or_name)
        encoder = AutoModel.from_pretrained(model_dir_or_name)
        return cls(encoder, tokenizer, config)
    
    def to_pyterrier(self) -> "FlaxDotTransformer":
        if not PT_AVAILIBLE: raise OptionalDependencyNotAvailable()
        return FlaxDotTransformer.from_model(self, self.tokenizer, text_field='text')

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
        return BiQueryEncoder(self, verbose=verbose, batch_size=batch_size)

    def doc_encoder(self, verbose=None, batch_size=None) -> pt.Transformer:
        """
        Doc encoding
        """
        return BiDocEncoder(self, verbose=verbose, batch_size=batch_size)

    def scorer(self, verbose=None, batch_size=None) -> pt.Transformer:
        """
        Scoring (re-ranking)
        """
        return BiScorer(self, verbose=verbose, batch_size=batch_size)

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
    
class BiDocEncoder(pt.Transformer):
    def __init__(self, bi_encoder_model: DotTransformer, verbose=None, batch_size=None, text_field=None):
        self.bi_encoder_model = bi_encoder_model
        self.verbose = verbose if verbose is not None else bi_encoder_model.verbose
        self.batch_size = batch_size if batch_size is not None else bi_encoder_model.batch_size
        self.text_field = text_field if text_field is not None else bi_encoder_model.text_field

    def encode(self, texts, batch_size=None) -> np.array:
        return self.bi_encoder_model.encode_docs(texts, batch_size=batch_size or self.batch_size)

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        assert all(c in inp.columns for c in [self.text_field])
        it = inp[self.text_field]
        if self.verbose:
            it = pt.tqdm(it, desc='Encoding Docs', unit='doc')
        return inp.assign(doc_vec=list(self.encode(it)))

    def __repr__(self):
        return f'{repr(self.bi_encoder_model)}.doc_encoder()'

class BiScorer(pt.Transformer):
    def __init__(self, bi_encoder_model: DotTransformer, verbose=None, batch_size=None, text_field=None):
        self.bi_encoder_model = bi_encoder_model
        self.verbose = verbose if verbose is not None else bi_encoder_model.verbose
        self.batch_size = batch_size if batch_size is not None else bi_encoder_model.batch_size
        self.text_field = text_field if text_field is not None else bi_encoder_model.text_field

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