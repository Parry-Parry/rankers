from copy import deepcopy
import os
from transformers import FlaxPreTrainedModel, PreTrainedTokenizer, AutoTokenizer, AutoConfig, FlaxAutoModel
import jax
import jax.numpy as jnp
import flax.linen as nn
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
    model_type = 'Dot'
    architecture_class = FlaxAutoModel
    config_class = DotConfig
    transformer_class = None
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
        if PT_AVAILIBLE:
            from ...pyterrier.dot.flaxdot import FlaxDotTransformer
            self.transformer_class = FlaxDotTransformer

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
        self.encoder.load_state_dict(self.architecture_class.from_pretrained(model_dir).state_dict())
        if not self.config.encoder_tied: self.encoder_d.load_state_dict(self.architecture_class.from_pretrained(model_dir + "/encoder_d").state_dict())
        if self.config.use_pooler: self.pooler.load_state_dict(self.architecture_class.from_pretrained(model_dir + "/pooler").state_dict())

    @classmethod
    def from_pretrained(cls, model_dir_or_name, **kwargs):
        """Load encoder"""
        if os.path.isdir(model_dir_or_name):
            config = DotConfig.from_pretrained(model_dir_or_name, **kwargs)
            encoder = cls.architecture_class.from_pretrained(model_dir_or_name)
            encoder_d = None if config.encoder_tied else cls.architecture_class.from_pretrained(model_dir_or_name + "/encoder_d") 
            pooler = None if not config.use_pooler else Pooler.from_pretrained(model_dir_or_name + "/pooler")

            return cls(encoder, config, encoder_d, pooler)
        config = DotConfig(model_dir_or_name, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_dir_or_name)
        encoder = cls.architecture_class.from_pretrained(model_dir_or_name)
        return cls(encoder, tokenizer, config)
    
    def to_pyterrier(self) -> "FlaxDotTransformer":
        assert self.transformer_class is not None, "PyTerrier is not available"
        return self.architecture_class.from_model(self, self.tokenizer, text_field='text')