from copy import deepcopy
import os
from transformers import PreTrainedModel, AutoModel, PretrainedConfig
import torch
from torch import nn
from ..inference.models import DotTransformer, PoolingType

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
                 model_name_or_path : str ,
                 mode='cls', 
                 encoder_tied=True,
                 use_pooler=False,
                 pooler_dim_in=768,
                 pooler_dim_out=768,
                 pooler_tied=True,
                 **kwargs):
        self.mode = mode
        self.encoder_tied = encoder_tied
        self.use_pooler = use_pooler
        self.pooler_dim_in = pooler_dim_in
        self.pooler_dim_out = pooler_dim_out
        self.pooler_tied = pooler_tied
        super().__init__(model_name_or_path, **kwargs)

class Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_q = nn.Linear(config.pooler_dim_in, config.pooler_dim_out)
        self.dense_d = nn.Linear(config.pooler_dim_in, config.pooler_dim_out) if not config.pooler_tied else self.dense_q
    
    @classmethod
    def from_pretrained(cls, model_name_or_path : str) -> 'Pooler':
        config = DotConfig.from_pretrained(model_name_or_path)
        model = cls(config)
        model.load_state_dict(torch.load(model_name_or_path + "/pooler"))
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
            PoolingType.MEAN: self._mean,
            PoolingType.CLS: self._cls,
        }[config.mode]

        if config.use_pooler: self.pooler = Pooler(config) if pooler is None else pooler
        else: self.pooler = lambda x, y=True : x
    
    def _cls(self, x : torch.Tensor, **kwargs) -> torch.Tensor:
        return x[:, 0]
    
    def _mean(self, x : torch.Tensor, **kwargs) -> torch.Tensor:
        return x.mean(dim=1)
    
    def _encode_d(self, **text):
        return self.pooler(self.encoder_d(**text), True)
    
    def _encode_q(self, **text):
        return self.pooler(self.encoder(**text))

    def forward(self, loss, queries, docs_batch, labels=None):
        """Compute the loss given (queries, docs, labels)"""
        queries = {k: v.to(self.encoder.device) for k, v in queries.items()}
        docs_batch = {k: v.to(self.encoder_d.device) for k, v in docs_batch.items()}
        labels = labels.to(self.encoder_d.device) if labels is not None else None
        q_reps = self.encode_q(**queries)
        docs_batch_rep = self.encode_d(**docs_batch)
    
        return loss(q_reps, docs_batch_rep) if labels is None else loss(q_reps, docs_batch_rep, labels)

    def save_pretrained(self, model_dir):
        """Save both query and document encoder"""
        self.config.save_pretrained(model_dir + "/config.json")
        self.encoder.save_pretrained(model_dir + "/encoder")
        if not self.config.encoder_tied: self.encoder_d.save_pretrained(model_dir + "/encoder_d")
        if self.config.use_pooler: self.pooler.save_pretrained(model_dir + "/pooler")
    
    def load_state_dict(self, model_dir):
        """Load state dict from a directory"""
        self.config = DotConfig.from_pretrained(model_dir)
        self.encoder.load_state_dict(AutoModel.from_pretrained(model_dir + "/encoder").state_dict())
        if not self.config.encoder_tied: self.encoder_d.load_state_dict(AutoModel.from_pretrained(model_dir + "/encoder_d").state_dict())
        if self.config.use_pooler: self.pooler.load_state_dict(AutoModel.from_pretrained(model_dir + "/pooler").state_dict())

    @classmethod
    def from_pretrained(cls, model_dir_or_name, **kwargs):
        """Load encoder"""
        if os.path.isdir(model_dir_or_name):
            config = DotConfig.from_pretrained(model_dir_or_name, **kwargs)
            encoder = AutoModel.from_pretrained(model_dir_or_name + "/encoder")
            encoder_d = None if config.encoder_tied else AutoModel.from_pretrained(model_dir_or_name + "/encoder_d") 
            pooler = None if not config.use_pooler else Pooler.from_pretrained(model_dir_or_name + "/pooler")

            return cls(encoder, config, encoder_d, pooler)
        config = DotConfig.from_pretrained(model_dir_or_name, **kwargs)
        encoder = AutoModel.from_pretrained(model_dir_or_name)
        return cls(encoder, config)
    
    def eval(self) -> DotTransformer:
        return DotTransformer.from_model(self, text_field='text')