from copy import deepcopy
import os
import torch
from torch import nn
import pyterrier as pt
from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig, AutoModel, AutoTokenizer, AutoConfig
from ..train.loss import batched_dot_product, cross_dot_product

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
    model_type = "Dot"
    def __init__(self, 
                 model_name_or_path : str='bert-base-uncased',
                 pooling_type ='cls', 
                 inbatch_loss=None,
                 model_tied=True,
                 use_pooler=False,
                 pooler_dim_in=768,
                 pooler_dim_out=768,
                 pooler_tied=True,
                 **kwargs):
        self.model_name_or_path = model_name_or_path
        self.pooling_type = pooling_type
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
                        pooling_type ='cls', 
                        inbatch_loss=None,
                        model_tied=True,
                        use_pooler=False,
                        pooler_dim_in=768,
                        pooler_dim_out=768,
                        pooler_tied=True,
                          ) -> 'DotConfig':
        config = super().from_pretrained(model_name_or_path)
        config.model_name_or_path = model_name_or_path
        config.pooling_type = pooling_type
        config.inbatch_loss = inbatch_loss
        config.model_tied = model_tied
        config.use_pooler = use_pooler
        config.pooler_dim_in = pooler_dim_in
        config.pooler_dim_out = pooler_dim_out
        config.pooler_tied = pooler_tied
        return config

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
    model_type = "Dot"
    architecture_class = AutoModel
    config_class = DotConfig
    transformer_class = None
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
            'mean': lambda x: x.mean(dim=1),
            'cls' : lambda x: x[:, 0],
            'late_interaction': lambda x: x,
            'none': lambda x: x,
        }[config.pooling_type]
        self.pooling_type = config.pooling_type

        if config.use_pooler: self.pooler = Pooler(config) if pooler is None else pooler
        else: self.pooler = lambda x, y =True : x

        self.inbatch_loss_fn = config.inbatch_loss

        from .pyterrier.dot import DotTransformer
        self.transformer_class = DotTransformer


    def prepare_outputs(self, query_reps, docs_batch_reps, labels=None):
        batch_size = query_reps.size(0)

        if self.pooling_type == 'late_interaction':
            pred = emb_q @ emb_d.permute(0, 2, 1)
            pred = pred.max(1).values
            pred = pred.sum(-1)
        else:
            emb_q = query_reps.reshape(batch_size, 1, -1)
            emb_d = docs_batch_reps.reshape(batch_size, self.config.group_size, -1)
            pred = batched_dot_product(emb_q, emb_d)

        if self.config.inbatch_loss is not None:
            if self.pooling_type == 'late_interaction':
                inbatch_d = emb_d[:, 0]
                inbatch_pred = emb_q @ inbatch_d.permute(0, 2, 1)
                inbatch_pred = inbatch_pred.max(1).values
                inbatch_pred = inbatch_pred.sum(-1)
            else:
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
        self.model.load_state_dict(self.architecture_class.from_pretrained(model_dir).state_dict())
        if not self.config.model_tied: self.model_d.load_state_dict(self.architecture_class.from_pretrained(model_dir + "/model_d").state_dict())
        if self.config.use_pooler: self.pooler.load_state_dict(self.architecture_class.from_pretrained(model_dir + "/pooler").state_dict())

    @classmethod
    def from_pretrained(cls, model_name_or_path, config = None, **kwargs) -> "Dot":
        """Load model"""
        if os.path.isdir(model_name_or_path):
            config = cls.config_class.from_pretrained(model_name_or_path) if config is None else config
            model = cls.architecture_class.from_pretrained(model_name_or_path, **kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            model_d = None if config.model_tied else cls.architecture_class.from_pretrained(model_name_or_path + "/model_d", **kwargs) 
            pooler = None if not config.use_pooler else Pooler.from_pretrained(model_name_or_path + "/pooler")

            return cls(model, tokenizer, config, model_d, pooler)
        config = cls.config_class(model_name_or_path, **kwargs) if config is None else config
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = cls.architecture_class.from_pretrained(model_name_or_path)
        return cls(model, tokenizer, config)
    
    def to_pyterrier(self) -> "DotTransformer":
        return self.transformer_class.from_model(self, self.tokenizer, text_field='text')

AutoConfig.register("Dot", DotConfig)
AutoModel.register(DotConfig, Dot)