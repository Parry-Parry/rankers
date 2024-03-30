from transformers import PreTrainedModel, AutoModel, PretrainedConfig
import torch
from ..inference.models import DotTransformer, PoolingType

class DotConfig(PretrainedConfig):
    model_type = "dot"
    def __init__(self, model_name_or_path : str , mode='cls', **kwargs):
        self.mode = mode
        super().__init__(model_name_or_path, **kwargs)

class Dot(PreTrainedModel):
    def __init__(
        self,
        encoder: PreTrainedModel,
        config: DotConfig,
    ):
        super().__init__(config)
        self.encoder = encoder
        self.pooling_type = config.mode
        self.pooling = {
            PoolingType.MEAN: self._mean,
            PoolingType.CLS: self._cls,
            PoolingType.LATE_INTERACTION: self._late_interaction
        }[config.mode]
    
    def _cls(self, x : torch.Tensor, **kwargs) -> torch.Tensor:
        return x[:,0,:]
    
    def _mean(self, x : torch.Tensor, **kwargs) -> torch.Tensor:
        return x.mean(dim=1)
    
    def _late_interaction(self, x : torch.Tensor, mask : torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError("Late interaction pooling is not implemented yet, use pyterrier_colbert")
        
    def encode(self, document : bool = False, **text):
        if document: return self.pooling(self.encoder(**text)[0])
        return self.pooling(self.encoder(**text)[0])

    def forward(self, loss, queries, docs_batch, labels=None):
        """Compute the loss given (queries, docs, labels)"""
        queries = {k: v.to(self.encoder.device) for k, v in queries.items()}
        docs_batch = {k: v.to(self.encoder.device) for k, v in docs_batch.items()}
        labels = labels.to(self.encoder.device) if labels is not None else None
        q_reps = self.encode(**queries)
        docs_batch_rep = self.encode(**docs_batch)
        if labels is None:
            output = loss(q_reps, docs_batch_rep)
        else:
            output = loss(q_reps, docs_batch_rep, labels)
        return output

    def save_pretrained(self, model_dir):
        """Save both query and document encoder"""
        self.config.save_pretrained(model_dir)
        self.encoder.save_pretrained(model_dir)
    
    def load_state_dict(self, model_dir):
        """Load state dict from a directory"""
        return self.encoder.load_state_dict(AutoModel.from_pretrained(model_dir).state_dict())
    
    def eval(self) -> DotTransformer:
        return DotTransformer.from_model(self.encoder, text_field='text')

    @classmethod
    def from_pretrained(cls, model_dir_or_name, mode='cls'):
        """Load encoder from a directory"""
        config = DotConfig.from_pretrained(mode, model_dir_or_name)
        encoder = AutoModel.from_pretrained(model_dir_or_name)
        return cls(encoder, config)