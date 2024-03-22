from transformers import PreTrainedModel, AutoModel, PretrainedConfig
from ..inference.models import DotTransformer

# TODO: Fix the dotconfig

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
        self.pooling = lambda x: x.mean(dim=1) if config.mode == 'mean' else x[:,0,:]
    
    def encode(self, **text):
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
        config = DotConfig.from_pretrained(model_dir_or_name)
        encoder = AutoModel.from_pretrained(model_dir_or_name)
        return cls(encoder, config)