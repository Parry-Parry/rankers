from rankers.modelling.dot import Pooler
import torch
from transformers import AutoModel, AutoConfig, AutoModel, PreTrainedModel, PreTrainedTokenizer
from .dot import DotConfig, Dot

class SparseConfig(DotConfig):
    model_type = "Sparse"
    def __init__(self, model_name_or_path: str = 'bert-base-uncased', pooling_type='cls', inbatch_loss=None, model_tied=True, use_pooler=False, pooler_dim_in=768, pooler_dim_out=768, pooler_tied=True, **kwargs):
        super().__init__(model_name_or_path, pooling_type, inbatch_loss, model_tied, use_pooler, pooler_dim_in, pooler_dim_out, pooler_tied, **kwargs)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str = 'bert-base-uncased', pooling_type='cls', inbatch_loss=None, model_tied=True, use_pooler=False, pooler_dim_in=768, pooler_dim_out=768, pooler_tied=True) -> 'SparseConfig':
        return cls(model_name_or_path, pooling_type, inbatch_loss, model_tied, use_pooler, pooler_dim_in, pooler_dim_out, pooler_tied)

class Sparse(Dot):
    model_type = "Sparse"
    transformer_class = None

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, config: DotConfig, model_d: PreTrainedModel = None, pooler: Pooler = None):
        super().__init__(model, tokenizer, config, model_d, pooler)

        from .pyterrier.sparse import SparseTransformer
        self.transformer_class = SparseTransformer

AutoConfig.register("Sparse", SparseConfig)
AutoModel.register(SparseConfig, Sparse)