from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer
from .dot import Dot, Pooler

class BGEConfig(PretrainedConfig):
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
    model_architecture = "BGE"
    def __init__(self, 
                 model_name_or_path : str='bert-base-uncased',
                 mode='cls', 
                 inbatch_loss = None,
                 encoder_tied=True,
                 use_pooler=False,
                 pooler_dim_in=768,
                 pooler_dim_out=768,
                 pooler_tied=True,
                 **kwargs):
        self.model_name_or_path = model_name_or_path
        self.mode = mode
        self.inbatch_loss = inbatch_loss
        self.encoder_tied = encoder_tied
        self.use_pooler = use_pooler
        self.pooler_dim_in = pooler_dim_in
        self.pooler_dim_out = pooler_dim_out
        self.pooler_tied = pooler_tied
        super().__init__(**kwargs)
    
    @classmethod
    def from_pretrained(cls, 
                        model_name_or_path : str='bert-base-uncased',
                        mode='cls', 
                        inbatch_loss = None,
                        encoder_tied=True,
                        use_pooler=False,
                        pooler_dim_in=768,
                        pooler_dim_out=768,
                        pooler_tied=True,
                          ) -> 'BGEConfig':
        config = super().from_pretrained(model_name_or_path)
        config.model_name_or_path = model_name_or_path
        config.mode = mode
        config.inbatch_loss = inbatch_loss
        config.encoder_tied = encoder_tied
        config.use_pooler = use_pooler
        config.pooler_dim_in = pooler_dim_in
        config.pooler_dim_out = pooler_dim_out
        config.pooler_tied = pooler_tied
        return config

class BGE(Dot):
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
        tokenizer : PreTrainedTokenizer,
        config : BGEConfig,
        encoder_d : PreTrainedModel = None,
        pooler : Pooler = None,
    ):
        raise NotImplementedError("Incomplete, do not use")
        super().__init__(config)
        self.encoder = encoder
        self.tokenizer = tokenizer
        if encoder_d: self.encoder_d = encoder_d
        else: self.encoder_d = self.encoder if config.encoder_tied else deepcopy(self.encoder)
        self.pooling = {
            'mean': self._mean,
            'cls' : self._cls,
            'none': lambda x: x,
        }[config.mode]

        if config.use_pooler: self.pooler = Pooler(config) if pooler is None else pooler
        else: self.pooler = lambda x, y =True : x

        if config.inbatch_loss is not None:
            if config.inbatch_loss not in loss.__all__:
                raise ValueError(f"Unknown loss: {config.inbatch_loss}")
            self.inbatch_loss_fn = getattr(loss, config.inbatch_loss)()
        else:
            self.inbatch_loss_fn = None