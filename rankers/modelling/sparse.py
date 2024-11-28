from rankers.modelling.dot import Pooler
import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForMaskedLM,
)
from .dot import DotConfig, Dot, DotOutput
from torch.nn import functional as F


class ProcessingConstructor:
    def __init__(self,
                 norm = "none",
                 activation = "relu",
                 aggregation = "max",
                 ) -> None:
        self._norm = {
            'none': nn.Identity(),
            'log1p': lambda x: torch.log(1 + x),
        }[norm]
        self._activation = {
            'none': nn.Identity(),
            'relu': nn.ReLU(),
        }[activation]
        self._aggregation = {
            'max': lambda x: torch.max(x, dim=1).values,
            'mean': lambda x: torch.mean(x, dim=1),
            'sum': lambda x: torch.sum(x, dim=1),
        }[activation]

    def _get_norm(self, norm_value):
        if norm_value == "log1p":
            return lambda x: torch.log(1 + x)
        else:
            return nn.Identity()
    
    def _get_activation(self, activation_value):
        if activation_value == "relu":
            return nn.ReLU()
        else:
            return nn.Identity()
    
    def _get_aggregation(self, aggregation_value):
        if aggregation_value == "max":
            return lambda x: torch.max(x, dim=1).values
        elif aggregation_value == "mean":
            return lambda x: torch.mean(x, dim=1)
        elif aggregation_value == "sum":
            return lambda x: torch.sum(x, dim=1)
        else:
            return nn.Identity()
    
    def __call__(self, x, mask):
        post_act = self._activation(x)
        norm = self._norm(post_act)
        masked = norm * mask.unsqueeze(-1)
        return self._aggregation(masked)


class SparseConfig(DotConfig):
    model_type = "Sparse"

    def __init__(
        self,
        model_name_or_path: str = "bert-base-uncased",
        query_norm="log1p",
        query_activation="relu",
        query_aggregation="max",
        doc_norm="log1p",
        doc_activation="relu",
        doc_aggregation="max",
        pooling_type="none",
        inbatch_loss=None,
        model_tied=True,
        use_pooler=False,
        pooler_dim_in=768,
        pooler_dim_out=768,
        pooler_tied=True,
        **kwargs,
    ):
        super().__init__(
            model_name_or_path,
            pooling_type,
            inbatch_loss,
            model_tied,
            use_pooler,
            pooler_dim_in,
            pooler_dim_out,
            pooler_tied,
            **kwargs,
        )
        self.query_activation = query_activation
        self.query_norm = query_norm
        self.query_aggregation = query_aggregation
        self.doc_activation = doc_activation
        self.doc_norm = doc_norm
        self.doc_aggregation = doc_aggregation

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "bert-base-uncased",
        query_activation="relu",
        query_norm="log1p",
        query_aggregation="max",
        doc_activation="relu",
        doc_norm="log1p",
        doc_aggregation="max",
        pooling_type="none",
        inbatch_loss=None,
        model_tied=True,
        use_pooler=False,
        pooler_dim_in=768,
        pooler_dim_out=768,
        pooler_tied=True,
    ) -> "SparseConfig":
        config = super().from_pretrained(
            model_name_or_path,
            pooling_type,
            inbatch_loss,
            model_tied,
            use_pooler,
            pooler_dim_in,
            pooler_dim_out,
            pooler_tied,
        )
        config.query_activation = query_activation
        config.query_norm = query_norm
        config.query_aggregation = query_aggregation
        config.doc_activation = doc_activation
        config.doc_norm = doc_norm
        config.doc_aggregation = doc_aggregation
        return config


def splade_max(outputs, mask):
    outputs = outputs.logits
    post_act = F.relu(outputs)
    norm = torch.log(1 + post_act)
    masked = norm * mask.unsqueeze(-1)
    return torch.max(masked, dim=1).values


class Sparse(Dot):
    model_type = "Sparse"
    architecture_class = AutoModelForMaskedLM
    config_class = SparseConfig
    transformer_class = None

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: DotConfig,
        model_d: PreTrainedModel = None,
        pooler: Pooler = None,
    ):
        super().__init__(model, tokenizer, config, model_d, pooler)

        self.query_processing = ProcessingConstructor(norm=config.query_norm, 
                                                    activation=config.query_activation, 
                                                    aggregation=config.query_aggregation)
        self.doc_processing = ProcessingConstructor(norm=config.doc_norm,
                                                    activation=config.doc_activation,
                                                    aggregation=config.doc_aggregation)

        from .pyterrier.sparse import SparseTransformer

        self.transformer_class = SparseTransformer

    def _encode_d(self, **text):
        return self.doc_processing(self.model_d(**text), text["attention_mask"])

    def _encode_q(self, **text):
        return self.query_processing(self.model(**text), text["attention_mask"])

    def forward(self, loss=None, queries=None, texts=None, labels=None):
        """Compute the loss given (queries, docs, labels)"""
        queries = (
            {k: v.to(self.model.device) for k, v in queries.items()}
            if queries is not None
            else None
        )
        texts = (
            {k: v.to(self.model_d.device) for k, v in texts.items()}
            if texts is not None
            else None
        )
        labels = labels.to(self.model_d.device) if labels is not None else None

        query_hidden_states = self._encode_q(**queries) if queries is not None else None
        text_hidden_states = self._encode_d(**texts) if texts is not None else None

        pred, labels, inbatch_pred = self.prepare_outputs(
            query_hidden_states, text_hidden_states, labels
        )
        inbatch_loss = (
            self.inbatch_loss_fn(
                inbatch_pred, torch.eye(inbatch_pred.shape[0]).to(inbatch_pred.device)
            )
            if (inbatch_pred is not None and self.config.inbatch_loss is not None)
            else 0.0
        )

        output = DotOutput(
            scores=pred,
            labels=labels,
            query_hidden_states=query_hidden_states,
            text_hidden_states=text_hidden_states,
            loss=inbatch_loss,
        )

        loss_value = (
            loss(
                pred=output.scores,
                labels=output.labels,
                query_hidden_states=query_hidden_states,
                text_hidden_states=text_hidden_states,
            )
            if loss is not None
            else 0.0
        )
        output.loss += loss_value
        return output


AutoConfig.register("Sparse", SparseConfig)
AutoModel.register(SparseConfig, Sparse)
