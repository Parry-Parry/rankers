from rankers.modelling.dot import Pooler
import torch
from transformers import (
    AutoModel,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForMaskedLM,
)
from ..dot import DotConfig, Dot
from ..._optional import is_pyterrier_available
from torch.nn import functional as F


class SparseConfig(DotConfig):
    model_type = "Sparse"

    def __init__(
        self,
        model_name_or_path: str = "bert-base-uncased",
        query_processing: str = "splade_max",
        doc_processing: str = "splade_max",
        pooling_type="cls",
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
        self.query_processing = query_processing
        self.doc_processing = doc_processing

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "bert-base-uncased",
        query_processing: str = "splade_max",
        doc_processing: str = "splade_max",
        pooling_type="cls",
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
        config.query_processing = query_processing
        config.doc_processing = doc_processing
        return config


def splade_max(outputs, mask):
    outputs = outputs.logits
    values, _ = torch.max(torch.log(1 + F.relu(outputs)) * mask.unsqueeze(-1), dim=1)
    return values


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

        self.query_processing = (
            splade_max
            if config.query_processing == "splade_max"
            else lambda x, y: x.logits
        )
        self.doc_processing = (
            splade_max
            if config.doc_processing == "splade_max"
            else lambda x, y: x.logits
        )

        if is_pyterrier_available():
            from ...pyterrier.sparse import SparseTransformer

            self.transformer_class = SparseTransformer

    def _encode_d(self, **text):
        return self.doc_processing(self.model_d(**text), text["attention_mask"])

    def _encode_q(self, **text):
        return self.query_processing(self.model(**text), text["attention_mask"])

    def forward(self, loss=None, queries=None, docs_batch=None, labels=None):
        """Compute the loss given (queries, docs, labels)"""
        queries = (
            {k: v.to(self.model.device) for k, v in queries.items()}
            if queries is not None
            else None
        )
        docs_batch = (
            {k: v.to(self.model_d.device) for k, v in docs_batch.items()}
            if docs_batch is not None
            else None
        )
        labels = labels.to(self.model_d.device) if labels is not None else None

        query_reps = self._encode_q(**queries) if queries is not None else None
        docs_batch_reps = (
            self._encode_d(**docs_batch) if docs_batch is not None else None
        )

        pred, labels, inbatch_pred = self.prepare_outputs(
            query_reps, docs_batch_reps, labels
        )
        inbatch_loss = (
            self.inbatch_loss_fn(
                inbatch_pred, torch.eye(inbatch_pred.shape[0]).to(inbatch_pred.device)
            )
            if (inbatch_pred is not None and self.config.inbatch_loss is not None)
            else 0.0
        )

        loss_value = (
            loss(pred, labels, query_reps, docs_batch_reps)
            if labels is not None
            else loss(pred, None, query_reps, docs_batch_reps)
        )
        if len(loss_value) == 2:
            loss_value, to_log = loss_value
        else:
            to_log = {}
        to_log["inbatch_loss"] = inbatch_loss
        loss_value += inbatch_loss
        return (loss_value, to_log, pred)


AutoConfig.register("Sparse", SparseConfig)
AutoModel.register(SparseConfig, Sparse)
