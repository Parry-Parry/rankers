from transformers import PreTrainedModel, PreTrainedTokenizer, AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from transformers.utils import OptionalDependencyNotAvailable
from typing import Union
import torch
import pandas as pd
from more_itertools import chunked
import numpy as np
from ..._optional import is_pyterrier_available

PT_AVAILIBLE = is_pyterrier_available()

if PT_AVAILIBLE:
    import pyterrier as pt
    if not pt.started():
        pt.init()


DEFAULT_MONO_PROMPT = r'query: {query} document: {text} relevant:'
DEFAULT_DUO_PROMPT = r'query: {query} positive: {text} negative: {text} relevant:'

class Seq2Seq(PreTrainedModel):
    """Wrapper for ConditionalGenerationCat Model
    
    Parameters
    ----------
    classifier : AutoModelForSeq2SeqLM
        the classifier model
    config : AutoConfig
        the configuration for the model
    """
    model_architecture = 'Seq2Seq'
    def __init__(
        self,
        classifier: AutoModelForSeq2SeqLM,
        tokenizer: PreTrainedTokenizer,
        config: AutoConfig,
    ):
        super().__init__(config)
        self.classifier = classifier
        self.tokenizer = tokenizer

    def prepare_outputs(self, logits):
        raise NotImplementedError

    def forward(self, loss, sequences, labels=None):
        """Compute the loss given (pairs, labels)"""
        sequences = {k: v.to(self.classifier.device) for k, v in sequences.items()}
        labels = labels.to(self.classifier.device) if labels is not None else None
        logits = self.classifier(**sequences).logits
        pred = self.prepare_outputs(logits)
        loss_value = loss(pred) if labels is None else loss(pred, labels)
        return (loss_value, pred)


    def save_pretrained(self, model_dir, **kwargs):
        """Save classifier"""
        self.config.save_pretrained(model_dir)
        self.classifier.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
    
    def load_state_dict(self, model_dir):
        """Load state dict from a directory"""
        return self.classifier.load_state_dict(AutoModelForSeq2SeqLM.from_pretrained(model_dir).state_dict())
    
    def to_pyterrier(self) -> "Seq2SeqTransformer":
        if not PT_AVAILIBLE: raise OptionalDependencyNotAvailable()
        return Seq2SeqTransformer.from_model(self.classifier, self.tokenizer, text_field='text')

    @classmethod
    def from_pretrained(cls, model_dir_or_name : str, num_labels=2):
        """Load classifier from a directory"""
        config = AutoConfig.from_pretrained(model_dir_or_name)
        classifier = AutoModelForSeq2SeqLM.from_pretrained(model_dir_or_name, num_labels=num_labels)
        return cls(classifier, config)

class Seq2SeqTransformer(pt.Transformer):
    def __init__(self, 
                 model : PreTrainedModel, 
                 tokenizer : PreTrainedTokenizer, 
                 config : AutoConfig, 
                 batch_size : int, 
                 text_field : str = 'text', 
                 device : Union[str, torch.device] = None,
                 pos_token : str = 'true',
                 neg_token : str = 'false',
                 prompt : str = None
                 ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.batch_size = batch_size
        self.text_field = text_field
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pos_token = self.tokenizer.encode(pos_token)[0]
        self.neg_token = self.tokenizer.encode(neg_token)[0]
        self.prompt = prompt if prompt is not None else DEFAULT_MONO_PROMPT
    
    @classmethod
    def from_pretrained(cls, 
                        model_name_or_path : str, 
                        batch_size : int = 64, 
                        text_field : str = 'text', 
                        device : Union[str, torch.device] = None
                        ):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        config = AutoConfig.from_pretrained(model_name_or_path)
        return cls(model, tokenizer, config, batch_size, text_field, device)

    @classmethod 
    def from_model(cls, 
                   model : PreTrainedModel, 
                   tokenizer : PreTrainedTokenizer,
                   batch_size : int = 64, 
                   text_field : str = 'text', 
                   ): 
        config = model.config
        return cls(model, tokenizer, config, batch_size, text_field, model.device)
    
    def transform(self, inp : pd.DataFrame) -> pd.DataFrame:
        scores = []
        it = inp[['query', self.text_field]].itertuples(index=False)
        if self.verbose:
            it = pt.tqdm(it, total=len(inp), unit='record', desc='Cat scoring')
        with torch.no_grad():
            for chunk in chunked(it, self.batch_size):
                queries, texts = map(list, zip(*chunk))
                prompts = [self.prompt.format(query=q, text=t) for q, t in zip(queries, texts)]
                inps = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
                inps = {k: v.to(self.device) for k, v in inps.items()}
                scores.append(self.model(**inps).logits[:, (self.pos_token, self.neg_token)].softmax(dim=-1)[0].cpu().detach().numpy())
        res = inp.assign(score=np.concatenate(scores))
        pt.model.add_ranks(res)
        res = res.sort_values(['qid', 'rank'])
        return res

class Seq2SeqDuoTransformer(Seq2SeqTransformer):
    def __init__(self,
                    model : PreTrainedModel,
                    tokenizer : PreTrainedTokenizer,
                    config : AutoConfig,
                    batch_size : int,
                    text_field : str = 'text',
                    device : Union[str, torch.device] = None,
                    pos_token : str = 'true',
                    neg_token : str = 'false',
                    prompt : str = None
                    ) -> None:
            super().__init__(model, tokenizer, config, batch_size, text_field, device, pos_token, neg_token, prompt)
            self.prompt = prompt if prompt is not None else DEFAULT_DUO_PROMPT

    def transform(self, inp : pd.DataFrame) -> pd.DataFrame:
        # TODO: Fix this mess
        scores = []
        it = inp[['query', self.text_field]].itertuples(index=False)
        if self.verbose:
            it = pt.tqdm(it, total=len(inp), unit='record', desc='Cat scoring')
        with torch.no_grad():
            for chunk in chunked(it, self.batch_size):
                queries, texts = map(list, zip(*chunk))
                prompts = [self.prompt.format(query=q, text1=t1, text2=t2) for q, t1, t2 in zip(queries, texts, texts) if t1 != t2]
                inps = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
                inps = {k: v.to(self.device) for k, v in inps.items()}
                scores.append(self.model(**inps).logits[:, (self.pos_token, self.neg_token)].softmax(dim=-1)[0].cpu().detach().numpy())
        res = inp.assign(score=np.concatenate(scores))
        pt.model.add_ranks(res)
        res = res.sort_values(['qid', 'rank'])
        return res