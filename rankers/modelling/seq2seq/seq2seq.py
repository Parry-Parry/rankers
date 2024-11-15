import pyterrier as pt
from transformers import PreTrainedModel, PreTrainedConfig, PreTrainedTokenizer, AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from typing import Union
import torch
import pandas as pd
from more_itertools import chunked
import numpy as np
from ..._util import not_tested
from ..base import Ranker

DEFAULT_MONO_PROMPT = r'query: {query} document: {text} relevant:'
DEFAULT_DUO_PROMPT = r'query: {query} positive: {text} negative: {text} relevant:'

class Seq2SeqConfig(PreTrainedConfig):
    model_type = 'Seq2Seq'
    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs) -> 'Seq2SeqConfig':
        config = super().from_pretrained(model_name_or_path, **kwargs)
        return config

@not_tested
class Seq2SeqTransformer(pt.Transformer):
    architecture_class = AutoModelForSeq2SeqLM
    config_class = Seq2SeqConfig
    def __init__(self, 
                 model : PreTrainedModel, 
                 tokenizer : PreTrainedTokenizer, 
                 config : PreTrainedConfig, 
                 batch_size : int, 
                 text_field : str = 'text', 
                 device : Union[str, torch.device] = None,
                 pos_token : str = 'true',
                 neg_token : str = 'false',
                 prompt : str = None,
                 verbose : bool = False
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
        self.verbose = verbose
    
    @classmethod
    def from_pretrained(cls, 
                        model_name_or_path : str, 
                        batch_size : int = 64, 
                        text_field : str = 'text', 
                        device : Union[str, torch.device] = None,
                        prompt : str = None,
                        verbose : bool = False,
                        **kwargs
                        ):
        model = cls.architecture_class.from_pretrained(model_name_or_path, **kwargs).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        config = AutoConfig.from_pretrained(model_name_or_path)
        return cls(model, tokenizer, config, batch_size, text_field, device, prompt, verbose=verbose)

    @classmethod 
    def from_model(cls, 
                   model : PreTrainedModel, 
                   tokenizer : PreTrainedTokenizer,
                   batch_size : int = 64, 
                   text_field : str = 'text', 
                   verbose : bool = False
                   ): 
        config = model.config
        return cls(model, tokenizer, config, batch_size, text_field, model.device, verbose=verbose)
    
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
    architecture_class = AutoModelForSeq2SeqLM
    config_class = Seq2SeqConfig
    def __init__(self,
                    model : PreTrainedModel,
                    tokenizer : PreTrainedTokenizer,
                    config : PreTrainedConfig,
                    batch_size : int,
                    text_field : str = 'text',
                    device : Union[str, torch.device] = None,
                    pos_token : str = 'true',
                    neg_token : str = 'false',
                    prompt : str = None,
                    verbose : bool = False
                    ) -> None:
            raise NotImplementedError("Incomplete, do not use")
            super().__init__(model, tokenizer, config, batch_size, text_field, device, pos_token, neg_token, prompt, verbose)
            self.prompt = prompt if prompt is not None else DEFAULT_DUO_PROMPT

    def transform(self, inp : pd.DataFrame) -> pd.DataFrame:
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
@not_tested
class Seq2Seq(Ranker):
    """Wrapper for ConditionalGenerationCat Model
    
    Parameters
    ----------
    model : AutoModelForSeq2SeqLM
        the underlying HF model
    config : AutoConfig
        the configuration for the model
    """
    model_type = 'Seq2Seq'
    architecture_class = AutoModelForSeq2SeqLM
    config_class = Seq2SeqConfig
    transformer_class = Seq2SeqTransformer
    
class CausalLMConfig(PreTrainedConfig):
    model_type = 'CausalLM'
    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs) -> 'CausalLMConfig':
        config = super().from_pretrained(model_name_or_path, **kwargs)
        return config

class CausalLMTransformer(Seq2SeqTransformer):
    architecture_class = AutoModelForCausalLM
    config_class = CausalLMConfig
    def __init__(self, 
                 model : PreTrainedModel, 
                 tokenizer : PreTrainedTokenizer, 
                 config : PreTrainedConfig, 
                 batch_size : int, 
                 text_field : str = 'text', 
                 device : Union[str, torch.device] = None,
                 prompt : str = None,
                 verbose : bool = False
                 ) -> None:
        raise NotImplementedError("Incomplete, do not use")
        super().__init__(model, tokenizer, config, batch_size, text_field, device, prompt, verbose)
    
    @classmethod
    def from_pretrained(cls, 
                        model_name_or_path : str, 
                        batch_size : int = 64, 
                        text_field : str = 'text', 
                        config : PreTrainedConfig = None,
                        device : Union[str, torch.device] = None,
                        prompt : str = None,
                        verbose : bool = False,
                        **kwargs
                        ):
        config = cls.config_class.from_pretrained(model_name_or_path) if config is None else config
        model = cls.architecture_class.from_pretrained(model_name_or_path, **kwargs).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return cls(model, tokenizer, config, batch_size, text_field, device, prompt, verbose=verbose)

    @classmethod 
    def from_model(cls, 
                   model : PreTrainedModel, 
                   tokenizer : PreTrainedTokenizer,
                   batch_size : int = 64, 
                   text_field : str = 'text', 
                   verbose : bool = False
                   ): 
        config = model.config
        return cls(model, tokenizer, config, batch_size, text_field, model.device, verbose=verbose)
    
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
                scores.append(self.model(**inps).logits[:, 0].cpu().detach().numpy())
        res = inp.assign(score=np.concatenate(scores))
        pt.model.add_ranks(res)
        res = res.sort_values(['qid', 'rank'])
        return res

class CausalLM(Seq2Seq):
    """Wrapper for CausalLM Model
    
    Parameters
    ----------
    model : AutoModelForCausalLM
        the underlying HF model
    tokenizer : PreTrainedTokenizer
        the tokenizer for the model
    config : AutoConfig
        the configuration for the model
    """
    model_type = 'CausalLM'
    architecture_class = AutoModelForCausalLM
    transformer_class = CausalLMTransformer
    config_class = CausalLMConfig
    def __init__(self, model, tokenizer, config):
        raise NotImplementedError("Incomplete, do not use")
        super().__init__(model, tokenizer, config)

AutoModelForSeq2SeqLM.register("Seq2Seq", Seq2Seq)
AutoModelForCausalLM.register("CausalLM", CausalLM)
AutoConfig.register("Seq2Seq", Seq2SeqConfig)
AutoConfig.register("CausalLM", CausalLMConfig)