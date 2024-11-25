from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    PreTrainedTokenizer,
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)
from typing import Union
import torch
import pandas as pd
from more_itertools import chunked
import numpy as np
from ..._util import not_tested
from ..base import Ranker

DEFAULT_MONO_PROMPT = r"query: {query} document: {text} relevant:"
DEFAULT_DUO_PROMPT = r"query: {query} positive: {text} negative: {text} relevant:"


class Seq2SeqConfig(PretrainedConfig):
    model_type = "Seq2Seq"

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs) -> "Seq2SeqConfig":
        config = super().from_pretrained(model_name_or_path, **kwargs)
        return config


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

    model_type = "Seq2Seq"
    architecture_class = AutoModelForSeq2SeqLM
    config_class = Seq2SeqConfig
    transformer_class =  None


class CausalLMConfig(PretrainedConfig):
    model_type = "CausalLM"

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs) -> "CausalLMConfig":
        config = super().from_pretrained(model_name_or_path, **kwargs)
        return config


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

    model_type = "CausalLM"
    architecture_class = AutoModelForCausalLM
    transformer_class = None
    config_class = CausalLMConfig

    def __init__(self, model, tokenizer, config):
        raise NotImplementedError("Incomplete, do not use")
        super().__init__(model, tokenizer, config)


AutoModelForSeq2SeqLM.register("Seq2Seq", Seq2Seq)
AutoModelForCausalLM.register("CausalLM", CausalLM)
AutoConfig.register("Seq2Seq", Seq2SeqConfig)
AutoConfig.register("CausalLM", CausalLMConfig)
