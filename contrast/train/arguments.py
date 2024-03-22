from transformers import TrainingArguments
from typing import Union
import torch.nn as nn
    
class ContrastArguments(TrainingArguments):
    def __init__(self, 
                 loss_fn : Union[nn.Module, callable], 
                 mode : str = None, 
                 num_negatives : int = 1,
                 margin : int = 1,
                 **kwargs):
        self.loss_fn = loss_fn
        self.mode = mode
        self.num_negatives = num_negatives
        self.margin = margin
        super().__init__(**kwargs)