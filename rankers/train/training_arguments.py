from transformers import TrainingArguments
from transformers.trainer_pt_utils import AcceleratorConfig
from transformers.utils import is_accelerate_available
from dataclasses import field, fields, dataclass
from typing import List, Optional
from enum import Enum
import os
from .. import is_ir_measures_available, seed_everything


def parse_ir_measure(measure : str):
    if not is_ir_measures_available(): return measure
    from ir_measures import parse_measure
    return parse_measure(measure)

def get_loss(loss_fn : str):
    from .loss import LOSS_REGISTRY
    if isinstance(loss_fn, str):
        if loss_fn not in LOSS_REGISTRY.available: raise ValueError(f"Unknown loss: {loss_fn}, choices are {LOSS_REGISTRY.available}")
        return LOSS_REGISTRY.get(loss_fn)
    else:
        return loss_fn

@dataclass
class RankerTrainingArguments(TrainingArguments):
    group_size : Optional[int] = field(
        default=2,
        metadata={"help": "Number of documents per query"}
    )
    eval_metrics : Optional[List[str]] = field(
        default_factory=lambda: [],
        metadata={"help": "Evaluation metrics"}
    )
    wandb_project : Optional[str] = field(
        default=None,
        metadata={"help": "Wandb project name"}
    )
    loss_fn : Optional[str] = field(
        default='lce',
        metadata={"help": "Loss function to use"}
    )

    def __post_init__(self):
        super().__post_init__()
        seed_everything(self.seed)

        if self.wandb_project is not None:
            os.environ['WANDB_PROJECT'] = self.wandb_project
        assert self.group_size > 0, "Group size must be greater than 0"
        
        self.eval_ir_metrics = [parse_ir_measure(metric) for metric in self.eval_metrics] if len(self.eval_metrics) > 0 else None
        self.loss_fn = get_loss(self.loss_fn)