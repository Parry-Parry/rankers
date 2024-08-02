from dataclasses import dataclass
import torch
from collections import namedtuple

@dataclass
class BasicOutput(namedtuple):
    loss : torch.Tensor
    scores: torch.Tensor