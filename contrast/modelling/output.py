from dataclasses import dataclass
import torch

@dataclass
class BasicOutput:
    loss : torch.Tensor
    scores: torch.Tensor