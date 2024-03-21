import torch
from torch import Tensor

def pointwiseMSE(pred : Tensor, labels : Tensor, **kwargs):
    pred = pred.view(-1)
    labels = labels.view(-1)
    mse = torch.nn.MSELoss(reduction="mean")
    return mse(pred, labels)