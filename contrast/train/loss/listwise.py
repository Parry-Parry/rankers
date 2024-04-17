import torch
from torch import Tensor
from torch.nn import functional as F
from . import reduce

def KL_DivergenceLoss(pred : Tensor, labels : Tensor, reduction='batchmean', **kwargs):
    return F.kl_div(F.log_softmax(pred, dim=1), F.softmax(labels, dim=1), reduction=reduction)

def RankNetLoss(pred : Tensor, labels : Tensor, reduction='mean', **kwargs):
    # ensure both are probabilities and if not apply softmax 
    if not torch.all((labels >= 0) & (labels <= 1)):
        labels = F.softmax(labels, dim=1)
    if not torch.all((pred >= 0) & (pred <= 1)):
        pred = F.softmax(pred, dim=1)
    return F.soft_margin_loss(pred, labels, reduction=reduction)

def ListNetLoss(pred : Tensor, labels : Tensor, reduction='mean', **kwargs):
    if not torch.all((labels >= 0) & (labels <= 1)):
        labels = F.softmax(labels, dim=1)
    return reduce(-torch.sum(labels * F.log_softmax(pred, dim=1), dim=-1), reduction)