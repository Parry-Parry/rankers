from torch import Tensor
import torch
import torch.nn.functional as F
from . import reduce

residual = lambda x : x[:, 0].unsqueeze(1) - x[:, 1:]

def MarginMSELoss(pred : Tensor, labels : Tensor, reduction='mean', **kwargs):
    return F.mse_loss(residual(pred), residual(labels), reduction=reduction)

def HingeLoss(pred : Tensor, labels : Tensor, margin : int = 1, reduction='mean', **kwargs):
    pred_residuals = residual(F.sigmoid(pred))
    label_residuals = torch.sign(residual(F.sigmoid(labels)))

    return reduce(F.relu(margin - (label_residuals * pred_residuals)), reduction)

def ClearLoss(pred : Tensor, labels : Tensor, margin : int = 1, reduction='mean', **kwargs):
    margin_b = margin - residual(labels)
    return reduce(F.relu(margin_b - residual(pred)), reduction)

def ContrastiveLoss(pred : Tensor, reduction='mean', **kwargs):
    softmax_scores = F.log_softmax(pred, dim=1)
    return F.nll_loss(softmax_scores, torch.zeros(pred.size(0), dtype=torch.long, device=pred.device), reduction=reduction)

