from torch import Tensor
import torch
import torch.nn.functional as F

residual = lambda x : x[:, 0].unsqueeze(1) - x[:, 1:]

def MarginMSE(pred : Tensor, labels : Tensor, **kwargs):
    return F.mse_loss(residual(pred), residual(labels), reduction='mean')

def Hinge(pred : Tensor, labels : Tensor, margin : int = 1, **kwargs):
    pred_residuals = residual(F.sigmoid(pred))
    label_residuals = torch.sign(residual(F.sigmoid(labels)))

    return torch.mean(F.relu(margin - (label_residuals * pred_residuals)))

def Clear(pred : Tensor, labels : Tensor, margin : int = 1, **kwargs):
    margin_b = margin - residual(labels)
    return torch.mean(F.relu(margin_b - residual(pred)))

def Contrastive(pred : Tensor, **kwargs):
    softmax_scores = F.log_softmax(pred, dim=1)
    return F.nll_loss(softmax_scores, torch.zeros(pred.size(0), dtype=torch.long, device=pred.device), reduction='mean')

