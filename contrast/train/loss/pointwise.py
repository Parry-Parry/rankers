from torch import Tensor
from torch.nn import functional as F

def PointwiseMSELoss(pred : Tensor, labels : Tensor, reduction='mean', **kwargs):
    return F.mse_loss(pred.view(-1), labels.view(-1), reduction=reduction)