from torch import Tensor
from torch.nn import functional as F

def PointwiseMSE(pred : Tensor, labels : Tensor, **kwargs):
    return F.mse_loss(pred.view(-1), labels.view(-1), reduction='mean')