from torch import Tensor
from torch.nn import functional as F

def KL_DivergenceLoss(pred : Tensor, labels : Tensor, **kwargs):
    return F.kl_div(F.log_softmax(pred, dim=1), F.softmax(labels, dim=1), reduction='batchmean')