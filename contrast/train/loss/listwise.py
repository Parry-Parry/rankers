from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

def kl_divergence(pred : Tensor, labels : Tensor, **kwargs):
    kl = nn.KLDivLoss(reduction='mean')
    return kl(F.log_softmax(pred, dim=1), F.softmax(labels, dim=1))

