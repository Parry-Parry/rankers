from torch import Tensor
import torch
import torch.nn.functional as F

def marginMSE(pred : Tensor, labels : Tensor, **kwargs):
    mse = torch.nn.MSELoss(reduction="mean")

    pred_a = pred[:, 0]
    pred_b = pred[:, 1:]

    labels_a = labels[:, 0]
    labels_b = labels[:, 1:]

    margin_a = pred_a.unsqueeze(1) - pred_b
    margin_b = labels_a.unsqueeze(1) - labels_b

    return mse(margin_a, margin_b)

def hinge(pred : Tensor, labels : Tensor, margin : int = 1, **kwargs):
    pred = F.sigmoid(pred)
    labels = F.sigmoid(labels)

    pred_a = pred[:, 0]
    pred_b = pred[:, 1:]
    loss = torch.zeros(1, requires_grad=True).to(pred.device)

    for i in range(pred.size(0)):
        for j in range(pred_b.size(1)):
            if j==0:
                continue
            _pred = pred_a[i] - pred_b[i, j]
            _labels = labels[i, 0] - labels[i, j+1]
            loss += F.hinge_embedding_loss(_pred, _labels, margin=margin, reduction='noe')
    return loss / (pred.size(0) * (pred_b.size(1)))

def clear(pred : Tensor, labels : Tensor, margin : int = 1, **kwargs):
    pred_a = pred[:, 0]
    pred_b = pred[:, 1:]

    labels_a = labels[:, 0]
    labels_b = labels[:, 1:]

    margin_b = margin - labels_a.unsqueeze(1) - labels_b

    return torch.mean(F.relu(margin_b - pred_a.unsqueeze(1) + pred_b))

def contrastive(pred : Tensor, **kwargs):
    softmax_scores = F.log_softmax(pred, dim=1)
    return F.nll_loss(softmax_scores, torch.zeros(pred.size(0), dtype=torch.long, device=pred.device), reduction='mean')

