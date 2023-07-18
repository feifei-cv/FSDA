import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import copy



def ent(logits, temperature):

    B, C = logits.shape
    epsilon = 1e-5
    pred = F.softmax(logits / temperature, dim=1)  ##
    entropy_weight = entropy(pred).detach()
    entropy_weight = 1 + torch.exp(-entropy_weight)
    entropy_weight = (B * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)
    weight_sum = torch.sum(pred * entropy_weight, dim=0).unsqueeze(dim=0)
    return torch.sum(torch.sum(-pred * torch.log(pred + epsilon) / weight_sum * entropy_weight, dim=-1))

def entropy_loss(predictions, reduction='none'):

    predictions = F.softmax(predictions, dim=1)  ##
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H

def kld(logits, num_classes, temperature):

    B, C = logits.shape
    epsilon = 1e-5
    pred = F.softmax(logits / temperature, dim=1)
    # entropy_weight = entropy(pred).detach()
    # entropy_weight = 1 + torch.exp(-entropy_weight)
    # entropy_weight = (B * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)
    # weight_sum = torch.sum(pred * entropy_weight, dim=0).unsqueeze(dim=0)

    #### mini entropy rank
    # indices = torch.argsort(entropy_weight, dim=0, descending=True).squeeze()
    # ratio = 0.8
    # indice_high = indices[:int(len(indices) * ratio)]
    # indice_low = indices[int(len(indices) * ratio):int(len(indices))]
    # H1 = torch.sum(torch.sum(-pred[indice_low] * torch.log(pred[indice_low] + epsilon) / weight_sum * entropy_weight[indice_low],dim=-1))
    # H2 = torch.sum(torch.sum(-torch.log(pred[indice_high] + epsilon) / (num_classes * weight_sum) * entropy_weight[indice_high], dim=-1))
    # H = H1+H2
    return torch.sum(torch.sum(-torch.log(pred + epsilon)/num_classes,dim=-1))

def l2(logits, temperature):

    B, C = logits.shape
    pred = F.softmax(logits / temperature, dim=1)  ##
    entropy_weight = entropy(pred).detach()
    entropy_weight = 1 + torch.exp(-entropy_weight)
    entropy_weight = (B * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)
    weight_sum = torch.sum(pred * entropy_weight, dim=0).unsqueeze(dim=0)
    return torch.sum(torch.sum(pred**2/ weight_sum*entropy_weight, dim=-1))

def reg(logits, temperature):
    B, C = logits.shape
    pred = F.softmax(logits / temperature, dim=1)  ##
    entropy_weight = entropy(pred).detach()
    entropy_weight = 1 + torch.exp(-entropy_weight)
    entropy_weight = (B * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)
    weight_sum = torch.sum(pred * entropy_weight, dim=0).unsqueeze(dim=0)
    return torch.sum((1 - torch.sum(pred ** 2 / weight_sum * entropy_weight, dim=-1)))


def weightAnchor(logits, temperature):
    B, C = logits.shape
    pred = F.softmax(logits / temperature, dim=1)  ##
    entropy_weight = entropy(pred).detach()
    entropy_weight = 1 + torch.exp(-entropy_weight)
    entropy_weight = (B * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)
    return entropy_weight


def _l2_normalize(d):
    d = d.numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2, 3))).reshape((-1, 1, 1, 1)) + 1e-16)
    return torch.from_numpy(d)



class TsallisEntropy(nn.Module):

    def __init__(self, temperature: float, alpha: float):
        super(TsallisEntropy, self).__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        N, C = logits.shape

        pred = F.softmax(logits / self.temperature, dim=1)
        entropy_weight = entropy(pred).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (N * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)

        sum_dim = torch.sum(pred * entropy_weight, dim=0).unsqueeze(dim=0)

        return 1 / (self.alpha - 1) * torch.sum(
            (1 / torch.mean(sum_dim) - torch.sum(pred ** self.alpha / sum_dim * entropy_weight, dim=-1)))



def MI(outputs_target):
    batch_size = outputs_target.size(0)
    softmax_outs_t = nn.Softmax(dim=1)(outputs_target)
    avg_softmax_outs_t = torch.sum(softmax_outs_t, dim=0) / float(batch_size)
    log_avg_softmax_outs_t = torch.log(avg_softmax_outs_t)
    item1 = -torch.sum(avg_softmax_outs_t * log_avg_softmax_outs_t)
    item2 = -torch.sum(softmax_outs_t * torch.log(softmax_outs_t)) / float(batch_size)
    return -item1 + item2

