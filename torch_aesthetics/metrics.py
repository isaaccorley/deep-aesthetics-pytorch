import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def spearmanr(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    
    vx = x1 - torch.mean(x1)
    vy = x2 - torch.mean(x2)

    num = torch.sum(vx * vy)
    den = torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))

    return num / den

@torch.no_grad()
def mean_ap(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    pass