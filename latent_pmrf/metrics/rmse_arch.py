import torch.nn as nn

from pyiqa.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class RMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        return ((pred - gt) ** 2).flatten(start_dim=1).mean(dim=-1).sqrt()
    

@ARCH_REGISTRY.register()
class MSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        return ((pred - gt) ** 2).flatten(start_dim=1).mean(dim=-1)