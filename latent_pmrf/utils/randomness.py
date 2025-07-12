import torch

def randint(low, high=None, size=None):
    if high is None:
        high = low
        low = 0
    if size is None:
        size = low.shape if isinstance(low, torch.Tensor) else high.shape
    return torch.randint(2**63 - 1, size=size, device=low.device) % (high - low) + low