from torch import nn


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module