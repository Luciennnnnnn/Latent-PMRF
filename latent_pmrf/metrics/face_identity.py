import math
import os

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms.functional import rgb_to_grayscale

from .arcface.config.config import Config
from .arcface.models.resnet import resnet_face18

from pyiqa.utils.registry import ARCH_REGISTRY


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


@ARCH_REGISTRY.register()
class FaceIdentity(nn.Module):
    def __init__(
        self,
        model_path: str = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'pretrained_models/arcface/resnet18_110.pth'),
        **kwargs,
    ):
        super().__init__()

        self.model = resnet_face18(Config().use_se)
        state_dict = torch.load(model_path)
        for key in list(state_dict.keys()):
            if key.startswith('module.'):
                state_dict[key[7:]] = state_dict.pop(key)

        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def forward(self, pred, gt):
        pred = rgb_to_grayscale(pred)
        gt = rgb_to_grayscale(gt)
        pred = F.interpolate(pred, (128, 128), mode='bicubic')
        gt = F.interpolate(gt, (128, 128), mode='bicubic')
        pred_out = self.model(pred)
        gt_out = self.model(gt)

        dist = cosin_metric(pred_out.cpu().numpy().squeeze(), gt_out.cpu().numpy().squeeze())
        dist = np.arccos(dist) / math.pi * 180
        return dist