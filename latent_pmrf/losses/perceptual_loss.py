from copy import deepcopy

import torch
from torch.nn import functional as F

from ..models.extractor import DecoderExtractor
from .vgg_arch import VGGFeatureExtractor


class DecoderPerceptualLoss(torch.nn.Module):
    def __init__(self,
                 model,
                 return_nodes={DecoderExtractor.BLOCK_KEY: [0]},
                 layer_weights={DecoderExtractor.BLOCK_KEY: [1]},
                 criterion='l1',):
        super().__init__()
        self.layer_weights = layer_weights

        self.extractor = DecoderExtractor(deepcopy(model), return_nodes=return_nodes)

        print(self.extractor.model)
        # self.resize = partial(F.interpolate, size=224, mode=interpolate_mode, align_corners=False, antialias=antialias)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def measure_distance(self, x, y):
        if self.criterion_type == 'fro':
            distance = torch.norm(x - y, p='fro')
        else:
            distance = self.criterion(x, y)
        return distance

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # calculate perceptual loss
        percep_loss = 0

        x_features = {}

        self.extractor(x)
        for feature_type in self.layer_weights.keys():
            if feature_type == 'block':
                features = self.extractor.get_block_for_all()
            elif feature_type == 'layer':
                features = self.extractor.get_layer_for_all()
            else:
                raise NotImplementedError(f"Do not support feature_type {feature_type}")

            x_features[feature_type] = features

        self.extractor._init_hooks_data()

        self.extractor(gt.detach())
        for feature_type in self.layer_weights.keys():
            if feature_type == 'block':
                features = self.extractor.get_block_for_all()
            elif feature_type == 'layer':
                features = self.extractor.get_layer_for_all()
            else:
                raise NotImplementedError(f"Do not support feature_type {feature_type}")
            for x_feature, gt_feature, weight in zip(x_features[feature_type], features, self.layer_weights[feature_type]):
                percep_loss += weight * self.measure_distance(x_feature, gt_feature)

        self.extractor._init_hooks_data()

        return percep_loss


class PerceptualLoss(torch.nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1',
                 softmax_reweighting=False,
                 requires_grad=True):
        super().__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.softmax_reweighting = softmax_reweighting
        
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm,
            requires_grad=requires_grad)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss(reduction='none')
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss(reduction='none')
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            if self.softmax_reweighting:
                _percep_loss = []
                for k in x_features.keys():
                    if self.criterion_type == 'fro':
                        _percep_loss.append(torch.norm(x_features[k] - gt_features[k], p='fro'))
                    else:
                        _percep_loss.append(self.criterion(x_features[k], gt_features[k]).flatten(start_dim=1).mean(dim=1))
                _percep_loss = torch.stack(_percep_loss, dim=-1)
                weights = F.softmax(_percep_loss, dim=-1)

                percep_loss = torch.dot(weights.detach(), _percep_loss)
            else:
                for k in x_features.keys():
                    if self.criterion_type == 'fro':
                        percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                    else:
                        percep_loss += self.criterion(x_features[k], gt_features[k]).flatten(start_dim=1).mean(dim=1) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])).flatten(start_dim=1).mean(dim=1) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def get_features(self, x):
        x_features = self.vgg(x)
        features = []

        for k in x_features.keys():
            features.append(x_features[k].flatten(start_dim=1))
        
        return tuple(features)

    def get_features_flat(self, x):
        features = self.get_features(x)
        return torch.cat(features, dim=1)
    
    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram