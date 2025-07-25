import torch
from torch import nn as nn
from torch.nn import functional as F


from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)



class StochasticVariation(nn.Module):
    def __init__(self, ch) -> None:
        super().__init__()
        self.channel_scale = torch.nn.Parameter(torch.ones([1, ch, 1, 1]))

    def forward(self, x):
        if self.training:
            x = x + torch.randn_like(x, device=x.device, dtype=x.dtype) * self.channel_scale
        return x


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32, apply_stochastic_variation=False):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

        self.apply_stochastic_variation = apply_stochastic_variation
        if apply_stochastic_variation:
            self.stochastic_variation = StochasticVariation(num_feat)


    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Emperically, we use 0.2 to scale the residual for better performance
        if self.apply_stochastic_variation:
            return self.stochastic_variation(x5 * 0.2 + x)
        else:
            return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32, apply_stochastic_variation=False):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch, apply_stochastic_variation)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch, apply_stochastic_variation)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch, apply_stochastic_variation)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Emperically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


import numpy as np
from scipy import signal

def gaussian_kernel_2d(kernel_size=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.windows.gaussian(kernel_size, std=std).reshape(kernel_size, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


class MULIER(nn.Module):
    def __init__(self, num_feat, mode='nearest', layers=2, kernel_size=5, sigma=1.0) -> None:
        super().__init__()
        self.mode = mode
        self.layers = layers

        if self.layers > 0:
            W_s = []
            G = gaussian_kernel_2d(kernel_size, sigma)
            W_l = np.ones_like(G)

            for i in range(layers + 1):
                W_s.append(W_l)
                W_l *= G

            W_s = [W_s[i] - W_s[i + 1] for i in range(layers)]
            W_s = [torch.tensor(W, dtype=torch.float32).view(1, 1, kernel_size, kernel_size).expand(num_feat, num_feat, -1, -1) for W in W_s]

            self.W_s = W_s
            self.a = nn.Parameter(torch.ones(layers))
            self.b = nn.Parameter(torch.zeros(layers))
            self.pad = nn.ReflectionPad2d(kernel_size // 2)

    def forward(self, x):
        z = F.interpolate(x, scale_factor=2, mode=self.mode)
        if self.layers > 0:
            x = self.pad(x)
            for a, b, W in zip(self.a, self.b, self.W_s):
                z += torch.tanh(a * F.interpolate(F.conv2d(x, W.to(x.device)), scale_factor=2, mode=self.mode) + b)
        return z


class RRDBNet(ModelMixin, ConfigMixin):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """
    @register_to_config
    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32,
                 apply_stochastic_variation=False, mulier_layers=0):
        super(RRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch, apply_stochastic_variation=apply_stochastic_variation)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.upsampler1 = MULIER(num_feat, layers=mulier_layers)
        self.upsampler2 = MULIER(num_feat, layers=mulier_layers)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(self.upsampler1(feat)))
        feat = self.lrelu(self.conv_up2(self.upsampler2(feat)))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class RRDBNetS(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32, last_kernel_size=7):
        super(RRDBNetS, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if scale in [8, 16]:
            assert last_kernel_size % 2 == 1, "last_kernel_size must be odd number."
            self.conv_up3 = nn.Conv2d(num_feat, num_feat, last_kernel_size, 1, (last_kernel_size - 1) // 2)

        if scale == 16:
            assert last_kernel_size % 2 == 1, "last_kernel_size must be odd number."
            self.conv_up4 = nn.Conv2d(num_feat, num_feat, last_kernel_size, 1, (last_kernel_size - 1) // 2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))

        if self.scale in [8, 16]:
            feat = self.lrelu(self.conv_up3(F.interpolate(feat, scale_factor=2, mode='nearest')))

        if self.scale == 16:
            feat = self.lrelu(self.conv_up4(F.interpolate(feat, scale_factor=2, mode='nearest')))

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out