import os
import math

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.nn import functional as F

from pyiqa.utils.registry import ARCH_REGISTRY

def calculate_points(heatmaps):
    # change heatmaps to landmarks
    B, N, H, W = heatmaps.shape
    HW = H * W
    BN_range = np.arange(B * N)

    heatline = heatmaps.reshape(B, N, HW)
    indexes = np.argmax(heatline, axis=2)

    preds = np.stack((indexes % W, indexes // W), axis=2)
    preds = preds.astype(float, copy=False)

    inr = indexes.ravel()

    heatline = heatline.reshape(B * N, HW)
    x_up = heatline[BN_range, inr + 1]
    x_down = heatline[BN_range, inr - 1]
    # y_up = heatline[BN_range, inr + W]

    if any((inr + W) >= 4096):
        y_up = heatline[BN_range, 4095]
    else:
        y_up = heatline[BN_range, inr + W]
    if any((inr - W) <= 0):
        y_down = heatline[BN_range, 0]
    else:
        y_down = heatline[BN_range, inr - W]

    think_diff = np.sign(np.stack((x_up - x_down, y_up - y_down), axis=1))
    think_diff *= .25

    preds += think_diff.reshape(B, N, 2)
    preds += .5
    return preds


class AddCoordsTh(nn.Module):

    def __init__(self, x_dim=64, y_dim=64, with_r=False, with_boundary=False):
        super(AddCoordsTh, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.with_r = with_r
        self.with_boundary = with_boundary

    def forward(self, input_tensor, heatmap=None):
        """
        input_tensor: (batch, c, x_dim, y_dim)
        """
        batch_size_tensor = input_tensor.shape[0]

        xx_ones = torch.ones([1, self.y_dim], dtype=torch.int32).cuda()
        xx_ones = xx_ones.unsqueeze(-1)

        xx_range = torch.arange(self.x_dim, dtype=torch.int32).unsqueeze(0).cuda()
        xx_range = xx_range.unsqueeze(1)

        xx_channel = torch.matmul(xx_ones.float(), xx_range.float())
        xx_channel = xx_channel.unsqueeze(-1)

        yy_ones = torch.ones([1, self.x_dim], dtype=torch.int32).cuda()
        yy_ones = yy_ones.unsqueeze(1)

        yy_range = torch.arange(self.y_dim, dtype=torch.int32).unsqueeze(0).cuda()
        yy_range = yy_range.unsqueeze(-1)

        yy_channel = torch.matmul(yy_range.float(), yy_ones.float())
        yy_channel = yy_channel.unsqueeze(-1)

        xx_channel = xx_channel.permute(0, 3, 2, 1)
        yy_channel = yy_channel.permute(0, 3, 2, 1)

        xx_channel = xx_channel / (self.x_dim - 1)
        yy_channel = yy_channel / (self.y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_tensor, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_tensor, 1, 1, 1)

        if self.with_boundary and heatmap is not None:
            boundary_channel = torch.clamp(heatmap[:, -1:, :, :], 0.0, 1.0)

            zero_tensor = torch.zeros_like(xx_channel)
            xx_boundary_channel = torch.where(boundary_channel > 0.05, xx_channel, zero_tensor)
            yy_boundary_channel = torch.where(boundary_channel > 0.05, yy_channel, zero_tensor)
        if self.with_boundary and heatmap is not None:
            xx_boundary_channel = xx_boundary_channel.cuda()
            yy_boundary_channel = yy_boundary_channel.cuda()
        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))
            rr = rr / torch.max(rr)
            ret = torch.cat([ret, rr], dim=1)

        if self.with_boundary and heatmap is not None:
            ret = torch.cat([ret, xx_boundary_channel, yy_boundary_channel], dim=1)
        return ret
    

class CoordConvTh(nn.Module):
    """CoordConv layer as in the paper."""

    def __init__(self, x_dim, y_dim, with_r, with_boundary, in_channels, first_one=False, *args, **kwargs):
        super(CoordConvTh, self).__init__()
        self.addcoords = AddCoordsTh(x_dim=x_dim, y_dim=y_dim, with_r=with_r, with_boundary=with_boundary)
        in_channels += 2
        if with_r:
            in_channels += 1
        if with_boundary and not first_one:
            in_channels += 2
        self.conv = nn.Conv2d(in_channels=in_channels, *args, **kwargs)

    def forward(self, input_tensor, heatmap=None):
        ret = self.addcoords(input_tensor, heatmap)
        last_channel = ret[:, -2:, :, :]
        ret = self.conv(ret)
        return ret, last_channel
    

def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False, dilation=1):
    '3x3 convolution with padding'
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=strd, padding=padding, bias=bias, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ConvBlock(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4), padding=1, dilation=1)
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4), padding=1, dilation=1)

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


class HourGlass(nn.Module):

    def __init__(self, num_modules, depth, num_features, first_one=False):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.coordconv = CoordConvTh(
            x_dim=64,
            y_dim=64,
            with_r=True,
            with_boundary=True,
            in_channels=256,
            first_one=first_one,
            out_channels=256,
            kernel_size=1,
            stride=1,
            padding=0)
        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(256, 256))

        self.add_module('b2_' + str(level), ConvBlock(256, 256))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(256, 256))

        self.add_module('b3_' + str(level), ConvBlock(256, 256))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        up2 = F.interpolate(low3, scale_factor=2, mode='nearest')

        return up1 + up2

    def forward(self, x, heatmap):
        x, last_channel = self.coordconv(x, heatmap)
        return self._forward(self.depth, x), last_channel


class FAN(nn.Module):

    def __init__(self, num_modules=4, end_relu=False, gray_scale=False, num_landmarks=98):
        super(FAN, self).__init__()
        self.num_modules = num_modules
        self.gray_scale = gray_scale
        self.end_relu = end_relu
        self.num_landmarks = num_landmarks

        # Base part
        if self.gray_scale:
            self.conv1 = CoordConvTh(
                x_dim=256,
                y_dim=256,
                with_r=True,
                with_boundary=False,
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3)
        else:
            self.conv1 = CoordConvTh(
                x_dim=256,
                y_dim=256,
                with_r=True,
                with_boundary=False,
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        # Stacking part
        for hg_module in range(self.num_modules):
            if hg_module == 0:
                first_one = True
            else:
                first_one = False
            self.add_module('m' + str(hg_module), HourGlass(1, 4, 256, first_one))
            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256))
            self.add_module('conv_last' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
            self.add_module('l' + str(hg_module), nn.Conv2d(256, num_landmarks + 1, kernel_size=1, stride=1, padding=0))

            if hg_module < self.num_modules - 1:
                self.add_module('bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module),
                                nn.Conv2d(num_landmarks + 1, 256, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x, _ = self.conv1(x)
        x = F.relu(self.bn1(x), True)
        # x = F.relu(self.bn1(self.conv1(x)), True)
        x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        previous = x

        outputs = []
        boundary_channels = []
        tmp_out = None
        for i in range(self.num_modules):
            hg, boundary_channel = self._modules['m' + str(i)](previous, tmp_out)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)](self._modules['conv_last' + str(i)](ll)), True)

            # Predict heatmaps
            tmp_out = self._modules['l' + str(i)](ll)
            if self.end_relu:
                tmp_out = F.relu(tmp_out)  # HACK: Added relu
            outputs.append(tmp_out)
            boundary_channels.append(boundary_channel)

            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

        return outputs, boundary_channels

    @torch.no_grad()
    def get_landmarks(self, img, device='cuda'):
        _, _, H, W = img.shape
        offset = W / 64, H / 64, 0, 0

        # img = cv2.resize(img, (256, 256))
        # inp = img[..., ::-1]
        # inp = torch.from_numpy(np.ascontiguousarray(inp.transpose((2, 0, 1)))).float()
        inp = img.to(device)
        inp.div_(255.0)
        inp = F.interpolate(inp, (256, 256), mode='bicubic')

        outputs, _ = self.forward(inp)
        out = outputs[-1][:, :-1, :, :]
        heatmaps = out.detach().cpu().numpy()

        pred = calculate_points(heatmaps).reshape(-1, 2)

        pred *= offset[:2]
        pred += offset[-2:]

        return pred

    @torch.no_grad()
    def get_heatmaps(self, img, device='cuda'):
        H, W, _ = img.shape

        img = cv2.resize(img, (256, 256))
        inp = img[..., ::-1]
        inp = torch.from_numpy(np.ascontiguousarray(inp.transpose((2, 0, 1)))).float()
        inp = inp.to(device)
        inp.div_(255.0).unsqueeze_(0)

        outputs, _ = self.forward(inp)
        out = outputs[-1][:, :-1, :, :]
        heatmaps = out.squeeze(0).detach().cpu()
        return heatmaps

    @torch.no_grad()
    def get_heatmaps_quant(self, img, device='cuda'):
        H, W, _ = img.shape
        offset = W / 64, H / 64, 0, 0

        img = cv2.resize(img, (256, 256))
        inp = img[..., ::-1]
        inp = torch.from_numpy(np.ascontiguousarray(inp.transpose((2, 0, 1)))).float()
        inp = inp.to(device)
        inp.div_(255.0).unsqueeze_(0)

        outputs, _ = self.forward(inp)
        out = outputs[-1][:, :-1, :, :]
        heatmaps = out.detach().cpu().numpy()
        pred = calculate_points(heatmaps).reshape(-1, 2)

        pred *= offset[:2]
        pred += offset[-2:]

        heatmaps_quant = self._putGaussianMaps(pred, 512, 512, 8, 5.0)
        heatmaps_quant = torch.from_numpy(heatmaps_quant).float()
        return heatmaps_quant

    def _putGaussianMap(self, center, visible_flag, crop_size_y, crop_size_x, stride, sigma):
        """
        根据一个中心点,生成一个heatmap
        :param center:
        :return:
        """
        grid_y = crop_size_y // stride
        grid_x = crop_size_x // stride
        if visible_flag is False:
            return np.zeros((grid_y, grid_x))
        start = stride / 2.0 - 0.5
        y_range = [i for i in range(grid_y)]
        x_range = [i for i in range(grid_x)]
        xx, yy = np.meshgrid(x_range, y_range)
        xx = xx * stride + start
        yy = yy * stride + start
        d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
        exponent = d2 / 2.0 / sigma / sigma
        heatmap = np.exp(-exponent)
        return heatmap

    def _putGaussianMaps(self, keypoints, crop_size_y, crop_size_x, stride, sigma):
        """
        :param keypoints: (15,2)
        :param crop_size_y: int
        :param crop_size_x: int
        :param stride: int
        :param sigma: float
        :return:
        """
        all_keypoints = keypoints
        point_num = all_keypoints.shape[0]
        heatmaps_this_img = []
        for k in range(point_num):
            flag = ~np.isnan(all_keypoints[k, 0])
            heatmap = self._putGaussianMap(all_keypoints[k], flag, crop_size_y, crop_size_x, stride, sigma)
            heatmap = heatmap[np.newaxis, ...]
            heatmaps_this_img.append(heatmap)
        heatmaps_this_img = np.concatenate(
            heatmaps_this_img, axis=0)  # (num_joint,crop_size_y/stride,crop_size_x/stride)
        return heatmaps_this_img


def get_landmark_distance(gt_landmark, pred_landmark):
    return np.sqrt(((gt_landmark - pred_landmark) ** 2).sum(1)).mean()


def init_alignment_model(model_name='FAN', device='cuda'):
    if model_name == 'FAN':
        model = FAN()
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')
    model_path = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'pretrained_models/alignment_WFLW_4HG.pth')
    model.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'], strict=True)
    model.eval()
    return model


@ARCH_REGISTRY.register()
class LandmarkDistance(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

        self.landmark_detector = init_alignment_model()

    @torch.no_grad()
    def forward(self, pred, gt):
        pred_landmark = self.landmark_detector.get_landmarks(pred)
        gt_landmark = self.landmark_detector.get_landmarks(gt)

        dist = get_landmark_distance(gt_landmark, pred_landmark)

        return dist