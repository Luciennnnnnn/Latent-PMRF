from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat

_LAYER_NORM_IMLP = 'official'  # defaulting to False for now


def set_layer_norm_imlp(imlp: str):
    global _LAYER_NORM_IMLP
    _LAYER_NORM_IMLP = imlp


def layer_norm_imlp():
    return _LAYER_NORM_IMLP


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        with torch.autocast(device_type='cuda', enabled=False):
            output = self._norm(x.float()).type_as(x)
        output = output * self.weight
        return output
    

# Copy from https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/archs/arch_util.py
# class LayerNormFunction(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, x, weight, bias, eps):
#         ctx.eps = eps
#         N, C, H, W = x.size()
#         mu = x.mean(1, keepdim=True)
#         var = (x - mu).pow(2).mean(1, keepdim=True)
#         y = (x - mu) / (var + eps).sqrt()
#         ctx.save_for_backward(y, var, weight)

#         if weight is not None:
#             y = weight * y + bias
#         return y

#     @staticmethod
#     @torch.autograd.function.once_differentiable
#     def backward(ctx, grad_output):
#         eps = ctx.eps

#         N, C, H, W = grad_output.size()
#         y, var, weight = ctx.saved_tensors

#         g = grad_output
#         if weight is not None:
#             g = g * weight

#         mean_g = g.mean(dim=1, keepdim=True)

#         mean_gy = (g * y).mean(dim=1, keepdim=True)
#         gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)

#         if weight is not None:
#             return gx, (grad_output * y).sum(dim=(0, 2, 3), keepdim=True), grad_output.sum(dim=(0, 2, 3), keepdim=True), None
#         else:
#             return gx, None, None, None

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        N, C, H, W = x.size()
        diff = x - x.mean(1, keepdim=True)
        rstd = torch.rsqrt(diff.pow(2).mean(1, keepdim=True) + eps)
        y = diff * rstd
        ctx.save_for_backward(y, rstd, weight)

        if weight is not None and bias is not None:
            y = torch.addcmul(bias, y, weight)
        return y

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        N, C, H, W = grad_output.size()
        y, rstd, weight = ctx.saved_tensors

        g = grad_output
        if weight is not None:
            g = g * weight

        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)

        gx = rstd * (g - y * mean_gy - mean_g)

        if weight is not None:
            return gx, (grad_output * y).sum(dim=(0, 2, 3), keepdim=True), grad_output.sum(dim=(0, 2, 3), keepdim=True), None
        else:
            return gx, None, None, None

class LayerNormFunctionSupportJVP(torch.autograd.Function):

    @staticmethod
    def forward(x, weight, bias, eps):
        # Perform the main computation and return intermediate results for context
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x_hat = (x - mu) / (var + eps).sqrt()

        if weight is not None:
            y = weight * x_hat + bias
            return y
        
        # Return the intermediate results needed for setup_context
        return x_hat

    @staticmethod
    @torch.autograd.function.once_differentiable
    def setup_context(ctx, inputs, outputs):
        # Extract intermediate values from inputs and outputs
        x, weight, bias, eps = inputs

        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)

        if weight is not None:
            x_hat = (x - mu) / (var + eps).sqrt()
        else:
            x_hat = outputs

        ctx.eps = eps
        ctx.x_hat = x_hat
        ctx.var = var
        ctx.weight = weight

        # Save necessary values for backward
        ctx.save_for_backward(x_hat, var, weight)

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        x_hat, var, weight = ctx.saved_tensors
        g = grad_output

        if weight is not None:
            g = g * weight

        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * x_hat).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - x_hat * mean_gy - mean_g)

        if weight is not None:
            grad_weight = (grad_output * x_hat).sum(dim=(0, 2, 3), keepdim=True)
            grad_bias = grad_output.sum(dim=(0, 2, 3), keepdim=True)
            return gx, grad_weight, grad_bias, None
        else:
            return gx, None, None, None
        
    @staticmethod
    def jvp(ctx, v_x, v_weight, v_bias, v_eps):
        # 从上下文获取保存的值
        eps = ctx.eps
        x_hat, var, weight = ctx.x_hat, ctx.var, ctx.weight
        del ctx.x_hat, ctx.var, ctx.weight

        B, C, H, W = v_x.shape

        if weight is not None:
            # v_x_weight = v_x * weight
            Jv = (
                weight * v_x
                - weight * v_x.mean(dim=1, keepdim=True)
                - weight * x_hat * (x_hat * v_x).mean(dim=1, keepdim=True)
            ) / torch.sqrt(var + eps)
        else:
            Jv = (
                v_x
                - repeat(v_x.mean(dim=1, keepdim=True), "b 1 h w -> b c h w", c=C)
                - x_hat * (x_hat * v_x).mean(dim=1, keepdim=True)
            ) / torch.sqrt(var + eps)

        if weight is not None:
            Jv = Jv + x_hat * v_weight + v_bias
        return Jv


class LayerNormFunctionSupportDoubleBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        mu = x.mean(1, keepdim=True)
        diff = x - mu
        rstd = torch.rsqrt(diff.pow(2).mean(1, keepdim=True) + eps)
        x_bar = diff * rstd
        ctx.save_for_backward(x, mu, rstd, weight)
        
        y = x_bar
        if weight is not None and bias is not None:
            # y = weight * x_bar + bias
            y = bias.addcmul(weight, x_bar)
        return y, mu, rstd

    @staticmethod
    def backward(ctx, grad_output, grad_mu, grad_rstd):
        x, mu, rstd, weight = ctx.saved_tensors

        diff = x - mu
        x_bar = diff * rstd

        g = grad_output
        if weight is not None:
            g = g * weight

        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * x_bar).mean(dim=1, keepdim=True)
        
        # gx = rstd * (g - x_bar * mean_gy - mean_g)
        gx = rstd * (g.addcmul(x_bar, mean_gy, value=-1) - mean_g)

        # gx += (grad_mu - grad_rstd * diff * rstd ** 3) / x_bar.size(1)
        gx += (grad_mu.addcmul(grad_rstd, diff * rstd ** 3, value=-1)) / x_bar.size(1)

        if weight is not None:
            g_weight = (grad_output * x_bar).sum(dim=(0, 2, 3), keepdim=True)
            g_bias = grad_output.sum(dim=(0, 2, 3), keepdim=True)

            return gx, g_weight, g_bias, None
        else:
            return gx, None, None, None
        

# @torch.compiler.disable(recursive=True)
# class LayerNorm2d(nn.Module):

#     def __init__(self, channels, eps=1e-6, elementwise_affine=True):
#         super().__init__()
        
#         if elementwise_affine:
#             self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
#             self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
#         else:
#             self.weight = None
#             self.bias = None
  
#         self.eps = eps
#         self.imlp = 'default'

#     def set_layer_norm_imlp(self, imlp: str):
#         self.imlp = imlp

#     def forward(self, x):
#         if self.imlp == 'default':
#             return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
#         elif self.imlp == 'jvp':
#             return LayerNormFunctionSupportJVP.apply(x, self.weight, self.bias, self.eps)
#         elif self.imlp == 'double_backward':
#             return LayerNormFunctionSupportDoubleBackward.apply(x, self.weight, self.bias, self.eps)[0]
#         else:
#             raise ValueError(f"Invalid imlp: {self.imlp}")


class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """

    def __init__(self, channels, eps=1e-6, elementwise_affine=True):
        super().__init__(channels, eps=eps, elementwise_affine=elementwise_affine)
        self.imlp = layer_norm_imlp()

        if self.imlp != 'official' and elementwise_affine:
            self.weight.data = self.weight.data.view(1, -1, 1, 1)
            self.bias.data = self.bias.data.view(1, -1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.imlp == 'official':
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = x.permute(0, 3, 1, 2)
        elif self.imlp == 'default':
            return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
        elif self.imlp == 'jvp':
            return LayerNormFunctionSupportJVP.apply(x, self.weight, self.bias, self.eps)
        elif self.imlp == 'double_backward':
            return LayerNormFunctionSupportDoubleBackward.apply(x, self.weight, self.bias, self.eps)[0]
        else:
            raise ValueError(f"Invalid imlp: {self.imlp}")
        return x


class LayerNorm2dTorchCompileCompatible(nn.Module):
    def __init__(self, channels, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        else:
            self.weight = None
            self.bias = None

    def forward(self, x):
        # 使用原生实现
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / (var + self.eps).sqrt()
        
        if self.weight is not None:
            x = x * self.weight + self.bias
        return x

class RMSNorm2d(nn.RMSNorm):
    """ RMSNorm for channels of '2D' spatial NCHW tensors """

    def __init__(self, channels, eps:float = None, elementwise_affine=True):
        super().__init__(channels, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x
    
class AdaLayerNorm2d(nn.Module):
    r"""
    Norm layer adaptive layer norm (adaLN).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(
        self,
        dim,
        embedding_dim: Optional[int] = None,
        patch_size: int = 8,
        adaptation_type: str = "channel_wise",
        bias: bool = True
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.adaptation_type = adaptation_type
        self.patch_size = patch_size

        if embedding_dim is not None and adaptation_type == 'channel_wise':
            self.linear = zero_init(nn.Linear(embedding_dim, 2 * dim, bias=bias))
        elif embedding_dim is not None and adaptation_type == 'frequency_wise':
            # self.linear = zero_init(nn.Linear(embedding_dim, 2 * patch_size * patch_size))
            self.linear = zero_init(nn.Linear(embedding_dim, 2 * patch_size * (patch_size // 2 + 1), bias=bias))
        self.norm = LayerNorm2d(dim, elementwise_affine=embedding_dim is None, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if emb is not None:
            emb = self.linear(emb)
            shift_msa, scale_msa = emb.chunk(2, dim=1)
            x = self.norm(x)
            if self.adaptation_type == 'channel_wise':
                x = x * (1 + scale_msa[..., None, None]) + shift_msa[..., None, None]
            elif self.adaptation_type == 'frequency_wise':
                B, C, H, W = x.shape
                shift_msa = repeat(shift_msa, 'b (p1 p2) -> (b h w) 1 p1 p2', h=H//self.patch_size, w=W//self.patch_size, p1=self.patch_size, p2=(self.patch_size // 2 + 1))
                scale_msa = repeat(scale_msa, 'b (p1 p2) -> (b h w) 1 p1 p2', h=H//self.patch_size, w=W//self.patch_size, p1=self.patch_size, p2=(self.patch_size // 2 + 1))

                x = rearrange(x, 'b c (h p1) (w p2) -> (b h w) c p1 p2', p1=self.patch_size, p2=self.patch_size)
                spec_x = torch.fft.rfft2(x)
                spec_x = spec_x * (1 + scale_msa) + shift_msa
                x = torch.fft.irfft2(spec_x)
                x = rearrange(x, '(b h w) c p1 p2 -> b c (h p1) (w p2)', b=B, h=H//self.patch_size, w=W//self.patch_size, p1=self.patch_size, p2=self.patch_size)
        else:
            x = self.norm(x)
        return x



class AdaRMSNorm2d(nn.Module):
    r"""
    Norm layer adaptive rms norm (adaRN).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(
        self,
        dim,
        embedding_dim: Optional[int] = None,
        patch_size: int = 8,
        bias: bool = True
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size

        if embedding_dim is not None:
            self.linear = zero_init(nn.Linear(embedding_dim, dim, bias=bias))
        self.norm = RMSNorm2d(dim, elementwise_affine=embedding_dim is None, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if emb is not None:
            scale_msa = self.linear(emb)
            x = self.norm(x)
            x = x * (1 + scale_msa[..., None, None])
        else:
            x = self.norm(x)
        return x
    
def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer