"""k-diffusion transformer diffusion models, version 2.
Codes adopted from https://github.com/crowsonkb/k-diffusion
"""
from typing import Tuple

from functools import reduce
import math
from typing import Union, Optional

from einops import rearrange
import torch
from torch import nn
import torch._dynamo
from torch.nn import functional as F

from .axial_rope import make_axial_pos

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.embeddings import GaussianFourierProjection

from ..norm import AdaLayerNorm2d, LayerNorm2d, RMSNorm2d, AdaRMSNorm2d

try:
    import natten
except ImportError:
    natten = None

def round_to_multiple_of_64(x):
    return ((int(x) + 31) // 64) * 64
    
# Helpers

def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def downscale_pos(pos):
    pos = rearrange(pos, "... (h nh) (w nw) e -> ... h w (nh nw) e", nh=2, nw=2)
    return torch.mean(pos, dim=-2)

# Kernels
def linear_geglu(x, weight, bias=None):
    x = x @ weight.mT
    if bias is not None:
        x = x + bias
    x, gate = x.chunk(2, dim=-1)
    return x * F.gelu(gate)


def rms_norm(x, scale, eps):
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype)**2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)


def rms_norm2d(x, scale, eps):
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype)**2, dim=1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)

class LinearGEGLU(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features * 2, bias=bias)
        self.out_features = out_features

    def forward(self, x):
        return linear_geglu(x, self.weight, self.bias)
    

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)

class RMSNorm(nn.Module):
    def __init__(self, shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(shape))

    def extra_repr(self):
        return f"shape={tuple(self.scale.shape)}, eps={self.eps}"

    def forward(self, x):
        return rms_norm(x, self.scale, self.eps)

def apply_rotary_emb(x, theta, conj=False):
    out_dtype = x.dtype
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2, x3 = x[..., :d], x[..., d : d * 2], x[..., d * 2 :]
    x1, x2, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    y1, y2 = y1.to(out_dtype), y2.to(out_dtype)
    return torch.cat((y1, y2, x3), dim=-1)

def _apply_rotary_emb_inplace(x, theta, conj):
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2 = x[..., :d], x[..., d : d * 2]
    x1_, x2_, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1_ * cos - x2_ * sin
    y2 = x2_ * cos + x1_ * sin
    x1.copy_(y1)
    x2.copy_(y2)

class ApplyRotaryEmbeddingInplace(torch.autograd.Function):
    @staticmethod
    def forward(x, theta, conj):
        _apply_rotary_emb_inplace(x, theta, conj=conj)
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, theta, conj = inputs
        ctx.save_for_backward(theta)
        ctx.conj = conj

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        _apply_rotary_emb_inplace(grad_output, theta, conj=not ctx.conj)
        return grad_output, None, None


def apply_rotary_emb_(x, theta):
    return ApplyRotaryEmbeddingInplace.apply(x, theta, False)


class AxialRoPE(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        log_min = math.log(math.pi)
        log_max = math.log(10.0 * math.pi)
        freqs = torch.linspace(log_min, log_max, n_heads * dim // 4 + 1)[:-1].exp()
        self.register_buffer("freqs", freqs.view(dim // 4, n_heads).T.contiguous())

    def extra_repr(self):
        return f"dim={self.freqs.shape[1] * 4}, n_heads={self.freqs.shape[0]}"

    def forward(self, pos):
        theta_h = pos[..., None, 0:1] * self.freqs.to(pos.dtype)
        theta_w = pos[..., None, 1:2] * self.freqs.to(pos.dtype)
        return torch.cat((theta_h, theta_w), dim=-1)

class ConvProjectLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dwconv_type: str = 'dwconv', bias: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=dwconv_type == 'none' and bias)
        if dwconv_type == 'dwconv':
            self.dwconv = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim, bias=bias)
        else:
            self.dwconv = None

    def forward(self, x):
        x = self.conv(x)
        if self.dwconv is not None:
            x = self.dwconv(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim, head_dim, dwconv_type: str = 'dwconv', bias: bool = True, zero_init_out_proj: bool = False):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = dim // head_dim

        self.qkv_proj = ConvProjectLayer(dim, dim * 3, dwconv_type, bias=bias)

        self.pos_emb = AxialRoPE(head_dim // 2, self.num_heads)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=bias)

        if zero_init_out_proj:
            self.out_proj = zero_init(self.out_proj)

    def extra_repr(self):
        return f"head_dim={self.head_dim}, num_heads={self.num_heads}"

    def forward(self, hidden_states, pos):
        b, c, h, w = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        q, k, v = rearrange(qkv, "n (t nh e) h w -> t n nh (h w) e", t=3, e=self.head_dim) # (b, nh, h * w, head_dim)
        
        pos = rearrange(pos, "... h w e -> ... (h w) e").to(qkv.dtype) # (b, h * w, 2)
        theta = self.pos_emb(pos).movedim(-2, -3) # (b, 1, h * w, head_dim // 4)
        q = apply_rotary_emb(q, theta)
        k = apply_rotary_emb(k, theta)

        hidden_states = F.scaled_dot_product_attention(q, k, v)
        hidden_states = rearrange(hidden_states, "n nh (h w) e -> n (nh e) h w", h=h, w=w)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class NeighborhoodAttention(nn.Module):
    def __init__(self, dim, head_dim, kernel_size, dwconv_type: str = 'dwconv', bias: bool = True, zero_init_out_proj: bool = False):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.kernel_size = kernel_size

        self.qkv_proj = ConvProjectLayer(dim, dim * 3, dwconv_type, bias=bias)

        self.pos_emb = AxialRoPE(head_dim // 2, self.num_heads)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=bias)

        self.scale = head_dim ** -0.5

        if zero_init_out_proj:
            self.out_proj = zero_init(self.out_proj)

    def extra_repr(self):
        return f"head_dim={self.head_dim}, num_heads={self.num_heads}"

    def forward(self, hidden_states: torch.Tensor, pos):
        qkv = self.qkv_proj(hidden_states)
        theta = self.pos_emb(pos) # (h, w, 1, head_dim // 4)

        if natten is None:
            raise ModuleNotFoundError("natten is required for neighborhood attention")
        
        if natten.has_fused_na():
            q, k, v = rearrange(qkv, "n (t nh e) h w -> t n h w nh e", t=3, e=self.head_dim) # (n, h, w, nh, head_dim)
            q = apply_rotary_emb(q, theta)
            k = apply_rotary_emb(k, theta)
            hidden_states = natten.functional.na2d(q, k, v, self.kernel_size, scale=self.scale)
            hidden_states = rearrange(hidden_states, "n h w nh e -> n (nh e) h w")
        else:
            q, k, v = rearrange(qkv, "n (t nh e) h w -> t n nh h w e", t=3, e=self.head_dim) # (n, nh, h, w, head_dim)
            theta = theta.movedim(-2, -4) # (1, h, w, head_dim // 4)
            q = apply_rotary_emb(q, theta)
            k = apply_rotary_emb(k, theta)
            a = natten.functional.na2d_qk(q, k, self.kernel_size)
            a = (a * self.scale).softmax(dim=-1).to(v.dtype)
            hidden_states = natten.functional.na2d_av(a, v, self.kernel_size)
            hidden_states = rearrange(hidden_states, "n nh h w e -> n (nh e) h w")
            
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult,
        dwconv_type: str = "dwconv",
        bias: bool = True,
        zero_init_out_proj: bool = False,
    ):
        super().__init__()
        
        inner_dim = round_to_multiple_of_64(dim * mult)
        
        self.project_in = ConvProjectLayer(dim, inner_dim * 2, dwconv_type, bias=bias)

        self.down_proj = nn.Conv2d(inner_dim, dim, kernel_size=1, bias=bias)
        if zero_init_out_proj:
            self.down_proj = zero_init(self.down_proj)

    def forward(self, x):
        x1, x2 = self.project_in(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.down_proj(x)
        return x
    

class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        head_dim,
        kernel_size: Optional[int] = None,
        mult: int = 3,
        attn_type: str = "global",
        cond_dim: Optional[int] = None,
        dwconv_type: str = "dwconv",
        bias: bool = True,
        zero_init_out_proj: bool = False,
        block_idx: int = 0,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        if attn_type == "global":
            self.attn = SelfAttention(dim, head_dim, dwconv_type, bias=bias, zero_init_out_proj=zero_init_out_proj)
        elif attn_type == "neighborhood":
            self.attn = NeighborhoodAttention(dim, head_dim, kernel_size, dwconv_type, bias=bias, zero_init_out_proj=zero_init_out_proj)
        else:
            raise ValueError(f"unsupported self attention spec {attn_type}")
        
        self.ff = FeedForward(dim, mult, dwconv_type, bias=bias, zero_init_out_proj=zero_init_out_proj)
        self.norm1 = AdaRMSNorm2d(dim, cond_dim, bias=bias)
        self.norm2 = AdaRMSNorm2d(dim, cond_dim, bias=bias)

    def forward(self, hidden_states, pos, cond = None):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states, cond), pos)
        hidden_states = hidden_states + self.ff(self.norm2(hidden_states, cond))
        return hidden_states


class Level(nn.ModuleList):
    def forward(self, x, *args, **kwargs):
        for layer in self:
            x = layer(x, *args, **kwargs)
        return x

# Mapping network

class MappingFeedForwardBlock(nn.Module):
    def __init__(self, dim, mult, dropout=0.0):
        super().__init__()
        inner_dim = round_to_multiple_of_64(dim * mult)

        self.norm = RMSNorm(dim)
        self.up_proj = LinearGEGLU(dim, inner_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.down_proj = zero_init(nn.Linear(inner_dim, dim, bias=False))

    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip


class MappingNetwork(nn.Module):
    def __init__(self, n_layers, dim, mult, dropout=0.0):
        super().__init__()
        self.in_norm = RMSNorm(dim)
        self.blocks = nn.ModuleList([MappingFeedForwardBlock(dim, mult, dropout=dropout) for _ in range(n_layers)])
        self.out_norm = RMSNorm(dim)

    def forward(self, x):
        x = self.in_norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return x

# Token merging and splitting
class TokenMerge(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2), bias: bool = True):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = nn.Conv2d(in_features * self.h * self.w, out_features, 1, 1, 0, bias=bias)

    def forward(self, x):
        x = rearrange(x, "... e (h nh) (w nw) -> ... (nh nw e) h w", nh=self.h, nw=self.w)
        return self.proj(x)

class TokenSplitWithoutSkip(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2), bias: bool = True):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = nn.Conv2d(in_features, out_features * self.h * self.w, 1, 1, 0, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, "... (nh nw e) h w-> ... e (h nh) (w nw)", nh=self.h, nw=self.w)
    
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias: bool = True):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class Downsample2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        downsample_method: str = "conv",
        bias: bool = True,
    ):
        super().__init__()
        self.downsample_method = downsample_method
        if downsample_method == "conv":
            assert kernel_size == 3
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=bias
            )
        elif downsample_method == "pixel_unshuffle+conv":
            self.pixel_unshuffle = nn.PixelUnshuffle(2)
            self.conv = nn.Conv2d(
                in_channels * 4,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                bias=bias,
            )
        elif downsample_method == "conv+pixel_unshuffle":
            self.conv = nn.Conv2d(
                in_channels,
                out_channels // 4,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                bias=bias,
            )
            self.pixel_unshuffle = nn.PixelUnshuffle(2)

    def forward(self, hidden_states: torch.Tensor):
        if self.downsample_method == "conv":
            hidden_states = self.conv(hidden_states)
        elif self.downsample_method == "pixel_unshuffle+conv":
            hidden_states = self.pixel_unshuffle(hidden_states)
            hidden_states = self.conv(hidden_states)
        elif self.downsample_method == "conv+pixel_unshuffle":
            hidden_states = self.conv(hidden_states)
            hidden_states = self.pixel_unshuffle(hidden_states)
        return hidden_states
    

class Upsample2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        upsample_method: str = "interpolate",
        bias: bool = True,
    ):
        super().__init__()
        self.upsample_method = upsample_method
        if self.upsample_method == "interpolate":
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias)
        self.fac = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, hidden_states: torch.Tensor, skip: torch.Tensor):
        if self.upsample_method == 'interpolate':
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")

        hidden_states = self.conv(hidden_states)

        if self.upsample_method == 'pixel_shuffle':
            hidden_states = F.pixel_shuffle(hidden_states, upscale_factor=2)
        return torch.lerp(skip.to(hidden_states.dtype), hidden_states, self.fac.to(hidden_states.dtype))
    

class LooUFormer(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: Tuple[int],
        block_dims: Tuple[int],
        block_layers: Tuple[int],
        block_types: Tuple[str],
        window_sizes: Tuple[int],
        head_dims: Optional[Tuple[int]] = None,
        dwconv_type: str = 'dwconv',
        upsample_method: str = 'interpolate',
        downsample_method: str = 'conv',
        upsample_kernel_size: int = 3,
        downsample_kernel_size: int = 3,
        patch_embed_type: str = 'conv',
        use_bias: bool = True,
        zero_init_last_layer: bool = False,
        time_embed_dim: Optional[int] = None,
        time_embed_log_space: bool = False,
        time_embed_freq_scale: float = 1.0,
        mult: float = 3.0,
        mapping_layers: int = 2,
        mapping_mult: float = 3.0,
        mapping_dropout: float = 0.0,
        scale: int = 1,
    ):
        super().__init__()

        self.time_embed_dim = time_embed_dim
        if time_embed_dim is not None:
            self.time_proj = GaussianFourierProjection(
                time_embed_dim // 2,
                scale=time_embed_freq_scale,
                set_W_to_weight=False,
                log=time_embed_log_space,
                flip_sin_to_cos=True,
            )
            self.timestep_embedder = nn.Linear(time_embed_dim, time_embed_dim, bias=False)
            self.mapping = MappingNetwork(mapping_layers, time_embed_dim, mapping_mult, dropout=mapping_dropout)

        if patch_embed_type == 'conv':
            self.proj_in = OverlapPatchEmbed(in_channels, block_dims[0], bias=use_bias)
        elif patch_embed_type == 'pixel_shuffle':
            self.proj_in = TokenMerge(in_channels, block_dims[0], patch_size, bias=use_bias)
        else:
            raise ValueError(f"unsupported patch embed type {patch_embed_type}")

        self.down_blocks, self.up_blocks = nn.ModuleList(), nn.ModuleList()
        block_idx = 0
        for i, block_type in enumerate(block_types[:-1]):
            down_block = []
            for j in range(block_layers[i]):
                down_block.append(
                    TransformerBlock(
                        dim=block_dims[i],
                        head_dim=head_dims[i],
                        kernel_size=window_sizes[i],
                        mult=mult,
                        attn_type=block_type,
                        cond_dim=time_embed_dim,
                        dwconv_type=dwconv_type,
                        bias=use_bias,
                        zero_init_out_proj=zero_init_last_layer,
                        block_idx=block_idx + j,
                    )
                )
            self.down_blocks.append(Level(down_block))
            block_idx += block_layers[i]

        for i, block_type in enumerate(block_types[:-1]):
            up_block = []
            for j in range(block_layers[i]):
                up_block.append(
                    TransformerBlock(
                        dim=block_dims[i],
                        head_dim=head_dims[i],
                        kernel_size=window_sizes[i],
                        mult=mult,
                        attn_type=block_type,
                        cond_dim=time_embed_dim,
                        dwconv_type=dwconv_type,
                        bias=use_bias,
                        zero_init_out_proj=zero_init_last_layer,
                        block_idx=block_idx + j,
                    )
                )
            self.up_blocks.append(Level(up_block))
            block_idx += block_layers[i]

        self.mid_block = Level(
            [
                TransformerBlock(
                    dim=block_dims[-1],
                    head_dim=head_dims[-1],
                    kernel_size=window_sizes[-1],
                    mult=mult,
                    attn_type=block_types[-1],
                    cond_dim=time_embed_dim,
                    dwconv_type=dwconv_type,
                    bias=use_bias,
                    zero_init_out_proj=zero_init_last_layer,
                    block_idx=block_idx + j,
                )
                for j in range(block_layers[-1])
            ]
        )

        self.downs = nn.ModuleList([Downsample2D(block_dims[i], block_dims[i+1], downsample_method=downsample_method, bias=use_bias, kernel_size=downsample_kernel_size) for i in range(len(block_dims) - 1)])
        self.ups = nn.ModuleList([Upsample2D(block_dims[i+1], block_dims[i], upsample_method=upsample_method, bias=use_bias, kernel_size=upsample_kernel_size) for i in range(len(block_dims) - 1)])
        
        self.scale = scale
        self.out_norm = RMSNorm2d(block_dims[0])

        if patch_embed_type == 'conv':
            self.proj_out = OverlapPatchEmbed(block_dims[0], out_channels, bias=use_bias)
        elif patch_embed_type == 'pixel_shuffle':
            self.proj_out = TokenSplitWithoutSkip(block_dims[0], out_channels, patch_size)

        if zero_init_last_layer:
            zero_init(self.proj_out.proj)

    def forward(self, x, sigma = None, cond: Optional[torch.Tensor] = None):
        time_cond = None
        if self.time_embed_dim is not None:
            timesteps_proj = self.time_proj(sigma)
            timesteps_emb = self.timestep_embedder(timesteps_proj)

            time_cond = self.mapping(timesteps_emb)

        x = self.proj_in(x)

        pos = make_axial_pos(x.shape[-2], x.shape[-1], device=x.device).view(x.shape[-2], x.shape[-1], 2) # (h, w, 2)
        
        skips, poses = [], []
        for down_block, downsample in zip(self.down_blocks, self.downs):
            x = down_block(x, pos, time_cond)
            skips.append(x)
            poses.append(pos)
            x = downsample(x)
            pos = downscale_pos(pos)

        x = self.mid_block(x, pos, time_cond)

        for up_block, upsample, skip, pos in reversed(list(zip(self.up_blocks, self.ups, skips, poses))):
            x = upsample(x, skip)
            x = up_block(x, pos, time_cond)

        # Unpatching
        x = self.out_norm(x)
        x = self.proj_out(x)
        return x
