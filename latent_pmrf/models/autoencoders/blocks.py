import torch
import torch.nn as nn
import torch.nn.functional as F


def round_to_multiple_of_64(x):
    return max(((int(x) + 31) // 64) * 64, 64)


def get_down_block(
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    norm_pos: str = 'pre_norm',
    norm_type: str = "layer_norm",
    add_downsample: bool = True,
):

    if down_block_type == "EncoderBlock2D":
        return EncoderBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            norm_pos=norm_pos,
            norm_type=norm_type,
            add_downsample=add_downsample,
        )

def get_up_block(
    up_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    norm_pos: str = 'pre_norm',
    norm_type: str = "layer_norm",
    upsample_method: str = 'interpolate',
    add_upsample: bool = True,
):
    if up_block_type == "DecoderBlock2D":
        return DecoderBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            norm_pos=norm_pos,
            norm_type=norm_type,
            upsample_method=upsample_method,
            add_upsample=add_upsample,
        )

def get_mid_block(
    up_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    norm_pos: str = 'pre_norm',
    norm_type: str = "layer_norm",
):
    if up_block_type == "Block2D":
        return DecoderBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            norm_pos=norm_pos,
            norm_type=norm_type,
            add_upsample=False,
        )

class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        norm_pos: str = "pre_norm",
        norm_type: str = "layer_norm",
    ):
        super().__init__()
        from ..norm import LayerNorm2d as LayerNorm2d

        self.norm_pos = norm_pos

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.non_linearity = nn.SiLU(inplace=True)

        if norm_type == "layer_norm":
            self.norm = LayerNorm2d(channels, elementwise_affine=True)
        elif norm_type == "group_norm":
            self.norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
        else:
            raise ValueError(f"norm_type {norm_type} is not supported")
        
    def forward(self, hidden_states: torch.Tensor):
        input_tensor = hidden_states
        if self.norm_pos == "pre_norm":
            hidden_states = self.norm(hidden_states)

        hidden_states = self.conv1(hidden_states)
        hidden_states = self.non_linearity(hidden_states)
        hidden_states = self.conv2(hidden_states)

        output = hidden_states + input_tensor

        if self.norm_pos == "post_norm":
            output = self.norm(output)
        return output


class FeedForwardBlock(nn.Module):
    def __init__(self, dim, mult):
        super().__init__()
        
        inner_dim = round_to_multiple_of_64(dim * mult)

        self.project_in = nn.Conv2d(dim, inner_dim * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(
            inner_dim * 2,
            inner_dim * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=inner_dim * 2,
        )
        self.down_proj = nn.Conv2d(inner_dim, dim, kernel_size=1)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.down_proj(x)
        return x
    

class Downsample2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.conv(hidden_states)
        return hidden_states
    
class Upsample2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upsample_method: str = 'interpolate'):
        super().__init__()
        self.upsample_method = upsample_method
        if self.upsample_method == 'interpolate':
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, stride=1, padding=1)

    def forward(self, hidden_states: torch.Tensor):
        if self.upsample_method == 'interpolate':
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")

        hidden_states = self.conv(hidden_states)

        if self.upsample_method == 'pixel_shuffle':
            hidden_states = F.pixel_shuffle(hidden_states, upscale_factor=2)
        return hidden_states
    
class EncoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        norm_pos: str = 'pre_norm',
        norm_type: str = "layer_norm",
        add_downsample: bool = True,
    ):
        super().__init__()
        self.add_downsample = add_downsample

        resnets = []

        for i in range(num_layers):
            resnets.append(
                ResnetBlock2D(
                    channels=in_channels,
                    norm_pos=norm_pos,
                    norm_type=norm_type,

                )
            )

        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsampler = Downsample2D(
                in_channels=in_channels,
                out_channels=out_channels,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        if self.add_downsample:
            hidden_states = self.downsampler(hidden_states)
        return hidden_states


class DecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        norm_pos: str = 'pre_norm',
        norm_type: str = "layer_norm",
        upsample_method: str = 'interpolate',
        add_upsample: bool = True,
    ):
        super().__init__()
        self.add_upsample = add_upsample
        resnets = []

        for i in range(num_layers):
            resnets.append(
                ResnetBlock2D(
                    channels=in_channels,
                    norm_pos=norm_pos,
                    norm_type=norm_type,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        if add_upsample:
            self.upsampler = Upsample2D(
                in_channels=in_channels,
                out_channels=out_channels,
                upsample_method=upsample_method,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        if self.add_upsample:
            hidden_states = self.upsampler(hidden_states)
        return hidden_states