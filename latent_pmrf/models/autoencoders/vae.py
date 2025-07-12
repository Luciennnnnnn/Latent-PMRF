from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from diffusers.utils import BaseOutput, is_torch_version
from .blocks import get_down_block, get_up_block, get_mid_block


@dataclass
class DecoderOutput(BaseOutput):
    r"""
    Output of decoding method.

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The decoded output sample from the last layer of the model.
    """

    sample: torch.Tensor
    commit_loss: Optional[torch.FloatTensor] = None


class Encoder(nn.Module):
    r"""
    The `Encoder` layer of a variational autoencoder that encodes its input into a latent representation.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        down_block_types (`Tuple[str, ...]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            The types of down blocks to use. See `~diffusers.models.unet_2d_blocks.get_down_block` for available
            options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        double_z (`bool`, *optional*, defaults to `True`):
            Whether to double the number of output channels for the last block.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("EncoderBlock2D",),
        mid_block_type: str = "EncoderBlock2D",
        block_out_channels: Tuple[int, ...] = (64,),
        mid_block_out_channel: int = 64,
        layers_per_block: int = 2,
        layers_mid_block: int = 3,
        double_z: bool = True,
        norm_pos: str = 'pre_norm',
        norm_type: str = "layer_norm",
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.layers_mid_block = layers_mid_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.down_blocks = nn.ModuleList([])

        # down
        for i, down_block_type in enumerate(down_block_types):
            is_final_block = i == len(down_block_types) - 1
            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=block_out_channels[i],
                out_channels=block_out_channels[i + 1] if not is_final_block else mid_block_out_channel,
                norm_pos=norm_pos,
                norm_type=norm_type,
                add_downsample=not is_final_block,
            )
            self.down_blocks.append(down_block)
        
        self.mid_block = get_mid_block(
            mid_block_type,
            num_layers=self.layers_mid_block,
            in_channels=mid_block_out_channel,
            out_channels=mid_block_out_channel,
            norm_pos=norm_pos,
            norm_type=norm_type,
        )

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        r"""The forward method of the `Encoder` class."""

        sample = self.conv_in(sample)

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # down
            if is_torch_version(">=", "1.11.0"):
                for down_block in self.down_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(down_block), sample, use_reentrant=False
                    )
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, use_reentrant=False
                )
            else:
                for down_block in self.down_blocks:
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(down_block), sample)
                # middle
                sample = torch.utils.checkpoint.checkpoint(create_custom_forward(self.mid_block), sample)

        else:
            # down
            for down_block in self.down_blocks:
                sample = down_block(sample)
            # middle
            sample = self.mid_block(sample)

        # post-process
        sample = self.conv_out(sample)

        return sample


class Decoder(nn.Module):
    r"""
    The `Decoder` layer of a variational autoencoder that decodes its latent representation into an output sample.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        up_block_types (`Tuple[str, ...]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            The types of up blocks to use. See `~diffusers.models.unet_2d_blocks.get_up_block` for available options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        norm_type (`str`, *optional*, defaults to `"group"`):
            The normalization type to use. Can be either `"group"` or `"spatial"`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        mid_block_type: str = "EncoderBlock2D",
        mid_block_out_channel: int = 64,
        layers_per_block: int = 2,
        layers_mid_block: int = 3,
        norm_pos: str = 'pre_norm',
        norm_type: str = "layer_norm",
        upsample_method: str = 'interpolate',
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.layers_mid_block = layers_mid_block
        self.conv_in = nn.Conv2d(
            in_channels,
            mid_block_out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid_block = get_mid_block(
            mid_block_type,
            num_layers=self.layers_mid_block,
            in_channels=mid_block_out_channel,
            out_channels=mid_block_out_channel,
            norm_pos=norm_pos,
            norm_type=norm_type,
        )

        self.up_blocks = nn.ModuleList([])
        
        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=self.layers_per_block + 1,
                in_channels=reversed_block_out_channels[i],
                out_channels=reversed_block_out_channels[i + 1] if i + 1 < len(reversed_block_out_channels) else reversed_block_out_channels[i],
                norm_pos=norm_pos,
                norm_type=norm_type,
                upsample_method=upsample_method,
                add_upsample=not is_final_block,
            )
            self.up_blocks.append(up_block)

        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

        self.gradient_checkpointing = False


    def forward(
        self,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        r"""The forward method of the `Decoder` class."""

        sample = self.conv_in(sample)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, use_reentrant=False
                )
                sample = sample.to(upscale_dtype)

                # up
                for i, up_block in enumerate(self.up_blocks):
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        sample,
                        use_reentrant=False,
                    )
            else:
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample
                )
                sample = sample.to(upscale_dtype)

                # up
                for i, up_block in enumerate(self.up_blocks):
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(up_block), sample)
        else:
            # middle
            sample = self.mid_block(sample)
            sample = sample.to(upscale_dtype)
            # up
            for i, up_block in enumerate(self.up_blocks):
                sample = up_block(sample)

        # post-process
        sample = self.conv_out(sample)

        return sample