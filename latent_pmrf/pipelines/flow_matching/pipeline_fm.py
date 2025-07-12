# Copyright 2024 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import math

import numpy as np
import torch

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    is_torch_xla_available,
    logging,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from ..tile_processing import get_tile_weights, get_tile_indices
from dataclasses import dataclass

import PIL.Image

from diffusers.utils import BaseOutput

from einops import repeat


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
class FMPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    mmse_model_outs: Optional[Union[List[PIL.Image.Image], np.ndarray]] = None
    mmse_model_outs_decoded: Optional[Union[List[PIL.Image.Image], np.ndarray]] = None

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class FlowMatchingPipeline(DiffusionPipeline):

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        transformer: FluxTransformer2DModel,
        vae: Optional[AutoencoderKL] = None,
        posterior_mean_model = None
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            posterior_mean_model=posterior_mean_model
        )
        if vae is not None:
            self.vae_scale_factor = (
                2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
            )
        else:
            self.vae_scale_factor = 1
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.default_sample_size = 64

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline.prepare_latents
    def prepare_sources(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        control_image = None,
        source_distribution: str = 'noise',
        posterior_mean_model = None,
        posterior_mean_model_range_norm: bool = False,
        noise_std: float = 0,
        upsample_scale: float = 1,
        return_mmse_model_out: bool = False
    ):
        height = int(height) // self.vae_scale_factor
        width = int(width) // self.vae_scale_factor

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if source_distribution == 'noise':
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        elif "latent" in source_distribution:
            if source_distribution == 'posterior_mean_latent':
                if not posterior_mean_model_range_norm:
                    sr = posterior_mean_model(control_image.to(dtype=torch.float32) * 0.5 + 0.5)
                    sr = sr.clip(0, 1) * 2 - 1
                else:
                    sr = posterior_mean_model(control_image.to(dtype=torch.float32))
                    sr = sr.clip(-1, 1)
            elif source_distribution == 'lq_latent':
                from torch.nn import functional as F
                sr = F.interpolate(control_image.to(dtype=torch.float32), scale_factor=upsample_scale, mode='bicubic')

            sr_latent = self.vae.encode(sr.to(dtype=self.vae.dtype)).latent_dist.sample(generator=generator)
            sr_latent_decoded = self.vae.decode(sr_latent, return_dict=False)[0]
            if self.vae.config.shift_factor is not None:
                sr_latent = sr_latent - self.vae.config.shift_factor
            if self.vae.config.scaling_factor is not None:
                sr_latent = sr_latent * self.vae.config.scaling_factor
            noise = noise_std * randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = sr_latent.to(dtype=dtype) + noise
        else:
            if source_distribution == 'posterior_mean':
                sr = posterior_mean_model(control_image.to(dtype=torch.float32) * 0.5 + 0.5)
                sr = sr.clip(0, 1) * 2 - 1
            elif source_distribution == 'lq':
                from torch.nn import functional as F
                sr = F.interpolate(control_image.to(dtype=torch.float32), scale_factor=upsample_scale, mode='bicubic')

            noise = noise_std * randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = sr.to(dtype=dtype) + noise

        if return_mmse_model_out:
            if source_distribution == 'posterior_mean_latent':
                return latents, sr, sr_latent_decoded
            return latents, sr
        else:
            return latents

    # Copied from diffusers.pipelines.controlnet_sd3.pipeline_stable_diffusion_3_controlnet.StableDiffusion3ControlNetPipeline.prepare_image
    def prepare_image(
        self,
        image,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        if isinstance(image, torch.Tensor):
            pass
        else:
            image = self.image_processor.preprocess(image)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    @property
    def num_timesteps(self):
        return self._num_timesteps
    
    @torch.no_grad()
    def __call__(
        self,
        upsample_scale: int = 1,
        tile_height: Optional[int] = None,
        tile_width: Optional[int] = None,
        tile_row_overlap = 256,
        tile_col_overlap = 256,
        tile_weights_version: str = 'gaussian',
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        control_image: PipelineImageInput = None,
        source_distribution: str = 'noise',
        posterior_mean_model = None,
        posterior_mean_model_range_norm: bool = False,
        noise_std: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        return_mmse_model_out: bool = False,
        verbose: bool = False
    ):

        batch_size = 1

        # device = self._execution_device
        device = self.vae.device if self.vae is not None else self.transformer.device
        dtype = self.vae.dtype if self.vae is not None else self.transformer.dtype

        # 3. Prepare control image
        control_image = self.prepare_image(
            image=control_image,
            batch_size=batch_size * 1,
            num_images_per_prompt=1,
            device=device,
            dtype=dtype,
        )

        height, width = control_image.shape[-2:]

        height = height * upsample_scale
        width = width * upsample_scale

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels

        _latents = self.prepare_sources(
            batch_size * 1,
            num_channels_latents,
            height,
            width,
            control_image.dtype,
            device,
            generator,
            latents,
            control_image=control_image,
            source_distribution=source_distribution,
            posterior_mean_model=posterior_mean_model,
            posterior_mean_model_range_norm=posterior_mean_model_range_norm,
            noise_std=noise_std,
            upsample_scale=upsample_scale,
            return_mmse_model_out=return_mmse_model_out
        )
        if return_mmse_model_out:
            if source_distribution == 'posterior_mean_latent':
                _latents, sr, sr_latent_decoded = _latents
            else:
                _latents, sr = _latents

        image = self.tiled_processing(
            control_image=control_image,
            upsample_scale=upsample_scale,
            latents=_latents,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            tile_height=tile_height,
            tile_width=tile_width,
            tile_row_overlap=tile_row_overlap,
            tile_col_overlap=tile_col_overlap,
            tile_weights_version=tile_weights_version,
            device=device,
            dtype=dtype,
            verbose=verbose
        )

        image = self.image_processor.postprocess(image, output_type=output_type)
        if return_mmse_model_out:
            sr = self.image_processor.postprocess(sr, output_type=output_type)
            if source_distribution == 'posterior_mean_latent':
                sr_latent_decoded = self.image_processor.postprocess(sr_latent_decoded, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            if return_mmse_model_out:
                return (image, sr, sr_latent_decoded if source_distribution == 'posterior_mean_latent' else None)
            else:
                return image
        if return_mmse_model_out:
            return FMPipelineOutput(images=image, mmse_model_outs=sr, mmse_model_outs_decoded=sr_latent_decoded if source_distribution == 'posterior_mean_latent' else None)
        else:
            return FMPipelineOutput(images=image)

    def tiled_processing(
        self,
        control_image,
        upsample_scale,
        latents,
        num_inference_steps,
        timesteps,
        tile_height,
        tile_width,
        tile_row_overlap,
        tile_col_overlap,
        tile_weights_version,
        device,
        dtype,
        verbose
    ):
        self.set_progress_bar_config(disable=not verbose)

        batch_size = control_image.shape[0]
        target_height = control_image.shape[-2] * upsample_scale
        target_width = control_image.shape[-1] * upsample_scale

        # Determine tile size and overlap
        if target_height * target_width <= tile_height * tile_width:
            tile_height, tile_width = target_height, target_width
            tile_row_overlap = tile_col_overlap = 0
        elif target_height < tile_height:
            tile_height = target_height
            tile_row_overlap = 0
        elif target_width < tile_width:
            tile_width = target_width
            tile_col_overlap = 0

        assert tile_height != tile_row_overlap and tile_width != tile_col_overlap

        cond_tile_height = tile_height // upsample_scale
        cond_tile_width = tile_width // upsample_scale

        cond_tile_row_overlap = tile_row_overlap // upsample_scale
        cond_tile_col_overlap = tile_col_overlap // upsample_scale

        grid_rows = math.ceil((control_image.shape[-2] - cond_tile_height) / (cond_tile_height - cond_tile_row_overlap)) + 1
        grid_cols = math.ceil((control_image.shape[-1] - cond_tile_width) / (cond_tile_width - cond_tile_col_overlap)) + 1

        tile_weights = get_tile_weights(
            tile_height=tile_height,
            tile_width=tile_width,
            in_latent_space=True,
            vae_scale_factor=self.vae_scale_factor,
            version=tile_weights_version,
            device=device,
        )

        tile_weights = repeat(tile_weights, "h w -> b c h w", b=batch_size, c=latents.shape[1])

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # process each tile
                model_pred_tiles = [[] for _ in range(grid_rows)]
                for row in range(grid_rows):
                    for col in range(grid_cols):
                        cond_px_row_init, cond_px_row_end, cond_px_col_init, cond_px_col_end = get_tile_indices(
                            row, col, 
                            cond_tile_height, cond_tile_width,
                            cond_tile_row_overlap, cond_tile_col_overlap,
                            control_image.shape[-2], control_image.shape[-1]
                        )
                        
                        px_row_init, px_row_end, px_col_init, px_col_end = get_tile_indices(
                            row, col,
                            tile_height, tile_width, 
                            tile_row_overlap, tile_col_overlap,
                            target_height, target_width,
                            latent_space=True,
                            scale_factor=self.vae_scale_factor
                        )
                        
                        condition_tile = control_image[:, :, cond_px_row_init:cond_px_row_end, cond_px_col_init:cond_px_col_end]
                        latents_tile = latents[:, :, px_row_init:px_row_end, px_col_init:px_col_end]

                        model_pred_tile = self.predict(
                            t=t,
                            latents=latents_tile,
                            control_image=condition_tile,
                        )

                        model_pred_tiles[row].append(model_pred_tile)
                # Stitch noise predictions for all tiles
                model_pred = torch.zeros((batch_size, latents.shape[1], latents.shape[2], latents.shape[3]), device=self.device, dtype=torch.float32)
                contributors = torch.zeros((batch_size, latents.shape[1], latents.shape[2], latents.shape[3]), device=self.device, dtype=torch.float32)
                # Add each tile contribution to overall latents
                for row in range(grid_rows):
                    for col in range(grid_cols):
                        px_row_init, px_row_end, px_col_init, px_col_end = get_tile_indices(row, col,
                                                                                            tile_height, tile_width,
                                                                                            tile_row_overlap, tile_col_overlap,
                                                                                            target_height, target_width,
                                                                                            latent_space=True,
                                                                                            scale_factor=self.vae_scale_factor)
                        model_pred[:, :, px_row_init:px_row_end, px_col_init:px_col_end] += model_pred_tiles[row][col] * tile_weights
                        contributors[:, :, px_row_init:px_row_end, px_col_init:px_col_end] += tile_weights
                # Average overlapping areas with more than 1 contributor
                model_pred /= contributors
                
                latents = self.scheduler.step(model_pred, t, latents, return_dict=False)[0]

                latents = latents.to(dtype=dtype)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        latents = latents.to(dtype=dtype)
        if self.vae is not None:
            if self.vae.config.scaling_factor is not None:
                latents = latents / self.vae.config.scaling_factor
            if self.vae.config.shift_factor is not None:
                latents = latents + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
        else:
            image = latents
        
        return image

    def predict(
        self,
        t,
        latents,
        control_image,
    ):
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        batch_size, num_channels_latents, height, width = latents.shape

        noise_pred = self.transformer(
            latents,
            sigma=timestep / 1000,
            cond=control_image,
        )[0]

        return noise_pred