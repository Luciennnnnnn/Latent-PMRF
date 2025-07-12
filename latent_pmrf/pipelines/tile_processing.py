import numpy as np

import torch


def gaussian_weights(tile_height, tile_width,
                     in_latent_space=True, vae_scale_factor=8,
                     device=None):
    """Generate a gaussian mask of weights for tile contributions."""

    if in_latent_space:
        latent_width = tile_width // vae_scale_factor
        latent_height = tile_height // vae_scale_factor
    else:
        latent_width = tile_width
        latent_height = tile_height

    var = 0.01
    midpoint_x = (latent_width - 1) / 2
    midpoint_y = (latent_height - 1) / 2
    
    x = torch.arange(latent_width, dtype=torch.float32, device=device)
    y = torch.arange(latent_height, dtype=torch.float32, device=device)
    
    x_probs = torch.exp(-((x - midpoint_x) ** 2) / (latent_width**2) / (2 * var)) / np.sqrt(2 * np.pi * var)
    y_probs = torch.exp(-((y - midpoint_y) ** 2) / (latent_height**2) / (2 * var)) / np.sqrt(2 * np.pi * var)
    
    weights = torch.outer(y_probs, x_probs)
    return weights
    # return torch.tile(
    #     torch.tensor(weights, device=self.device), (nbatches, channels, 1, 1)
    # )


def uniform_weights(tile_height, tile_width, in_latent_space=True, vae_scale_factor=8, device=None):
    """Generate a uniform mask of weights for tile contributions."""
    if in_latent_space:
        latent_width = tile_width // vae_scale_factor
        latent_height = tile_height // vae_scale_factor
    else:
        latent_width = tile_width
        latent_height = tile_height

    return torch.ones((latent_height, latent_width), device=device)


def get_tile_weights(
    tile_height,
    tile_width,
    in_latent_space=True,
    vae_scale_factor=8,
    version="gaussian",
    device=None,
):
    if version == "gaussian":
        return gaussian_weights(
            tile_height, tile_width, in_latent_space, vae_scale_factor, device
        )
    elif version == "uniform":
        return uniform_weights(
            tile_height, tile_width, in_latent_space, vae_scale_factor, device
        )
    else:
        raise NotImplementedError("Only support: [gaussian, uniform]")
    

def get_tile_indices(
    row,
    col,
    tile_height,
    tile_width,
    tile_row_overlap,
    tile_col_overlap,
    height,
    width,
    latent_space=False,
    scale_factor=8,
):
    """Given a tile row and column numbers returns the range of pixels affected by that tiles in the overall image

    Returns a tuple with:
        - Starting coordinates of rows in pixel space
        - Ending coordinates of rows in pixel space
        - Starting coordinates of columns in pixel space
        - Ending coordinates of columns in pixel space
    """
    px_row_init = row * (tile_height - tile_row_overlap)
    px_row_end = min(px_row_init + tile_height, height)
    px_row_init = px_row_end - tile_height

    px_col_init = col * (tile_width - tile_col_overlap)
    px_col_end = min(px_col_init + tile_width, width)
    px_col_init = px_col_end - tile_width

    assert (
        px_row_init >= 0 and px_col_init >= 0
    ), f"We support input image with height >= {tile_height} and width >= {tile_width} only."

    if latent_space:
        return (
            px_row_init // scale_factor,
            px_row_end // scale_factor,
            px_col_init // scale_factor,
            px_col_end // scale_factor,
        )
    else:
        return px_row_init, px_row_end, px_col_init, px_col_end