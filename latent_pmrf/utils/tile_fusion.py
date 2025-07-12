import numpy as np
from numpy import pi, exp, sqrt

import torch


def _gaussian_weights(tile_height, tile_width, nbatches, channels, device):
    """Generates a gaussian mask of weights for tile contributions"""

    var = 0.01
    midpoint = (tile_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
    x_probs = [
        exp(-(x - midpoint) * (x - midpoint) / (tile_width * tile_width) / (2 * var))
        / sqrt(2 * pi * var)
        for x in range(tile_width)
    ]
    midpoint = tile_height / 2
    y_probs = [
        exp(-(y - midpoint) * (y - midpoint) / (tile_height * tile_height) / (2 * var))
        / sqrt(2 * pi * var)
        for y in range(tile_height)
    ]

    weights = np.outer(y_probs, x_probs)
    return torch.tile(torch.tensor(weights, device=device, dtype=torch.float32), (nbatches, channels, 1, 1))


def _uniform_weights(tile_height, tile_width, nbatches, channels, device):
    """Generates a uniform mask of weights for tile contributions"""

    return torch.ones((nbatches, channels, tile_height, tile_width), device=device)
    

def get_tile_weights(tile_height, tile_width, nbatches, channels, device, version='gaussian'):
    if version == "gaussian":
        return _gaussian_weights(
            tile_height=tile_height,
            tile_width=tile_width,
            nbatches=nbatches,
            channels=channels,
            device=device,
        )
    elif version == "uniform":
        return _uniform_weights(
            tile_height=tile_height,
            tile_width=tile_width,
            nbatches=nbatches,
            channels=channels,
            device=device,
        )
    else:
        raise NotImplementedError("Only support: [gaussian, uniform]")
        

def tile2pixel_indices(tile_row, tile_col, tile_height, tile_width, tile_row_overlap, tile_col_overlap, height, width):
    """Given a tile row and column numbers returns the range of pixels affected by that tiles in the overall image
    
    Returns a tuple with:
        - Starting coordinates of rows in pixel space
        - Ending coordinates of rows in pixel space
        - Starting coordinates of columns in pixel space
        - Ending coordinates of columns in pixel space
    """
    px_row_init = tile_row * (tile_height - tile_row_overlap)
    px_row_end = min(px_row_init + tile_height, height)
    px_row_init = px_row_end - tile_height

    px_col_init = tile_col * (tile_width - tile_col_overlap)
    px_col_end = min(px_col_init + tile_width, width)
    px_col_init = px_col_end - tile_width

    assert px_row_init >= 0 and px_col_init >=0, f"We support input image with height >= {tile_height} and width >= {tile_width} only."

    return px_row_init, px_row_end, px_col_init, px_col_end


def _pixel2latent_indices(px_row_init, px_row_end, px_col_init, px_col_end, scale_factor=8):
    """Translates coordinates in pixel space to coordinates in latent space"""
    return px_row_init // scale_factor, px_row_end // scale_factor, px_col_init // scale_factor, px_col_end // scale_factor


def tile2latent_indices(tile_row, tile_col, tile_height, tile_width, tile_row_overlap, tile_col_overlap, height, width, scale_factor=8):
    """Given a tile row and column numbers returns the range of latents affected by that tiles in the overall image
    
    Returns a tuple with:
        - Starting coordinates of rows in latent space
        - Ending coordinates of rows in latent space
        - Starting coordinates of columns in latent space
        - Ending coordinates of columns in latent space
    """
    px_row_init, px_row_end, px_col_init, px_col_end = tile2pixel_indices(tile_row, tile_col, tile_height, tile_width, tile_row_overlap, tile_col_overlap, height, width)
    return _pixel2latent_indices(px_row_init, px_row_end, px_col_init, px_col_end, scale_factor=scale_factor)