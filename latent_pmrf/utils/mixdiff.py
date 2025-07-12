def _tile2pixel_indices(tile_row, tile_col, tile_width, tile_height, tile_row_overlap, tile_col_overlap):
    """Given a tile row and column numbers returns the range of pixels affected by that tiles in the overall image
    
    Returns a tuple with:
        - Starting coordinates of rows in pixel space
        - Ending coordinates of rows in pixel space
        - Starting coordinates of columns in pixel space
        - Ending coordinates of columns in pixel space
    """
    px_row_init = 0 if tile_row == 0 else tile_row * (tile_height - tile_row_overlap)
    px_row_end = px_row_init + tile_height
    px_col_init = 0 if tile_col == 0 else tile_col * (tile_width - tile_col_overlap)
    px_col_end = px_col_init + tile_width
    return px_row_init, px_row_end, px_col_init, px_col_end


def _pixel2latent_indices(px_row_init, px_row_end, px_col_init, px_col_end, vae_scale_factor=8):
    """Translates coordinates in pixel space to coordinates in latent space"""
    return px_row_init // vae_scale_factor, px_row_end // vae_scale_factor, px_col_init // vae_scale_factor, px_col_end // vae_scale_factor


def _tile2latent_indices(tile_row, tile_col, tile_width, tile_height, tile_row_overlap, tile_col_overlap, vae_scale_factor=8):
    """Given a tile row and column numbers returns the range of latents affected by that tiles in the overall image
    
    Returns a tuple with:
        - Starting coordinates of rows in latent space
        - Ending coordinates of rows in latent space
        - Starting coordinates of columns in latent space
        - Ending coordinates of columns in latent space
    """
    px_row_init, px_row_end, px_col_init, px_col_end = _tile2pixel_indices(tile_row, tile_col, tile_width, tile_height, tile_row_overlap, tile_col_overlap)
    return _pixel2latent_indices(px_row_init, px_row_end, px_col_init, px_col_end, vae_scale_factor=vae_scale_factor)