"""Module for creating browse images for the output product."""
from __future__ import annotations

import numpy as np
from dolphin._types import Filename
from numpy.typing import ArrayLike
from PIL import Image


def _scale_to_max_pixel_dimension(orig_shape, max_dim_allowed=2048):
    '''
    Scale up or down length and width represented by a shape to a maximum
    dimension. The larger of length or width used to compute scaling ratio.

    Parameters
    ----------
    orig_shape: tuple[int]
        Shape (length, width) to be scaled
    max_dim_allowed: int
        Maximum dimension allowed for either length or width

    Returns
    -------
    _: list(int)
        Shape (length, width) scaled up or down from original shape
    '''
    # compute scaling ratio based on larger dimension
    scaling_ratio = max([xy / max_dim_allowed for xy in orig_shape])

    # scale original shape by scaling ratio
    scaled_shape = [int(np.ceil(xy / scaling_ratio)) for xy in orig_shape]
    return scaled_shape


def make_browse_image(
    output_filename: Filename,
    arr: ArrayLike,
    max_dim_allowed: int = 2048,
) -> None:
    """Create a PNG browse image for the output product.

    Parameters
    ----------
    output_filename : Filename
        Name of output PNG
    arr : ArrayLike
        input 2D image array
    max_dim_allowed : int, default = 2048
        Size (in pixels) of the maximum allowed dimension of output image.
        Image gets rescaled with same aspect ratio.
    """
    # compute browse shape
    full_shape = arr.shape
    browse_h, browse_w = _scale_to_max_pixel_dimension(full_shape,
                                                       max_dim_allowed)

    orig_shape = arr.shape
    scaling_ratio = max([s / max_dim_allowed for s in orig_shape])
    # scale original shape by scaling ratio
    scaled_shape = [int(np.ceil(s / scaling_ratio)) for s in orig_shape]

    # TODO: Make actual browse image
    dummy = np.zeros(scaled_shape, dtype="uint8")
    img = Image.fromarray(dummy, mode="L")
    img.save(output_filename, transparency=0)


def make_unwrapped_phase_browse_image(
        output_filename: Filename):
    make
