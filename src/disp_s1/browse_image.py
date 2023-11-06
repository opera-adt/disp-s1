"""Module for creating browse images for the output product."""

from __future__ import annotations

import numpy as np
from dolphin._types import Filename
from numpy.typing import ArrayLike
from PIL import Image


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
    orig_shape = arr.shape
    scaling_ratio = max([s / max_dim_allowed for s in orig_shape])
    # scale original shape by scaling ratio
    scaled_shape = [int(np.ceil(s / scaling_ratio)) for s in orig_shape]

    # TODO: Make actual browse image
    dummy = np.zeros(scaled_shape, dtype="uint8")
    img = Image.fromarray(dummy, mode="L")
    img.save(output_filename, transparency=0)
