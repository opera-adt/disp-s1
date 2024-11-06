"""Module for creating browse images for the output product."""

from __future__ import annotations

import cmap
import h5netcdf
import matplotlib.pyplot as plt
import numpy as np
from dolphin._types import Filename
from numpy.typing import ArrayLike
from scipy import ndimage

from .product_info import DISPLACEMENT_PRODUCTS

DEFAULT_CMAP = cmap.Colormap("vik").to_mpl()


def _resize_to_max_pixel_dim(arr: ArrayLike, max_dim_allowed=2048) -> np.ndarray:
    """Scale shape of a given array."""
    if max_dim_allowed < 1:
        raise ValueError(f"{max_dim_allowed} is not a valid max image dimension")
    input_shape = arr.shape
    scaling_ratio = max([max_dim_allowed / xy for xy in input_shape])
    nan_mask = np.isnan(arr)
    arr[nan_mask] = 0
    arr = ndimage.zoom(arr, scaling_ratio)
    arr[ndimage.zoom(nan_mask, scaling_ratio, order=0)] = np.nan
    return arr


def _save_to_disk_as_color(
    arr: ArrayLike, fname: Filename, cmap: str, vmin: float, vmax: float
) -> None:
    """Save image array as color to file."""
    plt.imsave(fname, arr, cmap=cmap, vmin=vmin, vmax=vmax)


def make_browse_image_from_arr(
    output_filename: Filename,
    arr: ArrayLike,
    mask: ArrayLike,
    max_dim_allowed: int = 2048,
    cmap: str = DEFAULT_CMAP,
    vmin: float = -0.10,
    vmax: float = 0.10,
) -> None:
    """Create a PNG browse image for the output product from given array."""
    arr[mask == 0] = np.nan
    arr = _resize_to_max_pixel_dim(arr, max_dim_allowed)
    _save_to_disk_as_color(arr, output_filename, cmap, vmin, vmax)


def make_browse_image_from_nc(
    output_filename: Filename,
    input_filename: Filename,
    dataset_name: str,
    max_dim_allowed: int = 2048,
    cmap: str = DEFAULT_CMAP,
    vmin: float = -0.10,
    vmax: float = 0.10,
) -> None:
    """Create a PNG browse image for the output product from product in NetCDF file."""
    if dataset_name not in DISPLACEMENT_PRODUCTS.names:
        raise ValueError(f"{dataset_name} is not a valid dataset name")

    with h5netcdf.File(input_filename, "r") as hf:
        arr = hf[dataset_name][()]
        if "recommended_mask" in hf:
            mask = hf["recommended_mask"][()]
        else:
            conncomps = np.nan_to_num(
                hf[DISPLACEMENT_PRODUCTS.connected_component_labels.name]
            )
            mask = conncomps != 0

    make_browse_image_from_arr(
        output_filename, arr, mask, max_dim_allowed, cmap, vmin, vmax
    )
