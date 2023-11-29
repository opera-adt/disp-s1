"""Module for creating browse images for the output product."""

from __future__ import annotations

import argparse

import h5netcdf
import numpy as np
import scipy
from dolphin._types import Filename
from numpy.typing import ArrayLike
from PIL import Image

from .product_info import DISP_PRODUCT_NAMES


def _normalize_apply_gamma(arr: ArrayLike, gamma=1.0) -> np.ndarray:
    """Normalize to [0-1] and gamma correct an image array.

    Parameters
    ----------
    arr: np.ndarray
        Numpy array representing an image to be normalized and gamma corrected.
    gamma: float
        Exponent value used to gamma correct image.

    Returns
    -------
    arr: ArrayLike
        Normalized and gamma corrected image.
    """
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)

    # scale to 0-1 for gray scale and then apply gamma correction
    arr = (arr - vmin) / (vmax - vmin)

    # scale to 0-1 for gray scale and then apply gamma correction
    if gamma != 1.0:
        arr = np.power(arr, gamma)

    return arr


def _resize_to_max_pixel_dim(arr: ArrayLike, max_dim_allowed=2048) -> np.ndarray:
    """Scale shape of a given array.

    Scale up or down length and width of an array to a maximum dimension
    where the larger of length or width used to compute scaling ratio.

    Parameters
    ----------
    arr: ArrayLike
        Numpy array representing an image to be resized.
    max_dim_allowed: int
        Maximum dimension allowed for either length or width.

    Returns
    -------
    arr: ArrayLike
        Numpy array representing a resized image.
    """
    if max_dim_allowed < 1:
        raise ValueError(f"{max_dim_allowed} is not a valid max image dimension")

    # compute scaling ratio based on larger dimension
    input_shape = arr.shape
    scaling_ratio = max([max_dim_allowed / xy for xy in input_shape])

    # set NaNs set to 0 to correctly interpolate with zoom
    arr[np.isnan(arr)] = 0

    # scale original shape by scaling ratio
    arr = scipy.ndimage.zoom(arr, scaling_ratio)

    return arr


def _save_to_disk_as_greyscale(arr: ArrayLike, fname: Filename) -> None:
    """Save image array as greyscale to file.

    Parameters
    ----------
    arr: ArrayLike
        Numpy array representing an image to be saved to png file.
    fname: str
        File name of output browse image.
    """
    # scale to 1-255
    # 0 reserved for transparency
    arr = np.uint8(arr * (254)) + 1

    # save to disk in grayscale ('L')
    img = Image.fromarray(arr, mode="L")
    img.save(fname, transparency=0)


def make_browse_image_from_arr(
    output_filename: Filename,
    arr: ArrayLike,
    max_dim_allowed: int = 2048,
) -> None:
    """Create a PNG browse image for the output product from given array.

    Parameters
    ----------
    output_filename : Filename
        Name of output PNG
    arr : ArrayLike
        Array to be saved to image
    max_dim_allowed : int, default = 2048
        Size (in pixels) of the maximum allowed dimension of output image.
        Image gets rescaled with same aspect ratio.
    """
    # nomalize non-nan pixels to 0-1
    arr = _normalize_apply_gamma(arr)

    # compute browse shape and resize full size array to it
    arr = _resize_to_max_pixel_dim(arr, max_dim_allowed)

    _save_to_disk_as_greyscale(arr, output_filename)


def make_browse_image_from_nc(
    output_filename: Filename,
    input_filename: Filename,
    dataset_name: str,
    max_dim_allowed: int = 2048,
) -> None:
    """Create a PNG browse image for the output product from product in NetCDF file.

    Parameters
    ----------
    output_filename : Filename
        Name of output PNG
    input_filename : Filename
        Name of input NetCDF file.
    dataset_name: str
        Name of datast to be made into a browse image.
    max_dim_allowed : int, default = 2048
        Size (in pixels) of the maximum allowed dimension of output image.
        Image gets rescaled with same aspect ratio.
    """
    if dataset_name not in DISP_PRODUCT_NAMES:
        raise ValueError(f"{args.dataset_name} is not a valid dataset name")

    # get dataset as array from input NC file
    with h5netcdf.File(input_filename, "r") as nc_handle:
        arr = nc_handle[dataset_name][()]

    make_browse_image_from_arr(output_filename, arr, max_dim_allowed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create browse images for displacement products from command line",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o", "--out-fname", required=False, help="Optional path to output png file"
    )
    parser.add_argument("-i", "--in-fname", help="Path to input NetCDF file")
    parser.add_argument(
        "-n",
        "--dataset-name",
        choices=DISP_PRODUCT_NAMES,
        help="Name of dataset to plot from NetCDF file",
    )
    parser.add_argument(
        "-m",
        "--max-img-dim",
        type=int,
        default=2048,
        help="Maximum dimension allowed for either length or width of browse image",
    )
    args = parser.parse_args()

    # if no output file name given, set output file name to input path with
    # dataset name inserted before .nc
    if args.out_fname is None:
        args.out_fname = args.in_fname.replace(".nc", f".{args.dataset_name}.png")

    make_browse_image_from_nc(
        args.out_fname, args.in_fname, args.dataset_name, args.max_img_dim
    )
