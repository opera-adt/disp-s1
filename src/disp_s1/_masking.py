import logging
from pathlib import Path
from typing import Sequence

import numpy as np
from dolphin._types import Filename, PathOrStr
from dolphin.io import format_nc_filename, load_gdal, write_arr
from opera_utils import group_by_burst
from scipy import ndimage

logger = logging.getLogger(__name__)


def create_mask_from_distance(
    water_distance_file: PathOrStr,
    output_file: PathOrStr,
    land_buffer: int = 0,
    ocean_buffer: int = 0,
) -> None:
    """Create a binary mask from the NISAR water distance mask with buffer zones.

    This function reads a water distance file, converts it to a binary mask
    with consideration for buffer zones, and writes the result to a new file.

    Parameters
    ----------
    water_distance_file : PathOrStr
        Path to the input water distance file.
    output_file : PathOrStr
        Path to save the output binary mask file.
    land_buffer : int, optional
        Buffer distance (in km) for land pixels. Only pixels this far or farther
        from water will be considered land. Default is 0.
    ocean_buffer : int, optional
        Buffer distance (in km) for ocean pixels. Only pixels this far or farther
        from land will be considered water. Default is 0.

    Notes
    -----
    Format of `water_distance_file` is UInt8, where:
    - 0 means "land"
    - 1 - 99 are ocean water pixels. The value is the distance (in km) to the shore.
      Value is rounded up to the nearest integer.
    - 100 - 200 are inland water pixels. Value is the distance to land.

    Output is a mask where 0 represents water pixels ("bad" pixels to ignore during
    processing/unwrapping), and 1 are land pixels to use.

    The buffer arguments make the masking more conservative. For example, a land_buffer
    of 2 means only pixels 2 km or farther from water will be masked as land. This helps
    account for potential changes in water levels.

    """
    # Load the water distance data
    water_distance_data = load_gdal(water_distance_file, masked=True)

    binary_mask = convert_distance_to_binary(
        water_distance_data, land_buffer, ocean_buffer
    )

    write_arr(
        arr=binary_mask.astype(np.uint8),
        output_name=output_file,
        like_filename=water_distance_file,
        dtype="uint8",
        nodata=255,
    )


def convert_distance_to_binary(
    water_distance_data: np.ma.MaskedArray, land_buffer: int = 0, ocean_buffer: int = 0
) -> np.ndarray:
    """Convert water distance data to a binary mask considering buffer zones.

    Parameters
    ----------
    water_distance_data : np.ma.MaskedArray
        Input water distance data as a masked array.
    land_buffer : int, optional
        Buffer distance (in km) for land pixels. Only pixels this far or farther
        from water will be considered land. Default is 0.
    ocean_buffer : int, optional
        Buffer distance (in km) for ocean pixels. Only pixels this far or farther
        from land will be considered water. Default is 0.

    Returns
    -------
    np.ndarray
        Binary mask where True represents land pixels and False represents water pixels.

    Notes
    -----
    The function applies the following logic:
    - Starts with all pixels as land (True).
    - Masks inland water pixels as water (False) if they are farther from land
        than the land_buffer.
    - Masks ocean pixels as water (False) if they are farther from shore than
        `ocean_buffer`.

    """
    # Create the binary mask with buffer considerations. Start all on (assume all land)
    binary_mask = np.ma.MaskedArray(
        np.ones_like(water_distance_data, dtype=bool), mask=water_distance_data.mask
    )

    # Mask inland water pixels (considering land buffer): anything 101 or higher is land
    inland_water_mask = water_distance_data > land_buffer + 100
    binary_mask[inland_water_mask] = False
    # For ocean, only look at values 1-100, then consider buffer
    ocean_water_mask = (water_distance_data <= 100) & (
        water_distance_data > ocean_buffer
    )
    binary_mask[ocean_water_mask] = False
    # Erode away small single-pixels
    closed_mask = ndimage.binary_closing(
        binary_mask.filled(0), structure=np.ones((3, 3)), border_value=1
    )
    return closed_mask


def create_layover_shadow_masks(
    cslc_static_files: Sequence[Filename],
    output_dir: Filename,
) -> list[Path]:
    """Create binary masks from the layover shadow CSLC static files.

    In the outputs, 0 indicates a bad masked pixel, 1 is a good pixel.

    Parameters
    ----------
    cslc_static_files : Sequence[Filename]
        List of CSLC static layer files to process
    output_dir : Filename
        Directory where output masks will be saved

    Returns
    -------
    list[Path]
        List of paths to the created binary layover shadow mask files

    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    output_files = []

    for burst_id, files in group_by_burst(cslc_static_files).items():
        if len(files) > 1:
            logger.warning(f"Found multiple static files for {burst_id}: {files}")
        f = files[0]
        input_name = format_nc_filename(f, ds_name="/data/layover_shadow_mask")
        out_file = output_path / f"layover_shadow_{burst_id}.tif"
        if out_file.exists():
            output_files.append(out_file)
            continue

        logger.info(f"Extracting layover shadow mask from {f} to {out_file}")
        layover_data = load_gdal(input_name)
        # we'll ignore the nodata region to be conservative
        layover_data[layover_data == 127] = 0
        not_layover_pixels = layover_data == 0
        write_arr(
            arr=not_layover_pixels,
            output_name=out_file,
            like_filename=input_name,
            nodata=127,
        )

        output_files.append(out_file)

    return output_files
