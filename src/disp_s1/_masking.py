from pathlib import Path

import numpy as np
from dolphin._types import PathOrStr
from dolphin.io import load_gdal, write_arr


def convert_distance_to_binary(
    water_distance_file: PathOrStr,
    output_file: PathOrStr,
    land_buffer: int = 0,
    ocean_buffer: int = 0,
) -> Path:
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

    Returns
    -------
    Path
        Path to the created output file.

    Notes
    -----
    Format of `water_distance_file` is UInt8, where:
    - 0 means "land"
    - 1 - 100 are ocean water pixels. The value is the distance (in km) to the shore.
      Value is rounded up to the nearest integer.
    - 101 - 200 are inland water pixels. Value is the distance to land.

    Output is a mask where 0 represents water pixels ("bad" pixels to ignore during
    processing/unwrapping), and 1 are land pixels to use.

    The buffer arguments make the masking more conservative. For example, a land_buffer
    of 2 means only pixels 2 km or farther from water will be masked as land. This helps
    account for potential changes in water levels.

    """
    # Load the water distance data
    water_distance_data = load_gdal(water_distance_file)

    # Create the binary mask with buffer considerations
    binary_mask = np.zeros_like(water_distance_data, dtype=bool)

    # Mask land pixels (considering land buffer)
    binary_mask[water_distance_data == 0] = True

    # Mask inland water pixels (considering land buffer)
    inland_water_mask = (water_distance_data >= 101) & (water_distance_data <= 200)
    binary_mask[inland_water_mask & (water_distance_data - 100 <= land_buffer)] = True

    # Mask ocean pixels (considering ocean buffer)
    ocean_mask = (water_distance_data >= 1) & (water_distance_data <= 100)
    binary_mask[ocean_mask & (water_distance_data >= ocean_buffer)] = False

    # Write the binary mask to the output file
    output_path = write_arr(
        arr=binary_mask.astype(np.uint8),
        output_name=output_file,
        like_filename=water_distance_file,
        dtype="uint8",
        nodata=255,
    )

    return Path(output_path)
