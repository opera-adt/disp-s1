from pathlib import Path
from typing import NamedTuple

import rasterio.warp
from dolphin._types import PathOrStr


class ReferencePoint(NamedTuple):
    row: int
    col: int
    lat: float
    lon: float


def read_reference_point(timeseries_path: PathOrStr) -> ReferencePoint:
    """Read the reference point metadata from dolphin's timeseries output.

    We also convert the row and column indices to latitude and longitude coordinates.

    Parameters
    ----------
    timeseries_path : str or Path
        Path to the directory containing the timeseries data and reference point file.

    Returns
    -------
    NamedTuple[int, int, float, float]
        A ReferencePoint tuple containing (row, col, lat, lon).

    Raises
    ------
    FileNotFoundError
        If the reference point file is not found.
    ValueError
        If the reference point file content is invalid.

    Notes
    -----
    The reference point file should be a plain text file containing two
    comma-separated integers representing the reference row and column.

    """
    # TODO: make this better in dolphin
    # There should be a more clear way to mark this metadata on the rasters themselves,
    # Rather than a sidecar text file.
    timeseries_dir = Path(timeseries_path)
    ref_point_file = timeseries_dir / "reference_point.txt"

    if not ref_point_file.exists():
        raise FileNotFoundError(f"Reference point file not found: {ref_point_file}")

    try:
        ref_row, ref_col = map(int, ref_point_file.read_text().strip().split(","))
    except ValueError:
        raise ValueError(f"Invalid content in reference point file: {ref_point_file}")

    # Get the CRS from the first timeseries file
    timeseries_files = list(timeseries_dir.glob("*.tif"))
    if not timeseries_files:
        raise FileNotFoundError(f"No timeseries files found in {timeseries_dir}")

    import rasterio as rio

    with rio.open(timeseries_files[0]) as src:
        crs = src.crs
        # Convert row/col to x/y coordinates
        x, y = src.xy(ref_row, ref_col)

    # Convert x/y to lat/lon
    ref_lon, ref_lat = rasterio.warp.transform(crs, rio.CRS.from_epsg(4326), [x], [y])

    return ReferencePoint(ref_row, ref_col, ref_lat[0], ref_lon[0])
