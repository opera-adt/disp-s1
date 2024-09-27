import logging
from datetime import date, datetime
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd
import rasterio
from dolphin import Filename, io
from pysolid import calc_solid_earth_tides_grid
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject, transform_bounds
from scipy.ndimage import zoom

# https://github.com/pandas-dev/pandas-stubs/blob/1bc27e67098106089ce1e61b60c42aa81ec286af/pandas-stubs/_typing.pyi#L65-L66
DateTimeLike: TypeAlias = date | datetime | pd.Timestamp

logger = logging.getLogger(__name__)


def resample_to_target(array: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resample a 2D array to a target shape using zoom."""
    if array.shape == target_shape:
        return array
    zoom_factors = (target_shape[0] / array.shape[0], target_shape[1] / array.shape[1])
    return zoom(array, zoom_factors, order=1)  # Linear interpolation


def calculate_solid_earth_tides_correction(
    like_filename: Filename,
    reference_start_time: DateTimeLike,
    reference_stop_time: DateTimeLike,
    secondary_start_time: DateTimeLike,
    secondary_stop_time: DateTimeLike,
    los_east_file: Filename,
    los_north_file: Filename,
    orbit_direction: Literal["ascending", "descending"],
    reference_point: tuple[int, int] | None = None,
) -> np.ndarray:
    """Calculate the relative displacement correction for solid earth tides.

    This function computes the solid earth tides correction for InSAR data
    by calculating the difference between tidal displacements at two acquisition times
    and projecting it onto the satellite's line of sight (LOS).

    Parameters
    ----------
    like_filename : Filename
        Path to a reference raster file used for spatial information.
    reference_start_time : DateTimeLike
        Start time of the reference acquisition.
    reference_stop_time : DateTimeLike
        Stop time of the reference acquisition.
    secondary_start_time : DateTimeLike
        Start time of the secondary acquisition.
    secondary_stop_time : DateTimeLike
        Stop time of the secondary acquisition.
    los_east_file : Filename
        Path to the raster file containing the east component of the LOS vector.
    los_north_file : Filename
        Path to the raster file containing the north component of the LOS vector.
    orbit_direction : {'ascending', 'descending'}, optional
        Orbit direction of the satellite, by default "ascending".
    reference_point : tuple[int, int] or None, optional
        Spatial reference (row, col) for relative corrections.
        If None, no reference point correction is applied.
        Default is None.

    Returns
    -------
    np.ndarray
        2D numpy array containing the solid earth tides correction in meters,
        with the same shape and projection as the reference raster.
        Nodata values are filled with `nan`

    Notes
    -----
    The function uses a coarse grid for initial calculations to improve performance,
    then interpolates to the full resolution. The correction is provided in the
    line-of-sight (LOS) direction, with positive indicating ground motion toward
    the satellite.

    The returned array is masked where the LOS east component is zero.

    If a reference point is provided, the correction at that point is subtracted
    from the entire array to provide a relative correction.

    """
    # Load bounds, CRS, and transform from like_filename
    with rasterio.open(like_filename) as src:
        bounds = src.bounds
        crs = src.crs
        affine_transform = src.transform
        width = src.width
        height = src.height

    # Transform bounds to EPSG:4326
    if not crs.is_geographic:
        bounds_geo = transform_bounds(crs, "EPSG:4326", *bounds)
    else:
        bounds_geo = bounds

    min_lon, min_lat, max_lon, max_lat = bounds_geo

    # Define coarse grid size (adjust as needed)
    grid_rows = 500
    grid_cols = 500

    # Create meta dictionary for pysolid
    meta = {
        "LENGTH": grid_rows,
        "WIDTH": grid_cols,
        "X_FIRST": min_lon,
        "Y_FIRST": max_lat,
        "X_STEP": (max_lon - min_lon) / (grid_cols - 1),
        "Y_STEP": -(max_lat - min_lat) / (grid_rows - 1),
    }

    # Compute SET corrections at start and end times for reference
    logger.info("Computing SET corrections for reference image")
    tide_e_start_ref, tide_n_start_ref, tide_u_start_ref = calc_solid_earth_tides_grid(
        reference_start_time, meta, verbose=False
    )
    tide_e_end_ref, tide_n_end_ref, tide_u_end_ref = calc_solid_earth_tides_grid(
        reference_stop_time, meta, verbose=False
    )

    # Compute blending weights
    if orbit_direction.lower() == "ascending":
        # If ascending, the "start" is at the bottom of the north-up image
        start_weights = np.linspace(1, 0, grid_rows).reshape(-1, 1)
    elif orbit_direction.lower() == "descending":
        # descending has the "start" at the top of the image
        start_weights = np.linspace(0, 1, grid_rows).reshape(-1, 1)
    else:
        raise ValueError("orbit_direction must be 'ascending' or 'descending'")

    end_weights = 1 - start_weights

    # Blend corrections for reference
    tide_e_ref = tide_e_start_ref * start_weights + tide_e_end_ref * end_weights
    tide_n_ref = tide_n_start_ref * start_weights + tide_n_end_ref * end_weights
    tide_u_ref = tide_u_start_ref * start_weights + tide_u_end_ref * end_weights

    # Compute SET corrections at start and end times for secondary
    logger.info("Computing SET corrections for secondary image")
    tide_e_start_sec, tide_n_start_sec, tide_u_start_sec = calc_solid_earth_tides_grid(
        secondary_start_time, meta, verbose=False
    )
    tide_e_end_sec, tide_n_end_sec, tide_u_end_sec = calc_solid_earth_tides_grid(
        secondary_stop_time, meta, verbose=False
    )

    # Blend corrections for secondary
    tide_e_sec = tide_e_start_sec * start_weights + tide_e_end_sec * end_weights
    tide_n_sec = tide_n_start_sec * start_weights + tide_n_end_sec * end_weights
    tide_u_sec = tide_u_start_sec * start_weights + tide_u_end_sec * end_weights

    # Compute differential corrections
    logger.info("Computing differential SET corrections")
    delta_tide_e = tide_e_sec - tide_e_ref
    delta_tide_n = tide_n_sec - tide_n_ref
    delta_tide_u = tide_u_sec - tide_u_ref

    # Define source transform and CRS
    src_transform = Affine(
        meta["X_STEP"], 0, meta["X_FIRST"], 0, meta["Y_STEP"], meta["Y_FIRST"]
    )
    src_crs = CRS.from_epsg(4326)

    # Define destination arrays
    dest_delta_tide_e = np.empty((height, width), dtype=np.float32)
    dest_delta_tide_n = np.empty((height, width), dtype=np.float32)
    dest_delta_tide_u = np.empty((height, width), dtype=np.float32)

    # Reproject corrections to UTM
    logger.info("Reprojecting corrections to UTM")
    reproject(
        source=delta_tide_e,
        destination=dest_delta_tide_e,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=affine_transform,
        dst_crs=crs,
        resampling=Resampling.bilinear,
    )
    reproject(
        source=delta_tide_n,
        destination=dest_delta_tide_n,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=affine_transform,
        dst_crs=crs,
        resampling=Resampling.bilinear,
    )
    reproject(
        source=delta_tide_u,
        destination=dest_delta_tide_u,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=affine_transform,
        dst_crs=crs,
        resampling=Resampling.bilinear,
    )

    # Load LOS vectors
    logger.info("Loading LOS vectors")
    los_east = io.load_gdal(los_east_file, masked=True)
    los_north = io.load_gdal(los_north_file, masked=True)

    # Check shapes and resample if necessary
    if los_east.shape != (height, width):
        los_east = resample_to_target(los_east, (height, width))
        los_north = resample_to_target(los_north, (height, width))

    los_up = np.sqrt(1 - los_east**2 - los_north**2)

    logger.info("Projecting corrections to LOS")
    set_los = (
        dest_delta_tide_e * los_east
        + dest_delta_tide_n * los_north
        + dest_delta_tide_u * los_up
    )

    # Apply reference point correction if needed
    if reference_point is not None:
        ref_row, ref_col = reference_point
        set_los -= set_los[ref_row, ref_col]

    return set_los.filled(np.nan)
