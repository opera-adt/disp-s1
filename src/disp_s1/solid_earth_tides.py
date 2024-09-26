from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from typing import Optional, TypeAlias

import numpy as np
import pandas as pd
import rasterio
from dolphin import Filename, io
from pysolid.solid import solid_grid
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom

from disp_s1._reference import ReferencePoint

# https://github.com/pandas-dev/pandas-stubs/blob/1bc27e67098106089ce1e61b60c42aa81ec286af/pandas-stubs/_typing.pyi#L65-L66
DateTimeLike: TypeAlias = date | datetime | pd.Timestamp


def resample_to_target(array: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resample a 2D array to a target shape using zoom."""
    if array.shape == target_shape:
        return array
    zoom_factors = (target_shape[0] / array.shape[0], target_shape[1] / array.shape[1])
    return zoom(array, zoom_factors, order=1)  # Linear interpolation


def load_raster_bounds_and_transform(
    filename: Filename,
) -> tuple[BoundingBox, CRS, rasterio.Affine, tuple[int, int]]:
    """Load bounds, transform, CRS, and shape from a raster file."""
    with rasterio.open(filename) as src:
        bounds = src.bounds
        crs = src.crs
        affine_transform = src.transform
        height, width = src.shape
    return bounds, crs, affine_transform, (height, width)


def transform_bounds_to_epsg4326(bounds: BoundingBox, source_crs: CRS) -> BoundingBox:
    """Transform bounds to EPSG:4326 if CRS is not geographic."""
    if not source_crs.is_geographic:
        transformed_bounds = transform_bounds(source_crs, "EPSG:4326", *bounds)
        return BoundingBox(*transformed_bounds)
    return bounds


def generate_atr(
    bounds: BoundingBox, height: int, width: int, margin_degrees: float = 0.1
) -> dict:
    """Generate the ATR dictionary for pysolid grid."""
    min_grid_size = 25
    height_small = max(min_grid_size, height // 100)
    width_small = max(min_grid_size, width // 100)
    return {
        "LENGTH": height_small,
        "WIDTH": width_small,
        "X_FIRST": bounds.left - margin_degrees,
        "Y_FIRST": bounds.top + margin_degrees,
        "X_STEP": (bounds.right - bounds.left + 2 * margin_degrees) / width_small,
        "Y_STEP": -(bounds.top - bounds.bottom + 2 * margin_degrees) / height_small,
    }


def interpolate_set_components(
    n: np.ndarray,
    e: np.ndarray,
    y_arr: np.ndarray,
    x_arr: np.ndarray,
    set_east: np.ndarray,
    set_north: np.ndarray,
    set_up: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate SET components to the unwrapped interferogram grid."""
    points = np.stack((n, e), axis=-1)
    interpolator_east = RegularGridInterpolator(
        (y_arr, x_arr), set_east, bounds_error=False, fill_value=None
    )
    interpolator_north = RegularGridInterpolator(
        (y_arr, x_arr), set_north, bounds_error=False, fill_value=None
    )
    interpolator_up = RegularGridInterpolator(
        (y_arr, x_arr), set_up, bounds_error=False, fill_value=None
    )

    return (
        interpolator_east(points),
        interpolator_north(points),
        interpolator_up(points),
    )


def run_solid_grid(az_time, lat, lon, res_lat, res_lon):
    """Run solid_grid from pysolid."""
    az_time = pd.to_datetime(az_time)
    return solid_grid(
        az_time.year,
        az_time.month,
        az_time.day,
        az_time.hour,
        az_time.minute,
        az_time.second,
        lat,
        res_lat,
        1,
        lon,
        res_lon,
        1,
    )


def calculate_time_ranges(
    reference_start_time: DateTimeLike,
    reference_stop_time: DateTimeLike,
    secondary_start_time: DateTimeLike,
    secondary_stop_time: DateTimeLike,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Generate time ranges for reference and secondary files."""
    ref_time_range = pd.date_range(
        start=reference_start_time,
        end=reference_stop_time,
        periods=shape[0],
    )

    sec_time_range = pd.date_range(
        start=secondary_start_time,
        end=secondary_stop_time,
        periods=shape[0],
    )

    ref_time_tile = np.tile(ref_time_range.values, (shape[1], 1)).T
    sec_time_tile = np.tile(sec_time_range.values, (shape[1], 1)).T
    return ref_time_tile, sec_time_tile


def calculate_solid_earth_tides_correction(
    like_filename: Filename,
    reference_start_time: DateTimeLike,
    reference_stop_time: DateTimeLike,
    secondary_start_time: DateTimeLike,
    secondary_stop_time: DateTimeLike,
    los_east_file: Filename,
    los_north_file: Filename,
    reference_point: Optional[ReferencePoint] = None,
) -> np.ndarray:
    """Calculate the relative displacement correction for solid earth tides."""
    # Load bounds, transform, CRS, and shape from the unwrapped interferogram
    bounds, crs, affine_transform, (height, width) = load_raster_bounds_and_transform(
        like_filename
    )
    bounds_geo = transform_bounds_to_epsg4326(bounds, crs)
    atr = generate_atr(bounds_geo, height, width)

    # Generate the lat/lon arrays for the SET geogrid
    lat_geo_array = np.linspace(
        atr["Y_FIRST"],
        atr["Y_FIRST"] + atr["Y_STEP"] * atr["LENGTH"],
        num=atr["LENGTH"],
    )
    lon_geo_array = np.linspace(
        atr["X_FIRST"], atr["X_FIRST"] + atr["X_STEP"] * atr["WIDTH"], num=atr["WIDTH"]
    )
    lat_geo_mesh, lon_geo_mesh = np.meshgrid(
        lat_geo_array, lon_geo_array, indexing="ij"
    )

    # Generate time ranges
    ref_time_tile, sec_time_tile = calculate_time_ranges(
        reference_start_time,
        reference_stop_time,
        secondary_start_time,
        secondary_stop_time,
        shape=(atr["LENGTH"], atr["WIDTH"]),
    )

    # Prepare resolution arrays
    res_lat_arr = np.ones(ref_time_tile.shape) * atr["Y_STEP"]
    res_lon_arr = np.ones(ref_time_tile.shape) * atr["X_STEP"]

    # Parallelize grid calculation using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        ref_input_data = zip(
            ref_time_tile.ravel(),
            lat_geo_mesh.ravel(),
            lon_geo_mesh.ravel(),
            res_lat_arr.ravel(),
            res_lon_arr.ravel(),
        )
        sec_input_data = zip(
            sec_time_tile.ravel(),
            lat_geo_mesh.ravel(),
            lon_geo_mesh.ravel(),
            res_lat_arr.ravel(),
            res_lon_arr.ravel(),
        )

        ref_results = list(executor.map(lambda x: run_solid_grid(*x), ref_input_data))
        sec_results = list(executor.map(lambda x: run_solid_grid(*x), sec_input_data))

    ref_e_flat, ref_n_flat, ref_u_flat = zip(*ref_results)
    set_east_ref, set_north_ref, set_up_ref = (
        np.array(ref_e_flat).reshape(atr["LENGTH"], atr["WIDTH"]),
        np.array(ref_n_flat).reshape(atr["LENGTH"], atr["WIDTH"]),
        np.array(ref_u_flat).reshape(atr["LENGTH"], atr["WIDTH"]),
    )

    sec_e_flat, sec_n_flat, sec_u_flat = zip(*sec_results)
    set_east_sec, set_north_sec, set_up_sec = (
        np.array(sec_e_flat).reshape(atr["LENGTH"], atr["WIDTH"]),
        np.array(sec_n_flat).reshape(atr["LENGTH"], atr["WIDTH"]),
        np.array(sec_u_flat).reshape(atr["LENGTH"], atr["WIDTH"]),
    )

    # Generate coordinate arrays for the original unwrapped interferogram
    y_coord_array = np.linspace(bounds.top, bounds.bottom, num=atr["LENGTH"])
    x_coord_array = np.linspace(bounds.left, bounds.right, num=atr["WIDTH"])
    id_y, id_x = np.mgrid[0:height, 0:width]

    # Convert grid indices (x, y) to UTM
    x, y = rasterio.transform.xy(affine_transform, id_y, id_x)
    e_arr = np.clip(np.array(x), x_coord_array.min(), x_coord_array.max())
    n_arr = np.clip(np.array(y), y_coord_array.min(), y_coord_array.max())

    # Interpolate SET components
    set_east_interp, set_north_interp, set_up_interp = interpolate_set_components(
        n_arr,
        e_arr,
        y_coord_array,
        x_coord_array,
        set_east_sec - set_east_ref,
        set_north_sec - set_north_ref,
        set_up_sec - set_up_ref,
    )

    # Load LOS components
    los_east = io.load_gdal(los_east_file, masked=True)
    los_north = io.load_gdal(los_north_file, masked=True)

    # Check for dimension consistency and resample if they are not
    if not los_east.shape == (height, width):
        los_east = resample_to_target(los_east, (height, width))
        los_north = resample_to_target(los_north, (height, width))

    los_up = np.sqrt(1 - los_east**2 - los_north**2)

    # Solid earth tides datacube along the LOS in meters
    set_los = (
        set_east_interp * los_east
        + set_north_interp * los_north
        + set_up_interp * los_up
    )

    mask = los_east != 0

    if reference_point is None:
        return set_los * mask

    # Subtract the reference point
    ref_row, ref_col = reference_point.row, reference_point.col
    return (set_los - set_los[ref_row, ref_col]) * mask
