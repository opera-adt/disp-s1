from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from dolphin import Filename, io
from opera_utils import get_zero_doppler_time
from pysolid.solid import solid_grid
from rasterio.coords import BoundingBox
from rasterio.warp import transform_bounds
from scipy.interpolate import RegularGridInterpolator

from disp_s1._reference import ReferencePoint


def load_raster_bounds_and_transform(
    filename: Filename,
) -> Tuple[BoundingBox, rasterio.crs.CRS, rasterio.Affine, Tuple[int, int]]:
    """Load bounds, transform, CRS, and shape from a raster file."""
    try:
        with rasterio.open(filename) as src:
            bounds = src.bounds
            crs = src.crs
            affine_transform = src.transform
            height, width = src.shape
        return bounds, crs, affine_transform, (height, width)
    except Exception as e:
        raise RuntimeError(f"Error loading raster file {filename}: {e}") from e


def transform_bounds_to_epsg4326(
    bounds: BoundingBox, crs: rasterio.crs.CRS
) -> BoundingBox:
    """Transform bounds to EPSG:4326 if CRS is not geographic."""
    if not crs.is_geographic:
        transformed_bounds = transform_bounds(crs, "EPSG:4326", *bounds)
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate SET components to the unwrapped interferogram grid."""
    points = np.stack((n, e), axis=-1)
    return (
        RegularGridInterpolator((y_arr, x_arr), set_east)(points),
        RegularGridInterpolator((y_arr, x_arr), set_north)(points),
        RegularGridInterpolator((y_arr, x_arr), set_up)(points),
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
    reference_start_time: pd.Timestamp,
    reference_stop_time: pd.Timestamp,
    secondary_start_time: pd.Timestamp,
    secondary_stop_time: pd.Timestamp,
    atr: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate time ranges for reference and secondary files."""
    ref_time_range = pd.to_datetime(
        np.linspace(
            reference_start_time.value, reference_stop_time.value, atr["LENGTH"]
        )
    )
    sec_time_range = pd.to_datetime(
        np.linspace(
            secondary_start_time.value, secondary_stop_time.value, atr["LENGTH"]
        )
    )
    ref_time_tile = np.tile(ref_time_range, (atr["WIDTH"], 1)).T
    sec_time_tile = np.tile(sec_time_range, (atr["WIDTH"], 1)).T
    return ref_time_tile, sec_time_tile


def calculate_solid_earth_tides_correction(
    ifgram_filename: Filename,
    reference_cslc_file: Filename,
    secondary_cslc_file: Filename,
    los_east_file: Filename,
    los_north_file: Filename,
    reference_point: Optional[ReferencePoint] = None,
) -> np.ndarray:
    """Calculate the relative displacement correction for solid earth tides."""
    # Load bounds, transform, CRS, and shape from the unwrapped interferogram
    bounds, crs, affine_transform, (height, width) = load_raster_bounds_and_transform(
        ifgram_filename
    )
    bounds_geo = transform_bounds_to_epsg4326(bounds, crs)
    atr = generate_atr(bounds_geo, height, width)

    # Extract timing information from the CSLC files
    reference_start_time = pd.to_datetime(
        get_zero_doppler_time(reference_cslc_file, type_="start")
    )
    reference_stop_time = pd.to_datetime(
        get_zero_doppler_time(reference_cslc_file, type_="end")
    )
    secondary_start_time = pd.to_datetime(
        get_zero_doppler_time(secondary_cslc_file, type_="start")
    )
    secondary_stop_time = pd.to_datetime(
        get_zero_doppler_time(secondary_cslc_file, type_="end")
    )

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
        atr,
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
    los_up = np.sqrt(1 - los_east**2 - los_north**2)

    # Solid earth tides datacube along the LOS in meters
    set_los = (
        set_east_interp * los_east
        + set_north_interp * los_north
        + set_up_interp * los_up
    )

    if reference_point is None:
        return set_los

    # Subtract the reference point
    ref_row, ref_col = reference_point.row, reference_point.col
    return set_los - set_los[ref_row, ref_col]
