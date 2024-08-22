from typing import Tuple

import numpy as np
import pysolid
import rasterio
from dolphin import Filename, io
from opera_utils import get_zero_doppler_time
from rasterio.coords import BoundingBox
from rasterio.warp import transform as warp_transform
from rasterio.warp import transform_bounds
from scipy.interpolate import RegularGridInterpolator

from disp_s1._reference import ReferencePoint


def load_raster_bounds_and_transform(
    filename: Filename,
) -> Tuple[BoundingBox, rasterio.crs.CRS, rasterio.Affine, Tuple[int, int]]:
    """Load bounds, transform, CRS, and shape from a raster file."""
    with rasterio.open(filename) as src:
        bounds = src.bounds
        crs = src.crs
        affine_transform = src.transform  # Rename to avoid conflict
        height, width = src.shape
    return bounds, crs, affine_transform, (height, width)


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
    """Generate the ATR dictionary for pysolid."""
    height_small = max(25, height // 100)
    width_small = max(25, width // 100)
    return {
        "LENGTH": height_small,
        "WIDTH": width_small,
        "X_FIRST": bounds.left - margin_degrees,
        "Y_FIRST": bounds.top + margin_degrees,
        "X_STEP": (bounds.right - bounds.left + 2 * margin_degrees) / width_small,
        "Y_STEP": -(bounds.top - bounds.bottom + 2 * margin_degrees) / height_small,
    }


def interpolate_set_components(
    lat: np.ndarray,
    lon: np.ndarray,
    lat_geo: np.ndarray,
    lon_geo: np.ndarray,
    set_east: np.ndarray,
    set_north: np.ndarray,
    set_up: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate SET components to the unwrapped interferogram grid."""
    # Interpolate SET components
    set_east_interp = RegularGridInterpolator(
        (lat_geo, lon_geo),
        set_east,
    )(np.stack((lat, lon), axis=-1))
    set_north_interp = RegularGridInterpolator(
        (lat_geo, lon_geo),
        set_north,
    )(np.stack((lat, lon), axis=-1))
    set_up_interp = RegularGridInterpolator(
        (lat_geo, lon_geo),
        set_up,
    )(np.stack((lat, lon), axis=-1))

    return set_east_interp, set_north_interp, set_up_interp


def calculate_solid_earth_tides_correction(
    unw_filename: Filename,
    reference_cslc_file: Filename,
    secondary_cslc_file: Filename,
    los_east_file: Filename,
    los_north_file: Filename,
    reference_point: ReferencePoint | None = None,
) -> np.ndarray:
    """Calculate the relative displacement correction for solid earth tides."""
    # Load bounds, transform, CRS, and shape from the unwrapped interferogram
    bounds, crs, affine_transform, (height, width) = load_raster_bounds_and_transform(
        unw_filename
    )
    bounds = transform_bounds_to_epsg4326(bounds, crs)

    # Extract timing information from the CSLC files
    reference_start_time = get_zero_doppler_time(reference_cslc_file, type_="start")
    secondary_start_time = get_zero_doppler_time(secondary_cslc_file, type_="start")

    # Create the ATR object for pysolid
    atr = generate_atr(bounds, height, width)

    # Run pysolid to get SET in ENU coordinate system for both times
    set_east, set_north, set_up = pysolid.calc_solid_earth_tides_grid(
        secondary_start_time, atr, display=False, verbose=True
    )
    set_east_ref, set_north_ref, set_up_ref = pysolid.calc_solid_earth_tides_grid(
        reference_start_time, atr, display=False, verbose=True
    )

    # Generate the lat/lon arrays for the SET geogrid
    lat_geo_array = np.linspace(
        atr["Y_FIRST"],
        atr["Y_FIRST"] + atr["Y_STEP"] * atr["LENGTH"],
        num=atr["LENGTH"],
    )
    if np.sign(atr["Y_STEP"]) > 0:
        lat_geo_array = lat_geo_array[::-1]

    lon_geo_array = np.linspace(
        atr["X_FIRST"], atr["X_FIRST"] + atr["X_STEP"] * atr["WIDTH"], num=atr["WIDTH"]
    )

    # Create a grid of coordinates for the original unwrapped interferogram
    y, x = np.mgrid[0:height, 0:width]

    # Convert grid indices (x, y) to
    lon, lat = rasterio.transform.xy(affine_transform, y, x)

    # Convert the coordinates to numpy arrays
    lon = np.array(lon)
    lat = np.array(lat)

    # If the CRS is not geographic, reproject the coordinates to EPSG:4326
    if not crs.is_geographic:
        lon, lat = warp_transform(crs, "EPSG:4326", lon.flatten(), lat.flatten())
        lon = np.array(lon).reshape(x.shape)
        lat = np.array(lat).reshape(y.shape)

    # Clip lat/lon to ensure they are within grid bounds
    lat = np.clip(lat, lat_geo_array.min(), lat_geo_array.max())
    lon = np.clip(lon, lon_geo_array.min(), lon_geo_array.max())

    # Interpolate SET components
    set_east_interp, set_north_interp, set_up_interp = interpolate_set_components(
        lat,
        lon,
        lat_geo_array,
        lon_geo_array,
        set_east - set_east_ref,
        set_north - set_north_ref,
        set_up - set_up_ref,
    )

    # Load LOS components
    los_east = io.load_gdal(los_east_file, masked=True)
    los_north = io.load_gdal(los_north_file, masked=True)
    los_up = np.sqrt(1 - los_east**2 - los_north**2)

    # project ENU onto LOS
    set_los = (
        (set_east_interp * los_east)
        + (set_north_interp * los_north)
        + (set_up_interp * los_up)
    )

    if reference_point is None:
        return set_los

    # Subtract the reference point
    ref_row, ref_col = reference_point.row, reference_point.col
    return set_los - set_los[ref_row, ref_col]
