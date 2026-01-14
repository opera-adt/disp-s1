"""Water mask creation utilities for OPERA data.

This module provides tools for creating water masks for specific geographic regions
by downloading data from S3 and processing it with GDAL.
"""

import logging
from pathlib import Path

import backoff
import numpy as np
import shapely.ops
import shapely.wkt
from botocore.session import Session
from opera_utils import get_frame_bbox
from osgeo import gdal
from pyproj import Transformer
from shapely.geometry import LinearRing, Polygon, box

# Enable GDAL exceptions for better error handling
gdal.UseExceptions()

logger = logging.getLogger(__name__)

__all__ = ["create_water_mask"]

# Constants
EARTH_APPROX_CIRCUMFERENCE = 40_075_017.0  # meters
EARTH_RADIUS = EARTH_APPROX_CIRCUMFERENCE / (2 * np.pi)
MASK_S3_URL = "s3://opera-water-mask/v0.3/EPSG4326.vrt"


# ============================================================================
# Geospatial Utilities
# ============================================================================


def margin_km_to_deg(margin_in_km: float) -> float:
    """Convert margin from kilometers to degrees at the equator.

    Parameters
    ----------
    margin_in_km : float
        Margin in kilometers

    Returns
    -------
    float
        Margin in degrees

    """
    km_to_deg_at_equator = 1000.0 / (EARTH_APPROX_CIRCUMFERENCE / 360.0)
    return margin_in_km * km_to_deg_at_equator


def margin_km_to_longitude_deg(margin_in_km: float, lat: float = 0) -> float:
    """Convert margin from kilometers to longitude degrees at a given latitude.

    Parameters
    ----------
    margin_in_km : float
        Margin in kilometers
    lat : float, optional
        Latitude in degrees, by default 0 (equator)

    Returns
    -------
    float
        Margin in longitude degrees

    """
    delta_lon = (
        180 * 1000 * margin_in_km / (np.pi * EARTH_RADIUS * np.cos(np.pi * lat / 180))
    )
    return delta_lon


def check_dateline(poly: Polygon) -> list[Polygon]:
    """Split a polygon if it crosses the international dateline.

    Parameters
    ----------
    poly : Polygon
        Input polygon in EPSG:4326 coordinates

    Returns
    -------
    list[Polygon]
        List containing either the original polygon (if no crossing) or
        two polygons split at the dateline

    """
    x_min, _, x_max, _ = poly.bounds

    # Check for dateline crossing
    crosses_dateline = (x_max - x_min > 180.0) or (x_min <= 180.0 <= x_max)

    if not crosses_dateline:
        return [poly]

    # Handle dateline crossing
    dateline = shapely.wkt.loads("LINESTRING(180.0 -90.0, 180.0 90.0)")

    # Normalize longitudes to [0, 360] range
    x, y = poly.exterior.coords.xy
    new_x = [xi + (xi <= 0.0) * 360 for xi in x]
    new_ring = LinearRing(zip(new_x, y, strict=True))

    # Split the polygon at the dateline
    merged_lines = shapely.ops.linemerge([dateline, new_ring])
    border_lines = shapely.ops.unary_union(merged_lines)
    decomp = shapely.ops.polygonize(border_lines)

    polys = list(decomp)

    # Wrap longitudes > 180 back to [-180, 180] range
    for i, polygon in enumerate(polys):
        x, y = polygon.exterior.coords.xy
        if any(xi > 180 for xi in x):
            x_wrapped = np.asarray(x) - 360
            polys[i] = Polygon(zip(x_wrapped, y, strict=True))

    return polys


def polygon_from_bounding_box(
    bounding_box: tuple[float, float, float, float],
    margin_in_km: float,
) -> Polygon:
    """Create a polygon from a bounding box with margin.

    Parameters
    ----------
    bounding_box : tuple[float, float, float, float]
        Bounding box as (West, South, East, North) in decimal degrees
    margin_in_km : float
        Margin to add in kilometers

    Returns
    -------
    Polygon
        Polygon in EPSG:4326 coordinates with margin applied

    """
    lon_min, lat_min, lon_max, lat_max = bounding_box
    logger.info(f"Creating polygon from bounding box: {bounding_box}")

    # Use worst-case latitude for longitude margin calculation
    lat_worst_case = max(abs(lat_min), abs(lat_max))

    # Convert margin to degrees
    lat_margin = margin_km_to_deg(margin_in_km)
    lon_margin = margin_km_to_longitude_deg(margin_in_km, lat=lat_worst_case)

    # Handle antimeridian crossing
    if lon_max - lon_min > 180:
        lon_min, lon_max = lon_max, lon_min

    # Create polygon with margin, clamping to valid lat range
    poly = box(
        lon_min - lon_margin,
        max(lat_min - lat_margin, -90.0),
        lon_max + lon_margin,
        min(lat_max + lat_margin, 90.0),
    )

    return poly


# ============================================================================
# Water Mask Creation
# ============================================================================
def set_aws_env_from_saml(profile_name="saml-pub", region="us-west-2"):
    """Set AWS credentials from SAML/SSO profile as environment variables."""
    session = Session(profile=profile_name)
    creds = session.get_credentials().get_frozen_credentials()

    gdal.SetConfigOption("AWS_REGION", region)
    gdal.SetConfigOption("AWS_SECRET_ACCESS_KEY", creds.secret_key)
    gdal.SetConfigOption("AWS_ACCESS_KEY_ID", creds.access_key)
    gdal.SetConfigOption("AWS_SESSION_TOKEN", creds.token)

    print("AWS credentials loaded into environment from profile:", profile_name)


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_time=600,
    max_value=32,
    on_backoff=lambda details: logger.warning(
        f"Retrying download after error (attempt {details['tries']})"
    ),
)
def download_map(polys: list[Polygon], outfile: Path) -> list[Path]:
    """Download water mask regions from S3 for the given polygons.

    Uses GDAL's /vsis3/ virtual file system to access S3 data directly.

    Parameters
    ----------
    polys : list[Polygon]
        Polygons defining regions to download
    outfile : Path
        Output VRT file path

    Returns
    -------
    list[Path]
        Paths to downloaded GeoTIFF files

    Raises
    ------
    RuntimeError
        If GDAL cannot open the S3 data
    Exception
        If download fails after retries

    """
    logger.info(f"Creating water mask VRT: {outfile}")

    file_root = outfile.parent / outfile.stem
    output_tifs: list[Path] = []

    # Use GDAL's virtual S3 file system
    vrt_filename = f"/vsis3/{MASK_S3_URL.replace('s3://', '')}"

    for idx, poly in enumerate(polys):
        output_path = f"{file_root}_{idx}.tif"
        x_min, y_min, x_max, y_max = poly.bounds

        logger.info(
            f"Downloading region {idx + 1}/{len(polys)}: "
            f"bbox=[{x_min:.4f}, {y_min:.4f}, {x_max:.4f}, {y_max:.4f}]"
        )

        # Open the VRT and extract the region
        ds = gdal.Open(vrt_filename, gdal.GA_ReadOnly)
        if ds is None:
            raise RuntimeError(
                f"Failed to open {MASK_S3_URL}. "
                "Check S3 access credentials and network connection."
            )

        gdal.Translate(
            output_path,
            ds,
            format="GTiff",
            projWin=[x_min, y_max, x_max, y_min],
        )
        ds = None  # Close dataset

        output_tifs.append(Path(output_path))

    # Build VRT from downloaded tiles
    gdal.BuildVRT(str(outfile), [str(p) for p in output_tifs])
    logger.info(f"Downloaded {len(output_tifs)} tile(s) to {outfile.parent}")

    return output_tifs


def create_mask_from_distance(
    water_distance_file: Path,
    output_file: Path,
    land_buffer: int = 1,
    ocean_buffer: int = 1,
) -> None:
    """Create a binary water mask from a distance raster.

    Parameters
    ----------
    water_distance_file : Path
        Input water distance raster (negative for land, positive for water)
    output_file : Path
        Output binary mask file path
    land_buffer : int, optional
        Buffer in km to add to land water regions (reduces masking), by default 1
    ocean_buffer : int, optional
        Buffer in km to add to ocean water regions (reduces masking), by default 1

    """
    logger.info(f"Creating binary mask from distance raster: {water_distance_file}")

    # Open the distance raster
    ds = gdal.Open(str(water_distance_file), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Failed to open water distance file: {water_distance_file}")

    band = ds.GetRasterBand(1)
    distance_data = band.ReadAsArray()

    # Create binary mask
    # Negative values = land water (lakes, rivers) - keep if > -land_buffer km
    # Positive values = ocean water - keep if < ocean_buffer km
    # Convert km to same units as distance raster (typically meters)
    land_threshold = -land_buffer * 1000
    ocean_threshold = ocean_buffer * 1000

    # Create mask: 1 = keep (not water), 0 = mask (water)
    mask = np.ones_like(distance_data, dtype=np.uint8)
    mask[(distance_data >= land_threshold) & (distance_data <= ocean_threshold)] = 0

    # Write output
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        str(output_file),
        ds.RasterXSize,
        ds.RasterYSize,
        1,
        gdal.GDT_Byte,
        options=["COMPRESS=LZW"],
    )

    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())

    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(mask)
    out_band.SetNoDataValue(255)

    # Close datasets
    out_band = None
    out_ds = None
    band = None
    ds = None

    logger.info(f"Binary mask created: {output_file}")


def create_water_mask(
    frame_id: int | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    output: Path = Path("water_binary_mask.vrt"),
    margin: int = 5,
    land_buffer: int = 1,
    ocean_buffer: int = 1,
    debug: bool = False,
    aws_profile: str = "saml-pub",
    aws_region: str = "us-west-2",
) -> None:
    """Create a binary water mask for a geographic region.

    Either frame_id or bbox must be provided.

    Parameters
    ----------
    frame_id : int, optional
        DISP-S1 frame ID to create mask for
    bbox : tuple[float, float, float, float], optional
        Bounding box as (West, South, East, North) in decimal degrees
    output : Path, optional
        Output binary mask file path, by default "water_binary_mask.tif"
    margin : int, optional
        Margin in kilometers to add to region, by default 5
    land_buffer : int, optional
        Buffer in km to add to land water regions (reduces masking), by default 1
    ocean_buffer : int, optional
        Buffer in km to add to ocean water regions (reduces masking), by default 1
    debug : bool, optional
        Enable debug logging, by default False
    aws_profile : str, optional
        AWS profile name for authentication, by default "saml-pub"
    aws_region : str, optional
        AWS region to use, by default "us-west-2"

    Raises
    ------
    ValueError
        If neither frame_id nor bbox is provided
    RuntimeError
        If S3 data cannot be accessed

    """
    if debug:
        logger.setLevel(logging.DEBUG)

    # Auth
    set_aws_env_from_saml(profile_name=aws_profile, region=aws_region)

    # Validate inputs
    if frame_id is None and bbox is None:
        raise ValueError("Must provide either frame_id or bbox")

    # Get bounding box from frame if needed
    if frame_id is not None:
        logger.info(f"Retrieving bounding box for frame {frame_id}")
        epsg, bounds = get_frame_bbox(frame_id=frame_id)
        t = Transformer.from_crs(epsg, 4326, always_xy=True)
        bbox = t.transform_bounds(*bounds)

    logger.info(f"Processing bounding box: {bbox}")
    logger.info(f"Using S3 data from: {MASK_S3_URL}")

    # Create polygon from bounding box
    assert bbox is not None  # bbox is guaranteed to be set by now
    poly = polygon_from_bounding_box(bbox, margin)

    # Handle dateline crossing
    polys = check_dateline(poly)
    if len(polys) > 1:
        logger.info("Polygon crosses dateline, split into 2 regions")

    # Download water distance map
    out_dir = output.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_vrt = out_dir / "water_mask.vrt"

    download_map(polys, temp_vrt)

    # Create binary mask with buffers
    logger.info(
        f"Creating binary mask with land_buffer={land_buffer}km, "
        f"ocean_buffer={ocean_buffer}km"
    )
    create_mask_from_distance(
        water_distance_file=temp_vrt,
        output_file=output,
        land_buffer=land_buffer,
        ocean_buffer=ocean_buffer,
    )

    logger.info(f"Water mask created successfully: {output}")
