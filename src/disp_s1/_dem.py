"""Staging utility for downloading the OPERA/NISAR DEM.

Handles coordinate transformation, dateline crossing, and AWS S3 integration.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import backoff
import numpy as np
from dolphin import Bbox
from osgeo import gdal, osr
from shapely.geometry import Polygon, box
from shapely.ops import linemerge, polygonize, unary_union
from shapely.wkt import loads

# Enable GDAL exceptions
gdal.UseExceptions()

logger = logging.getLogger(__name__)

S3_DEM_BUCKET = "opera-dem"
S3_LONLAT_VRT_KEY = "EPSG4326/EPSG4326.vrt"


@dataclass
class DEMConfig:
    """Configuration for DEM processing."""

    output_path: Path
    bbox: Bbox
    margin_km: int
    s3_bucket: str = S3_DEM_BUCKET
    s3_key: str = S3_LONLAT_VRT_KEY


def margin_km_to_deg(margin_km: float) -> float:
    """Convert margin in kilometers to degrees latitude."""
    return margin_km / 111.0  # Approximate conversion


def margin_km_to_longitude_deg(margin_km: float, lat: float) -> float:
    """Convert margin in kilometers to degrees longitude at given latitude.

    Parameters
    ----------
    margin_km : float
        Margin distance in kilometers
    lat : float
        Latitude at which to calculate the longitude degree equivalent

    Returns
    -------
    float
        Margin in longitude degrees

    """
    return margin_km / (111.0 * np.cos(np.radians(lat)))


def check_dateline(poly: Polygon) -> list[Polygon]:
    """Split polygon if it crosses the dateline (180° longitude).

    Parameters
    ----------
    poly : Polygon
        Input polygon to check for dateline crossing

    Returns
    -------
    list[Polygon]
        list containing either the original polygon or two split polygons

    """
    x_min, _, x_max, _ = poly.bounds

    if not ((x_max - x_min > 180.0) or (x_min <= 180.0 <= x_max)):
        return [poly]

    dateline = loads("LINESTRING(180.0 -90.0, 180.0 90.0)")

    # Build new polygon with longitudes between 0 and 360
    x, y = poly.exterior.coords.xy
    new_x = [k + (360 if k <= 0 else 0) for k in x]
    new_ring = type(poly.exterior)(zip(new_x, y))

    # Split at dateline
    merged = linemerge([dateline, new_ring])
    border = unary_union(merged)
    polys = list(polygonize(border))

    # Normalize coordinates
    for idx, split_poly in enumerate(polys):
        x, y = split_poly.exterior.coords.xy
        if any(k > 180 for k in x):
            x_wrapped = np.array(x) - 360
            polys[idx] = Polygon(zip(x_wrapped, y))

    return polys


def polygon_from_bounding_box(bbox: Bbox, margin_km: float) -> Polygon:
    """Create a polygon from a bounding box with added margin.

    Parameters
    ----------
    bbox : Bbox
        Bounding box coordinates [West, South, East, North] in decimal degrees
    margin_km : float
        Margin to add in kilometers

    Returns
    -------
    Polygon
        Shapely polygon representing the bounded area with margin

    """
    lon_min, lat_min, lon_max, lat_max = bbox
    lat_worst_case = max(lat_min, lat_max)

    lat_margin = margin_km_to_deg(margin_km)
    lon_margin = margin_km_to_longitude_deg(margin_km, lat_worst_case)

    if lon_max - lon_min > 180:
        lon_min, lon_max = lon_max, lon_min

    return box(
        lon_min - lon_margin,
        max(lat_min - lat_margin, -90),
        lon_max + lon_margin,
        min(lat_max + lat_margin, 90),
    )


@backoff.on_exception(backoff.expo, Exception, max_time=600, max_value=32)
def translate_dem(vrt_path: str, output_path: str, bounds: Bbox) -> None:
    """Translate a DEM from S3 to match specified boundaries.

    Parameters
    ----------
    vrt_path : str
        Path to input VRT file
    output_path : str
        Path for output GeoTIFF
    bounds : Tuple[float, float, float, float]
        Boundary coordinates (x_min, x_max, y_min, y_max)

    """
    x_min, x_max, y_min, y_max = bounds
    logger.info(f"Translating DEM from {vrt_path} to {output_path} over {bounds}")

    ds = gdal.Open(vrt_path, gdal.GA_ReadOnly)
    input_x_min, xres, _, input_y_max, _, yres = ds.GetGeoTransform()

    # Snap coordinates to DEM grid
    def snap(val, res, offset, round_func):
        return round_func(float(val - offset) / res) * res + offset

    snapped_bounds = (
        snap(x_min, xres, input_x_min, np.floor),
        snap(x_max, xres, input_x_min, np.ceil),
        snap(y_min, yres, input_y_max, np.floor),
        snap(y_max, yres, input_y_max, np.ceil),
    )

    try:
        gdal.Translate(
            output_path,
            ds,
            format="GTiff",
            projWin=[
                snapped_bounds[0],
                snapped_bounds[3],
                snapped_bounds[2],
                snapped_bounds[1],
            ],
        )
    except RuntimeError as err:
        if "negative width and/or height" in str(err):
            logger.warning("Using original bounds due to negative dimensions")
            gdal.Translate(
                output_path, ds, format="GTiff", projWin=[x_min, y_max, x_max, y_min]
            )
        else:
            raise

    # Handle dateline crossing
    sr = osr.SpatialReference(ds.GetProjection())
    if x_min <= -180.0 and sr.GetAttrValue("AUTHORITY", 1) == "4326":
        ds = gdal.Open(output_path, gdal.GA_Update)
        geotransform = list(ds.GetGeoTransform())
        geotransform[0] += 360.0
        ds.SetGeoTransform(tuple(geotransform))


def download_dem(config: DEMConfig, polygons: list[Polygon]) -> None:
    """Download DEM data for specified polygons.

    Parameters
    ----------
    config : DEMConfig
        Configuration for DEM download
    polygons : list[Polygon]
        list of polygons defining areas to download

    """
    file_prefix = config.output_path.with_suffix("")
    dem_files = []

    for idx, poly in enumerate(polygons):
        vrt_path = f"/vsis3/{config.s3_bucket}/{config.s3_key}"
        output_path = f"{file_prefix}_{idx}.tif"
        dem_files.append(output_path)

        bounds = poly.bounds
        translate_dem(vrt_path, output_path, bounds)

    gdal.BuildVRT(str(config.output_path), dem_files)

    # Cleanup intermediate files
    for file in dem_files:
        Path(file).unlink()


def stage_dem(
    output: Path,
    bbox: Bbox,
    margin: int = 5,
    s3_bucket: str = S3_DEM_BUCKET,
    s3_key: str = S3_LONLAT_VRT_KEY,
    debug: bool = False,
) -> None:
    """Stage a DEM for local processing."""
    if not output.suffix.lower() == ".vrt":
        raise ValueError("Output must have .vrt extension")

    if debug:
        logger.setLevel(logging.DEBUG)

    config = DEMConfig(
        output_path=output,
        bbox=bbox,
        margin_km=margin,
        s3_bucket=s3_bucket,
        s3_key=s3_key,
    )

    # Create polygon and check dateline crossing
    poly = polygon_from_bounding_box(config.bbox, config.margin_km)
    polygons = check_dateline(poly)

    # Download and process DEM
    download_dem(config, polygons)
    logger.info(f"DEM staged successfully at {output}")
