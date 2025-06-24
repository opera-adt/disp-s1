from __future__ import annotations

import logging
from datetime import datetime, timezone
from os import fsdecode
from pathlib import Path
from typing import NamedTuple

import isce3
import numpy as np
import opera_utils.download
import opera_utils.geometry
import rasterio as rio
from dolphin import Bbox, PathOrStr, io, stitching
from dolphin._log import log_runtime, setup_logging
from dolphin.utils import get_max_memory_usage, numpy_to_gdal_type
from numpy.typing import DTypeLike
from opera_utils.geometry import Layer
from osgeo import gdal

from disp_s1 import __version__
from disp_s1.pge_runconfig import StaticLayersRunConfig

logger = logging.getLogger(__name__)

gdal.UseExceptions()

__all__ = ["run_static_layers"]


# Constants (adapt these to match your actual constants)
DATE_TIME_METADATA_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
PRODUCT_SPECIFICATION_VERSION = "1.0.0"  # Update as needed
SOFTWARE_VERSION = __version__

DEM_METADATA = [
    (
        "processing_information_dem_interpolation_algorithm",
        "biquintic",
        "DEM interpolation method",
    ),
    (
        "processing_information_dem_egm_model",
        "Earth Gravitational Model 2008 (EGM2008)",
        "Earth Gravitational Model associated with the DEM",
    ),
    (
        "input_dem_source",
        "Copernicus GLO-30 DEM for OPERA",
        "Description of the input digital elevation model (DEM)",
    ),
]
MASK_DESCRIPTION = (
    "Mask Layer. Values: 0: not masked; 1: shadow; 2: layover; 3: layover and shadow;"
    " 255: invalid/fill value"
)


class StaticLayersOutputs(NamedTuple):
    los_combined_path: Path
    layover_shadow_mask_path: Path
    dem_path: Path


@log_runtime
def run_static_layers(
    pge_runconfig: StaticLayersRunConfig,
) -> StaticLayersOutputs:
    """Run the Static Layers workflow for one of the DISP-S1 frames.

    Parameters
    ----------
    pge_runconfig : RunConfig
        PGE-specific metadata for the output product.

    Returns
    -------
    StaticLayersOutputs
        Paths to the output files.

    """
    # Record processing start time
    processing_start_datetime = datetime.now(tz=timezone.utc)

    setup_logging(logger_name="disp_s1", filename=pge_runconfig.log_file)
    epsg, bounds = opera_utils.get_frame_bbox(
        pge_runconfig.input_file_group.frame_id,
        json_file=pge_runconfig.static_ancillary_file_group.frame_to_burst_json,
    )
    logger.info("Creating geometry layers Workflow")
    output_dir = pge_runconfig.product_path_group.scratch_path

    layover_shadow_mask_path = Path(output_dir) / "layover_shadow_mask.tif"
    logger.info("Stitching RTC layover shadow mask files")
    mask_options = ["COMPRESS=deflate", "TILED=yes", "PREDICTOR=1"]
    stitching.merge_images(
        file_list=pge_runconfig.dynamic_ancillary_file_group.rtc_static_layers_files,
        outfile=layover_shadow_mask_path,
        driver="GTIff",
        resample_alg="nearest",
        in_nodata=255,
        out_nodata=255,
        out_bounds=bounds,
        out_bounds_epsg=epsg,
        options=mask_options,
    )
    io.set_raster_description(layover_shadow_mask_path, description=MASK_DESCRIPTION)

    logger.info("Stitching CSLC line of sight geometry files")
    _geom_files = opera_utils.geometry.stitch_geometry_layers(
        local_hdf5_files=pge_runconfig.dynamic_ancillary_file_group.geometry_files,
        layers=[Layer.LOS_EAST, Layer.LOS_NORTH],
        strides={"x": 6, "y": 3},
        output_dir=output_dir,
        out_bounds=bounds,
        out_bounds_epsg=epsg,
    )

    los_east_path, los_north_path, los_up_path = _make_los_up(output_dir)
    los_combined_path = _make_3band_los(los_east_path, los_north_path, los_up_path)

    # Delete old single-band TIFFs
    Path(los_east_path).unlink()
    Path(los_north_path).unlink()
    Path(los_up_path).unlink()

    logger.info("Warping DEM to match UTM frame boundary")
    dem_path = warp_dem_to_utm(
        pge_runconfig.dynamic_ancillary_file_group.dem_file,
        epsg,
        bounds,
        output_dir=output_dir,
    )

    logger.info(f"Product type: {pge_runconfig.primary_executable.product_type}")
    logger.info(f"Product version: {pge_runconfig.product_path_group.product_version}")
    max_mem = get_max_memory_usage(units="GB")
    logger.info(f"Maximum memory usage: {max_mem:.2f} GB")
    logger.info(f"Current running disp_s1 version: {__version__}")

    static_layers_paths = StaticLayersOutputs(
        los_combined_path=los_combined_path,
        dem_path=dem_path,
        layover_shadow_mask_path=layover_shadow_mask_path,
    )
    create_outputs(
        static_layers_paths=static_layers_paths,
        output_dir=pge_runconfig.product_path_group.output_directory,
    )
    # Create metadata for the products
    add_product_metadata(
        static_layers_paths=static_layers_paths,
        pge_runconfig=pge_runconfig,
        frame_id=pge_runconfig.input_file_group.frame_id,
        processing_datetime=processing_start_datetime,
    )

    return static_layers_paths


def create_outputs(static_layers_paths: StaticLayersOutputs, output_dir: Path):
    """Create formatted geotiffs in `output_dir`."""
    # TODO: Take metadata from RTC, DISP-S1, etc.
    import shutil

    from dolphin._overviews import create_overviews

    from disp_s1.browse_image import make_browse_image_from_arr

    output_dir.mkdir(exist_ok=True, parents=True)

    create_overviews(
        file_paths=static_layers_paths,
        levels=[4, 8, 16, 32, 64],
        resampling="nearest",
    )

    # Add metadata for layover shadow mask

    arr = io.load_gdal(static_layers_paths[0], masked=True)
    make_browse_image_from_arr(
        output_filename=output_dir / "los_enu.browse.png",
        arr=arr[-1],
        vmin=0.5,
        vmax=1,
        cmap="gray",
        # Mask should be 0 for bad, 1 for good, so flip Numpy convention
        mask=(~arr[0].mask).astype(int),
    )
    for path in static_layers_paths:
        _new_path = shutil.move(path, output_dir)


def warp_dem_to_utm(
    dem_file: PathOrStr, epsg: int, bounds: Bbox, output_dir: Path
) -> Path:
    """Warp a lat/lon EPSG:4326 DEM into a UTM grid using cubic interpolation.

    Parameters
    ----------
    dem_file : PathOrStr
        Path to the input DEM file in EPSG:4326 projection.
    epsg : int
        EPSG code for the target UTM projection.
    bounds : Bbox
        Bounding box of the area of interest in the target UTM projection.
    output_dir : Path
        Directory where the output warped DEM will be saved.

    Returns
    -------
    Path
        Path to the output warped DEM file.

    Notes
    -----
    This function uses GDAL's Warp functionality to perform the reprojection
    and resampling of the input DEM.

    """
    from osgeo import gdal

    output_path = output_dir / "dem_warped_utm.tif"
    temp_path = output_path.with_suffix(".temp.tif")

    left, bottom, right, top = bounds
    width = int(np.round((right - left) / 30))
    length = int(np.round((top - bottom) / 30))
    in_raster = isce3.io.Raster(fsdecode(dem_file))
    out_raster = create_single_band_gtiff(temp_path, (length, width), "float32")
    geo_grid = isce3.product.GeoGridParameters(
        start_x=left,
        start_y=top,
        spacing_x=30,
        spacing_y=-30,
        width=width,
        length=length,
        epsg=epsg,
    )
    logger.info("Warping DEM with isce3")
    isce3.geogrid.relocate_raster(in_raster, geo_grid, out_raster)
    out_raster.close_dataset()
    del out_raster

    # Recompress afterward
    gdal.Translate(
        fsdecode(output_path),
        fsdecode(temp_path),
        creationOptions=list(opera_utils.geometry.EXTRA_COMPRESSED_TIFF_OPTIONS),
    )
    io.set_raster_units(output_path, "meters")

    return output_path


def _make_los_up(output_path: Path) -> tuple[Path, Path, Path]:
    """Create los_up.tif from east/north."""
    los_east_path = output_path / "los_east.tif"
    los_north_path = output_path / "los_north.tif"
    los_up_path = output_path / "los_up.tif"

    with rio.open(los_east_path) as src_east, rio.open(los_north_path) as src_north:
        profile = src_east.profile
        los_east = src_east.read(1)
        los_north = src_north.read(1)

        los_up = np.sqrt(1 - los_east**2 - los_north**2)
        los_up[los_east == 0] = 0

        profile.update(dtype=rio.float32, count=1, compress="deflate", tiled=True)
        with rio.open(los_up_path, "w", **profile) as dst:
            dst.write(los_up.astype(rio.float32), 1)

    return los_east_path, los_north_path, los_up_path


def _make_3band_los(
    los_east_path: Path, los_north_path: Path, los_up_path: Path
) -> Path:
    """Combine the 3 TIFFs into one, 3-band TIFF."""
    output_path = los_east_path.parent
    combined_los_path = output_path / "los_enu.tif"

    with (
        rio.open(los_east_path) as src_east,
        rio.open(los_north_path) as src_north,
        rio.open(los_up_path) as src_up,
    ):
        profile = src_east.profile
        profile.update(
            count=3,
            compress="deflate",
            zlevel=4,
            tiled=True,
            blockxsize=128,
            blockysize=128,
            predictor=2,
            # We will usually want E,N, and U components,
            # rather than an entire image of just one component
            interleave="pixel",
        )
        desc_base = "{} component of line of sight unit vector (ground to satellite)"
        with rio.open(combined_los_path, "w", **profile) as dst:
            arr = src_east.read(1)
            io.round_mantissa(arr, keep_bits=9)
            dst.write(arr, 1)
            dst.set_band_description(1, desc_base.format("East"))

            arr = src_north.read(1)
            io.round_mantissa(arr, keep_bits=9)
            dst.write(arr, 2)
            dst.set_band_description(2, desc_base.format("North"))

            arr = src_up.read(1)
            io.round_mantissa(arr, keep_bits=9)
            dst.write(arr, 3)
            dst.set_band_description(3, desc_base.format("Vertical"))

    return combined_los_path


def create_single_band_gtiff(
    path: PathOrStr,
    shape: tuple[int, int],
    dtype: DTypeLike,
) -> isce3.io.Raster:
    """Create a single-band GeoTIFF `isce3.io.Raster`.

    Parameters
    ----------
    path : PathOrStr
        Path where the GeoTIFF will be created.
    shape : tuple[int, int]
        The (length, width) dimensions of the raster.
    dtype : DTypeLike
        NumPy data type of the raster. Will be converted to corresponding GDAL type.

    Returns
    -------
    isce3.io.Raster
        Newly created ISCE3 Raster object pointing to the file

    """
    gdal_dtype = numpy_to_gdal_type(dtype)
    length, width = shape
    return isce3.io.Raster(
        path=fsdecode(path),
        width=width,
        length=length,
        num_bands=1,
        dtype=gdal_dtype,
        driver_name="GTiff",
    )


def get_static_layers_metadata(
    pge_runconfig: StaticLayersRunConfig,
    processing_datetime: datetime,
    frame_id: int,
) -> list[tuple[str, str, str]]:
    """Create static layers metadata dictionary.

    Parameters
    ----------
    pge_runconfig : StaticLayersRunConfig
        PGE runconfig containing processing parameters
    processing_datetime : datetime
        Processing datetime object
    frame_id : int
        Frame ID for this product

    Returns
    -------
    list[tuple[str, str, str]]
        Metadata dict organized for static layers product

    """
    # Product type
    # Product version
    product_version = pge_runconfig.product_path_group.product_version

    # Data access URLs
    product_data_access = (
        "https://search.asf.alaska.edu/#/?dataset=OPERA-S1&productTypes=DISP-S1-STATIC"
    )
    source_data_access = "https://search.asf.alaska.edu/#/?dataset=OPERA-S1&productTypes=CSLC-S1-STATIC,RTC-S1-STATIC"

    common_metadata = [
        ("platform", "Sentinel-1", "Platform name"),
        (
            "instrument_name",
            "Sentinel-1 CSAR",
            "Name of the instrument used to collect the remote sensing data",
        ),
        (
            "processing_facility",
            "NASA Jet Propulsion Laboratory on AWS",
            "Product processing facility",
        ),
        ("project", "OPERA", "Project name"),
        ("institution", "NASA JPL", "Institution that created this product"),
        (
            "contact_information",
            "opera-sds-ops@jpl.nasa.gov",
            "Contact information for producer of this product",
        ),
        ("product_version", str(product_version), "Product version"),
        ("frame_id", str(frame_id), "Frame identification"),
        ("acquisition_mode", "IW", "Acquisition mode"),
        ("look_direction", "right", "Look direction can be left or right"),
        (
            "processing_datetime",
            processing_datetime.strftime(DATE_TIME_METADATA_FORMAT),
            "Processing date and time",
        ),
        (
            "product_data_access",
            product_data_access,
            "Location from where this product can be retrieved",
        ),
        (
            "source_data_access",
            source_data_access,
            "Location from where the source data can be retrieved",
        ),
        (
            "source_data_institution",
            "ESA",
            "Institution that created the source data product",
        ),
        ("software_version", str(SOFTWARE_VERSION), "Software version"),
    ]
    return common_metadata


def metadata_items_to_geotiff_metadata_dict(metadata_items: list[tuple[str, str, str]]):
    """Convert metadata dict to GeoTIFF metadata dict.

    Parameters
    ----------
    metadata_items : list[tuple[str, str, str]]
        Metadata dict with GeoTIFF keys, values, and descriptions

    Returns
    -------
    dict
        Metadata dict formatted for GeoTIFF files

    """
    geotiff_metadata_dict = {}
    for key, value, description in metadata_items:
        geotiff_metadata_dict[key.upper()] = value
        geotiff_metadata_dict[key.upper() + "_DESCRIPTION"] = description

    return geotiff_metadata_dict


def add_product_metadata(
    static_layers_paths: StaticLayersOutputs,
    pge_runconfig: StaticLayersRunConfig,
    frame_id: int,
    processing_datetime: datetime,
):
    """Create and add metadata to all static layers products.

    Parameters
    ----------
    static_layers_paths : StaticLayersOutputs
        Paths to the static layers output files
    pge_runconfig : StaticLayersRunConfig
        PGE runconfig
    frame_id : int
        DISP-S1 Frame ID
    processing_datetime : datetime
        Processing datetime

    """
    logger.info("Creating metadata for static layers products")

    metadata_items = get_static_layers_metadata(
        pge_runconfig=pge_runconfig,
        processing_datetime=processing_datetime,
        frame_id=frame_id,
    )

    # Convert to GeoTIFF format
    geotiff_metadata = metadata_items_to_geotiff_metadata_dict(metadata_items)

    # Add common metadata to each file
    for file_path in static_layers_paths:
        io.set_raster_metadata(file_path, geotiff_metadata, domain="")

    # Add DEM metadata
    dem_metadata_dict = metadata_items_to_geotiff_metadata_dict(DEM_METADATA)
    io.set_raster_metadata(static_layers_paths.dem_path, dem_metadata_dict, domain="")
