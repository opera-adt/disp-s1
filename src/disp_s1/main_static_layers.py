from __future__ import annotations

import logging
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
