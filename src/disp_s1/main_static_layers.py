from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import opera_utils.geometry
import rasterio as rio
from dolphin import Bbox, PathOrStr
from dolphin._log import log_runtime, setup_logging
from dolphin.utils import get_max_memory_usage
from opera_utils.geometry import Layer
from osgeo import gdal

from disp_s1 import __version__
from disp_s1.pge_runconfig import StaticLayersRunConfig

logger = logging.getLogger(__name__)

gdal.UseExceptions()

__all__ = ["run_static_layers"]


class StaticLayersOutputs(NamedTuple):
    los_combined_path: Path
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

    _geom_files = opera_utils.geometry.stitch_geometry_layers(
        local_hdf5_files=pge_runconfig.dynamic_ancillary_file_group.geometry_files,
        layers=[Layer.LOS_EAST, Layer.LOS_NORTH, Layer.LAYOVER_SHADOW_MASK],
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

    return StaticLayersOutputs(
        los_combined_path=los_combined_path,
        dem_path=dem_path,
    )


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

    warp_options = gdal.WarpOptions(
        dstSRS=f"EPSG:{epsg}",
        outputBounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
        resampleAlg=gdal.GRA_Cubic,
        format="GTiff",
        srcNodata=None,
        dstNodata=None,
    )

    gdal.Warp(str(output_path), str(dem_file), options=warp_options)

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

        profile.update(
            dtype=rio.float32,
            count=1,
            compress="deflate",
            zlevel=4,
            tiled=True,
            blockxsize=128,
            blockysize=128,
            predictor=2,
        )

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
            dst.write(src_east.read(1), 1)
            dst.set_band_description(1, desc_base.format("East"))

            dst.write(src_north.read(1), 2)
            dst.set_band_description(2, desc_base.format("North"))

            dst.write(src_up.read(1), 3)
            dst.set_band_description(3, desc_base.format("Up"))

    return combined_los_path
